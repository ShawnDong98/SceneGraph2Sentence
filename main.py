import os
import json
import random
from anyio import LockStatistics

import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.losses import sparse_categorical_crossentropy
from datasets import COCOTrainDataset, COCOTestDataset, COCOCacheDataset, MyBatchLoader
from spektral.data import BatchLoader
from spektral.transforms import GCNFilter
from models import MyGNN, TransformerDecoderBlock, TransformerDecoder, PositionalEmbedding
from metric import BleuScore

import numpy as np
from tqdm import tqdm


seed = 999
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.config.experimental.enable_op_determinism()
os.environ['PYTHONHASHSEED'] = str(seed)


source_data_path = "test.json"
graph_idx2sen_path = "graph_idx2sen_test.json"
vocab_path = "tv_layer.pkl"

train_dataset = COCOTrainDataset(
    source_data_path="train.json",
    vocab_path = vocab_path,
    graph_idx2sen_path = graph_idx2sen_path,
)

test_dataset = COCOTestDataset(
    source_data_path="test.json",
    vocab_path = vocab_path,
    graph_idx2sen_path = graph_idx2sen_path,
)

cache_dataset = COCOCacheDataset(
    source_data_path,
    vocab_path = vocab_path,
    graph_idx2sen_path = graph_idx2sen_path,
)

train_dataset.apply(GCNFilter())
test_dataset.apply(GCNFilter())
cache_dataset.apply(GCNFilter())

vocab_size = len(cache_dataset.vocab.get_vocabulary())


train_loader = MyBatchLoader(train_dataset, batch_size=512)
val_loader = MyBatchLoader(test_dataset, batch_size=512)



# sample = next(loader)
from nltk.translate.bleu_score import sentence_bleu


class SGSC(keras.Model):
    def __init__(self, 
                num_layers,
                **kwargs):
        super().__init__(**kwargs)
        self.encoder = MyGNN(n_hidden=768)
        self.decoder = TransformerDecoder(
                            embed_dim=512,
                            dense_dim=2048,
                            num_heads=4,
                            num_layers=num_layers,
                            sequence_length=32,
                            vocab_size=vocab_size,
                            )
        self.classifier = layers.Dense(vocab_size, activation="softmax")
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')


    def compute_loss(self, x, y, y_pred, sample_weight):
        (_, target) = y
        target = tf.cast(target, tf.float32)
        logits = y_pred
        loss = tf.reduce_mean(sparse_categorical_crossentropy(target, logits))
        self.loss_tracker.update_state(loss)
        return loss

    def compute_metrics(self, x, y, y_pred, sample_weight):
        loss = self.loss_tracker.result()
        return {"loss":  loss}


    def call(self, inputs):
        (X, A), dec_input = inputs
        enc_out = self.encoder((X, A))
        enc_out = tf.expand_dims(enc_out, axis=1)
        enc_out = tf.tile(enc_out, [1, 31, 1])

        dec_out = self.decoder(dec_input, enc_out)
        
        logits = self.classifier(dec_out)

        
        return logits

    def custom_evaluate(self,
                loader,
                index_lookup,
                graph_idx2sen,
                max_seq_len=32,
                
    ):
        
        out = []
        idx = []
        total_dataset = len(graph_idx2sen)
        for inputs, y in tqdm(loader.load(), total=total_dataset//loader.batch_size):
            X, A = inputs[0]
            idx.append(y[0])
            batch, = y[0].shape
            tokenized_decoded_sentence = tf.Variable(tf.zeros(
                shape=(batch, max_seq_len),
                dtype=tf.int64,
            ))
            tokenized_decoded_sentence = tokenized_decoded_sentence[:, 0].assign([4] * batch)
            for i in range(max_seq_len-1):
                inputs = ((X, A), tokenized_decoded_sentence[:, :-1])
                logits = self.call(inputs)
                prediction = tf.math.argmax(logits, axis=2)
                
                tokenized_decoded_sentence = tokenized_decoded_sentence[:, i+1].assign(prediction[:, i])
            out.append(tokenized_decoded_sentence)

        out = np.concatenate(out, axis=0)
        idx = np.concatenate(idx, axis=0)
        bleu1_score = 0
        bleu2_score = 0
        bleu3_score = 0
        bleu4_score = 0
        count_th_05 = 0
        for idx, pred in zip(idx, out):
            graph = graph_idx2sen[str(idx)]['graph']
            sentence = graph_idx2sen[str(idx)]['sentence']
            pred = ' '.join([index_lookup[token] for token in pred])
            
            bleu1 = sentence_bleu([sentence.split()], pred.strip().split()[1:-1],
            weights=(1, 0, 0, 0))
            bleu1_score += bleu1 / total_dataset
            bleu2 = sentence_bleu([sentence], pred,
            weights=(0, 1, 0, 0))
            bleu2_score += bleu2 / total_dataset
            bleu3 = sentence_bleu([sentence], pred,
            weights=(0, 0, 1, 0))
            bleu3_score += bleu3 / total_dataset
            bleu4 = sentence_bleu([sentence], pred,
            weights=(0, 0, 0, 1))
            bleu4_score += bleu4 / total_dataset
            if bleu1 < 0.5:
                count_th_05 += 1
                print("graph: ", graph)
                print("sentence: ", sentence)
                print("pred: ", pred)
        print(count_th_05)
        return [bleu1_score, bleu2_score, bleu3_score, bleu4_score]
        
    
vocab = cache_dataset.vocab.get_vocabulary()
index_lookup = dict(zip(range(len(vocab)), vocab))

def decode_sequence(model, inputs, max_length=30):
    def sample_next(predictions, temperature=1.0):
        predictions = np.asarray(predictions).astype("float64") 
        predictions = np.log(predictions) / temperature 
        exp_preds = np.exp(predictions)
        predictions = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, predictions, 1) 
        return np.argmax(probas)
    X, A = inputs[0]
    decoded_sentence = "[start]"
    for i in range(max_length):
        tokenized_decoded_sentence = cache_dataset.vocab([decoded_sentence])[:, :-1]
        print(tokenized_decoded_sentence)
        inputs = ((X, A), tokenized_decoded_sentence)
        predictions = model(inputs)
        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = index_lookup[sampled_token_index]        
        decoded_sentence += " " + sampled_token
        if sampled_token == "[end]":
            break

    return decoded_sentence


model = SGSC(num_layers=3)
model.compile(
    optimizer = keras.optimizers.Adam(learning_rate=1e-4),
    # run_eagerly=True
)

checkpoint_filepath = './checkpoints/SGSC'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)


history = model.fit(
    train_loader.load(),
    validation_data=val_loader.load(),
    steps_per_epoch=train_loader.steps_per_epoch, 
    validation_steps=val_loader.steps_per_epoch,
    epochs=20,
    callbacks=[
        model_checkpoint_callback,
        earlystopping_callback,
    ]
)

model.load_weights(checkpoint_filepath)


with open("graph_idx2sen_test.json", "r") as f:
    graph_idx2sen = json.load(f)


test_loader = MyBatchLoader(test_dataset, batch_size=512, epochs=1)

print(model.custom_evaluate(
    test_loader, 
    index_lookup=index_lookup,
    graph_idx2sen=graph_idx2sen
))


# bleu_score_1gram = BleuScore(1, 0, 0, 0)
# bleu_score_2gram = BleuScore(0, 1, 0, 0)
# bleu_score_3gram = BleuScore(0, 0, 1, 0)
# bleu_score_4gram = BleuScore(0, 0, 0, 1)
# Tx_word = []
# Rx_word = []

# for inputs, y in tqdm(test_loader.load(), total=len(cache_dataset)):
#     gen_text = decode_sequence(model, inputs)
#     # print(graph_idx2sen[str(y[0][0])]['graph'])
#     # print(graph_idx2sen[str(y[0][0])]['sentence'])
#     # print(gen_text)
#     TX = graph_idx2sen[str(y[0][0])]['sentence'].split()
#     RX = gen_text.split()[1:-1]
#     Tx_word.append(TX)
#     Rx_word.append(RX)

# print("bleu1: ", np.mean(bleu_score_1gram.compute_bleu_score(Tx_word, Rx_word)))
# print("bleu2: ", np.mean(bleu_score_2gram.compute_bleu_score(Tx_word, Rx_word)))
# print("bleu3: ", np.mean(bleu_score_3gram.compute_bleu_score(Tx_word, Rx_word)))
# print("bleu4: ", np.mean(bleu_score_4gram.compute_bleu_score(Tx_word, Rx_word)))

