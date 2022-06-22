from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from spektral.layers import GCNConv, GlobalSumPool

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class MyGNN(Model):
    def __init__(self, n_hidden):
        super().__init__()
        self.graph_conv = GCNConv(n_hidden)
        self.pool = GlobalSumPool()
        self.dropout = Dropout(0.5)
        # self.dense = Dense(n_labels, 'softmax')

    def call(self, inputs):
        out = self.graph_conv(inputs)
        out = self.dropout(out)
        out = self.pool(out)
        # out = self.dense(out)

        return out


class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=input_dim, output_dim=output_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=output_dim
        )
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim
        })
        return config


class TransformerEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim)
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask
        )
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim
        })
        return config

class TransformerEncoder(layers.Layer):
    def __init__(self,
                embed_dim,
                dense_dim,
                num_heads,
                num_layers,
                sequence_length,
                vocab_size,

    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        self.Pos_embedding = PositionalEmbedding(
            sequence_length = sequence_length,
            input_dim = vocab_size,
            output_dim = embed_dim
        )
        self.layers = [TransformerEncoderBlock(
                        embed_dim = embed_dim,
                        dense_dim = dense_dim,
                        num_heads = num_heads,
                    ) for _ in range(num_layers)]
    
    def call(self, inputs, mask=None):
        x = self.Pos_embedding(inputs)
        for layer in self.layers:
            x = layer(x)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size
        })
        return config




class TransformerDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.supports_masking = True

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim
        })
        return config

    def get_casual_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), # (1, )
            tf.constant([1, 1], dtype=tf.int32)],  # (2, )
            axis = 0
        )
        # 在 mult 维度上复制几次: (batch_size, 1, 1)
        return tf.tile(mask, mult) # (3, )
    def call(self, inputs, encoder_outputs, mask=None):
        """
        params: inputs 
        params: encoder_outputs - shape(batch_size, seq_len, embed_dim)
        """
        casual_mask = self.get_casual_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(
                mask[:, tf.newaxis, :], dtype="int32"
            )
            padding_mask = tf.minimum(padding_mask, casual_mask)
        attention_output_1 = self.attention_1(
            query=inputs,
            value=inputs,
            key=inputs,
            attention_mask=casual_mask
        )
        out = inputs + attention_output_1
        attention_output_1 = self.layernorm_1(out)
        attention_output_2 = self.attention_2(
            query=attention_output_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask=padding_mask,
        )
        attention_output_2 = self.layernorm_2(
            attention_output_1 + attention_output_2
        )
        proj_output = self.dense_proj(attention_output_2)
        return self.layernorm_3(attention_output_2 + proj_output)
        

class TransformerDecoder(layers.Layer):
    def __init__(self,
                embed_dim,
                dense_dim,
                num_heads,
                num_layers,
                sequence_length,
                vocab_size,

    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size

        self.Pos_embedding = PositionalEmbedding(
            sequence_length = sequence_length,
            input_dim = vocab_size,
            output_dim = embed_dim
        )
        self.layers = [TransformerDecoderBlock(
                        embed_dim = embed_dim,
                        dense_dim = dense_dim,
                        num_heads = num_heads,
                    ) for _ in range(num_layers)]
    
    def call(self, inputs, enc_out, mask=None):
        x = self.Pos_embedding(inputs)
        for layer in self.layers:
            x = layer(x, enc_out)
        
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
            "num_layers": self.num_layers,
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size
        })
        return config

