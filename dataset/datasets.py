import os
import json
from collections import defaultdict

from spektral.data import Dataset, Graph
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import networkx as nx
import sng_parser
from pprint import pprint
import numpy as np

import pickle
from glob import glob


from spektral.data.utils import (
    collate_labels_disjoint,
    get_spec,
    prepend_none,
    sp_matrices_to_sp_tensors,
    to_batch,
    to_disjoint,
    to_mixed,
    to_tf_signature,
)

def shuffle_inplace(*args):
    rng_state = np.random.get_state()
    for a in args:
        np.random.set_state(rng_state)
        np.random.shuffle(a)


def batch_generator(data, batch_size=32, epochs=None, shuffle=True):
    if not isinstance(data, (list, tuple)):
        data = [data]
    if len(data) < 1:
        raise ValueError("data cannot be empty")
    if len({len(item) for item in data}) > 1:
        raise ValueError("All inputs must have the same __len__")

    if epochs is None or epochs == -1:
        epochs = np.inf
    len_data = len(data[0])
    batches_per_epoch = int(np.ceil(len_data / batch_size))
    epoch = 0
    while epoch < epochs:
        epoch += 1
        if shuffle:
            shuffle_inplace(*data)
        for batch in range(batches_per_epoch):
            start = batch * batch_size
            stop = min(start + batch_size, len_data)
            to_yield = [item[start:stop] for item in data]
            if len(data) == 1:
                to_yield = to_yield[0]

            yield to_yield

def pad_jagged_array(x, target_shape):
    if len(x) < 1:
        raise ValueError("Jagged array cannot be empty")
    target_len = len(x)
    target_shape = tuple(
        shp if shp != -1 else x[0].shape[j] for j, shp in enumerate(target_shape)
    )
    output = np.zeros((target_len,) + target_shape, dtype=x[0].dtype)
    for i in range(target_len):
        slc = (i,) + tuple(slice(shp) for shp in x[i].shape)
        output[slc] = x[i]

    return output


def collate_labels_batch(y_list, node_level=False):
    if node_level:
        n_max = max([x.shape[0] for x in y_list])
        return pad_jagged_array(y_list, (n_max, -1))
    else:
        graph_idx = []
        tokens = []
        for _ in y_list: 
            _ = _.item()
            graph_idx.append(_['graph_idx'])
            tokens.append(_['tokens'])
        graph_idx = np.array(graph_idx)
        tokens = np.array(tokens)
        return {'graph_idx': graph_idx, 'tokens': tokens}

class Loader:
    def __init__(self, dataset, batch_size=1, epochs=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.shuffle = shuffle
        self._generator = self.generator()

    def __iter__(self):
        return self

    def __next__(self):
        nxt = self._generator.__next__()
        return self.collate(nxt)

    def generator(self):
        """
        Returns lists (batches) of `Graph` objects.
        """
        return batch_generator(
            self.dataset,
            batch_size=self.batch_size,
            epochs=self.epochs,
            shuffle=self.shuffle,
        )
    def collate(self, batch):
        """
        Converts a list of graph objects to Tensors or np.arrays representing the batch.
        :param batch: a list of `Graph` objects.
        """
        raise NotImplementedError

    def load(self):
        """
        Returns an object that can be passed to a Keras model when using the `fit`,
        `predict` and `evaluate` functions.
        By default, returns the Loader itself, which is a generator.
        """
        return self

    def tf_signature(self):
        """
        Returns the signature of the collated batches using the tf.TypeSpec system.
        By default, the signature is that of the dataset (`dataset.signature`):
            - Adjacency matrix has shape `[n_nodes, n_nodes]`
            - Node features have shape `[n_nodes, n_node_features]`
            - Edge features have shape `[n_edges, n_node_features]`
            - Targets have shape `[..., n_labels]`
        """
        signature = self.dataset.signature
        return to_tf_signature(signature)

    def pack(self, batch):
        """
        Given a batch of graphs, groups their attributes into separate lists and packs
        them in a dictionary.
        For instance, if a batch has three graphs g1, g2 and g3 with node
        features (x1, x2, x3) and adjacency matrices (a1, a2, a3), this method
        will return a dictionary:
        ```python
        >>> {'a_list': [a1, a2, a3], 'x_list': [x1, x2, x3]}
        ```
        :param batch: a list of `Graph` objects.
        """
        output = [list(elem) for elem in zip(*[g.numpy() for g in batch])]
        keys = [k + "_list" for k in self.dataset.signature.keys()]
        return dict(zip(keys, output))

    @property
    def steps_per_epoch(self):
        """
        :return: the number of batches of size `self.batch_size` in the dataset (i.e.,
        how many batches are in an epoch).
        """
        return int(np.ceil(len(self.dataset) / self.batch_size))

class MyBatchLoader(Loader):
    def __init__(
        self,
        dataset,
        mask=False,
        batch_size=1,
        epochs=None,
        shuffle=True,
        node_level=False,
    ):
        self.mask = mask
        self.node_level = node_level
        self.signature = dataset.signature
        super().__init__(dataset, batch_size=batch_size, epochs=epochs, shuffle=shuffle)

    def collate(self, batch):
        packed = self.pack(batch)

        y = packed.pop("y_list", None)
        if y is not None:
            y = collate_labels_batch(y, node_level=self.node_level)

        output = to_batch(**packed, mask=self.mask)
        output = sp_matrices_to_sp_tensors(output)

        if len(output) == 1:
            output = output[0]

        if y is None:
            return output
        else:
            graph_idx = y['graph_idx']
            tokens = y['tokens']

            return [output,tokens[:, :-1]], [graph_idx, tokens[:, 1:]]
    def tf_signature(self):
        """
        Adjacency matrix has shape [batch, n_nodes, n_nodes]
        Node features have shape [batch, n_nodes, n_node_features]
        Edge features have shape [batch, n_nodes, n_nodes, n_edge_features]
        Labels have shape [batch, n_labels]
        """
        signature = self.signature
        for k in signature:
            signature[k]["shape"] = prepend_none(signature[k]["shape"])
        if "x" in signature and self.mask:
            # In case we have a mask, the mask is concatenated to the features
            signature["x"]["shape"] = signature["x"]["shape"][:-1] + (
                signature["x"]["shape"][-1] + 1,
            )
        if "a" in signature:
            # Adjacency matrix in batch mode is dense
            signature["a"]["spec"] = tf.TensorSpec
        if "e" in signature:
            # Edge attributes have an extra None dimension in batch mode
            signature["e"]["shape"] = prepend_none(signature["e"]["shape"])
        if "y" in signature and self.node_level:
            # Node labels have an extra None dimension
            signature["y"]["shape"] = prepend_none(signature["y"]["shape"])

        return to_tf_signature(signature)


from transformers import DistilBertTokenizer, TFDistilBertModel


class COCOTrainDataset(Dataset):
    def __init__(self, 
                source_data_path, 
                vocab_path = None,
                graph_idx2sen_path = None,
                max_len=32, 
                **kwargs):
       
        self.max_len = max_len
        if isinstance(source_data_path, list): self.source_data_path = source_data_path
        else: self.source_data_path = [source_data_path]
        self.vocab_path = vocab_path
        self.graph_idx2sen_path = graph_idx2sen_path

        from_disk = pickle.load(open(self.vocab_path, "rb"))
        new_v = layers.TextVectorization(
            max_tokens=from_disk['config']['max_tokens'],
            output_mode='int',
            output_sequence_length=from_disk['config']['output_sequence_length'])
        new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        new_v.set_weights(from_disk['weights'])
        self.vocab = new_v
        
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = TFDistilBertModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        super().__init__(**kwargs)

    def download(self):
        os.mkdir(self.path)

        all_data = []
        for data_path in self.source_data_path:
            with open(data_path, "r") as f:
                data = json.load(f)['images']
                all_data.extend(data)

        all_data = all_data
        self.graph_idx2sen = defaultdict(str)
        sentences = [data['sentence'] for data in all_data] 
        processed_sentences = ["[start] " + data['sentence'] + " [end]" for data in all_data] 
        
        for i, (sen, processed_sen) in tqdm(enumerate(zip(sentences, processed_sentences)), total=len(processed_sentences)):
            graph = sng_parser.parse(sen)

            G = nx.DiGraph(name='G')

            for entity in graph['entities']:
                G.add_node(entity['head'])
                for mod in entity['modifiers']:
                    G.add_node(mod['span'])
                    G.add_edge(entity['head'], mod['span'])
                    
            for rela in graph['relations']:
                G.add_node(rela['relation'])
                G.add_edge(graph['entities'][rela['subject']]['head'], rela['relation'])
                G.add_edge(rela['relation'], graph['entities'][rela['object']]['head'])
            try:
                A = np.array(nx.attr_matrix(G)[0])
                X = self.get_bert_embeddings(nx.attr_matrix(G)[1])
                y = {
                    'graph_idx' : i,
                    'tokens' : self.vocab(processed_sen)
                }
                self.graph_idx2sen[i] = {
                    'graph' : nx.attr_matrix(G)[1],
                    'sentence' : sen,
                }
                filename = os.path.join(self.path, f'graph_{i}')
                np.savez(filename, x=X, a=A, y=y)
            except:
                print(sen)
                print(nx.attr_matrix(G)[1])
                continue
        
        with open(self.graph_idx2sen_path, "w") as f:
            json.dump(self.graph_idx2sen, f)

    def get_bert_embeddings(self, tokens):
        node_features = []
        for token in tokens:
            encoded_input = self.tokenizer(token, return_tensors='tf')
            outputs = self.model(encoded_input)
            last_hidden_states = outputs.last_hidden_state
            embeddings = last_hidden_states[:, 0, :].numpy()
            node_features.append(embeddings)
            
        node_features = np.concatenate(node_features, axis=0)
        return node_features 

            
    def read(self):
        output = []
        for graph_path in glob(os.path.join(self.path, "*")):
            data = np.load(graph_path, allow_pickle=True)
        
            output.append(
                Graph(x=data['x'], a=data['a'], y=data['y'])
            )
        return output

class COCOTestDataset(Dataset):
    def __init__(self, 
                source_data_path, 
                vocab_path = None,
                graph_idx2sen_path = None,
                max_len=32, 
                **kwargs):
       
        self.max_len = max_len
        if isinstance(source_data_path, list): self.source_data_path = source_data_path
        else: self.source_data_path = [source_data_path]
        self.vocab_path = vocab_path
        self.graph_idx2sen_path = graph_idx2sen_path

        from_disk = pickle.load(open(self.vocab_path, "rb"))
        new_v = layers.TextVectorization(
            max_tokens=from_disk['config']['max_tokens'],
            output_mode='int',
            output_sequence_length=from_disk['config']['output_sequence_length'])
        new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        new_v.set_weights(from_disk['weights'])
        self.vocab = new_v
        
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = TFDistilBertModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        super().__init__(**kwargs)

    def download(self):
        os.mkdir(self.path)

        all_data = []
        for data_path in self.source_data_path:
            with open(data_path, "r") as f:
                data = json.load(f)['images']
                all_data.extend(data)

        all_data = all_data
        self.graph_idx2sen = defaultdict(str)
        sentences = [data['sentence'] for data in all_data] 
        processed_sentences = ["[start] " + data['sentence'] + " [end]" for data in all_data] 
        
        for i, (sen, processed_sen) in tqdm(enumerate(zip(sentences, processed_sentences)), total=len(processed_sentences)):
            graph = sng_parser.parse(sen)

            G = nx.DiGraph(name='G')

            for entity in graph['entities']:
                G.add_node(entity['head'])
                for mod in entity['modifiers']:
                    G.add_node(mod['span'])
                    G.add_edge(entity['head'], mod['span'])
                    
            for rela in graph['relations']:
                G.add_node(rela['relation'])
                G.add_edge(graph['entities'][rela['subject']]['head'], rela['relation'])
                G.add_edge(rela['relation'], graph['entities'][rela['object']]['head'])
            try:
                A = np.array(nx.attr_matrix(G)[0])
                X = self.get_bert_embeddings(nx.attr_matrix(G)[1])
                y = {
                    'graph_idx' : i,
                    'tokens' : self.vocab(processed_sen)
                }
                self.graph_idx2sen[i] = {
                    'graph' : nx.attr_matrix(G)[1],
                    'sentence' : sen,
                }
                filename = os.path.join(self.path, f'graph_{i}')
                np.savez(filename, x=X, a=A, y=y)
            except:
                print(sen)
                print(nx.attr_matrix(G)[1])
                continue
        
        with open(self.graph_idx2sen_path, "w") as f:
            json.dump(self.graph_idx2sen, f)

    def get_bert_embeddings(self, tokens):
        node_features = []
        for token in tokens:
            encoded_input = self.tokenizer(token, return_tensors='tf')
            outputs = self.model(encoded_input)
            last_hidden_states = outputs.last_hidden_state
            embeddings = last_hidden_states[:, 0, :].numpy()
            node_features.append(embeddings)
            
        node_features = np.concatenate(node_features, axis=0)
        return node_features 

            
    def read(self):
        output = []
        for graph_path in glob(os.path.join(self.path, "*")):
            data = np.load(graph_path, allow_pickle=True)
        
            output.append(
                Graph(x=data['x'], a=data['a'], y=data['y'])
            )
        return output




class COCOCacheDataset(Dataset):
    def __init__(self, 
                source_data_path, 
                vocab_path = None,
                graph_idx2sen_path = None,
                max_len=32, 
                **kwargs):
       
        self.max_len = max_len
        if isinstance(source_data_path, list): self.source_data_path = source_data_path
        else: self.source_data_path = [source_data_path]
        self.vocab_path = vocab_path
        self.graph_idx2sen_path = graph_idx2sen_path

        from_disk = pickle.load(open(self.vocab_path, "rb"))
        new_v = layers.TextVectorization(
            max_tokens=from_disk['config']['max_tokens'],
            output_mode='int',
            output_sequence_length=from_disk['config']['output_sequence_length'])
        new_v.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
        new_v.set_weights(from_disk['weights'])
        self.vocab = new_v
        
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.model = TFDistilBertModel.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        super().__init__(**kwargs)

    def download(self):
        os.mkdir(self.path)

        all_data = []
        for data_path in self.source_data_path:
            with open(data_path, "r") as f:
                data = json.load(f)['images']
                all_data.extend(data)

        all_data = all_data[:2000]
        self.graph_idx2sen = defaultdict(str)
        sentences = [data['sentence'] for data in all_data] 
        processed_sentences = ["[start] " + data['sentence'] + " [end]" for data in all_data] 
        
        for i, (sen, processed_sen) in tqdm(enumerate(zip(sentences, processed_sentences)), total=len(processed_sentences)):
            graph = sng_parser.parse(sen)

            G = nx.DiGraph(name='G')

            for entity in graph['entities']:
                G.add_node(entity['head'])
                for mod in entity['modifiers']:
                    G.add_node(mod['span'])
                    G.add_edge(entity['head'], mod['span'])
                    
            for rela in graph['relations']:
                G.add_node(rela['relation'])
                G.add_edge(graph['entities'][rela['subject']]['head'], rela['relation'])
                G.add_edge(rela['relation'], graph['entities'][rela['object']]['head'])
            try:
                A = np.array(nx.attr_matrix(G)[0])
                X = self.get_bert_embeddings(nx.attr_matrix(G)[1])
                y = {
                    'graph_idx' : i,
                    'tokens' : self.vocab(processed_sen)
                }
                self.graph_idx2sen[i] = {
                    'graph' : nx.attr_matrix(G)[1],
                    'sentence' : sen,
                }
                filename = os.path.join(self.path, f'graph_{i}')
                np.savez(filename, x=X, a=A, y=y)
            except:
                print(sen)
                print(nx.attr_matrix(G)[1])
                continue
        
        with open(self.graph_idx2sen_path, "w") as f:
            json.dump(self.graph_idx2sen, f)

    def get_bert_embeddings(self, tokens):
        node_features = []
        for token in tokens:
            encoded_input = self.tokenizer(token, return_tensors='tf')
            outputs = self.model(encoded_input)
            last_hidden_states = outputs.last_hidden_state
            embeddings = last_hidden_states[:, 0, :].numpy()
            node_features.append(embeddings)
            
        node_features = np.concatenate(node_features, axis=0)
        return node_features 

            
    def read(self):
        output = []
        for graph_path in glob(os.path.join(self.path, "*")):
            data = np.load(graph_path, allow_pickle=True)
        
            output.append(
                Graph(x=data['x'], a=data['a'], y=data['y'])
            )
        return output


