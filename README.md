# Environment

```
tensorflow == 2.9.0
spacy == 3.3.0
networkx = 2.6.3
```

# How to Use

Firstly, download the prepared coco Captions(train.json & test.json).

url: https://pan.baidu.com/s/1R1P3Lly38evk5uwqz_X0YQ code: 209s 


## Prepare TextVectorization Layer

```python
source_data_path = ["train.json", "test.json"]
all_data = []
for data_path in source_data_path:
    with open(data_path, "r") as f:
        data = json.load(f)['images']
        all_data.extend(data)


vocab = layers.TextVectorization(
                max_tokens=23000,
                output_mode="int",
                output_sequence_length=32,
            )

processed_sentences = ["[start] " + data['sentence'] + " [end]" for data in all_data] 
vocab.adapt(processed_sentences)
pickle.dump({'config': vocab.get_config(),
            'weights': vocab.get_weights()}
            , open("tv_layer.pkl", "wb"))
```

## Prepare Cache Datasets for Debug

You can preprocess a few captions to make a small dataset for debug.

like this 

```python
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
```


```python
source_data_path = ["test.json"]
vocab_path = 'tv_layer.pkl'
graph_idx2sen_path = 'graph_idx2sen_cache.json'
dataset = COCOCacheDataset(
    source_data_path,
    vocab_path=vocab_path,
    graph_idx2sen_path = graph_idx2sen_path)
dataset.read()
```

## Prepare Train Dataset

You can preprocess train captions to make a train dataset.

like this:

```python
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
```

```python
source_data_path = ["train.json"]
vocab_path = 'tv_layer.pkl'
graph_idx2sen_path = 'graph_idx2sen_train.json'
dataset = COCOTrainDataset(
    source_data_path,
    vocab_path=vocab_path,
    graph_idx2sen_path = graph_idx2sen_path)
dataset.read()
```


## Prepare Test Dataset

You can preprocess test captions to make a test dataset.

like this:

```python
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
```

```python
source_data_path = ["test.json"]
vocab_path = 'tv_layer.pkl'
graph_idx2sen_path = 'graph_idx2sen_test.json'
dataset = COCOTestDataset(
    source_data_path,
    vocab_path=vocab_path,
    graph_idx2sen_path = graph_idx2sen_path)
dataset.read()
```

## Train & Test

```bash
python main.py
```

# Results

|BLEU@1|BLEU@2|BLEU@3|BLEU@4|
|:-:|:-:|:-:|:-:|
|73.36|50.57|45.36|41.64|



