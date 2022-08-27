import pandas as pd
import torch
from torchtext import data
from torchtext.vocab import Vectors
from torchtext.data import TabularDataset,Dataset,BucketIterator,Iterator
from torchtext.data import Field, BucketIterator
from torch.nn import init
from tqdm import tqdm
import nltk
import spacy


class WordEmbedding():
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.spacy_en = spacy.load('en_core_web_sm')
        self.load_glove()

    def tokenize(self, text): # create a tokenizer function
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def load_glove(self):
        DOCUMENT = data.Field(sequential=True, tokenize=self.tokenize, lower=True, fix_length=200)
        LABEL = data.Field(sequential=False, use_vocab=False)
        fields = [("ID", None), ("document",DOCUMENT), ("label",LABEL)]

        train_data = TabularDataset.splits(
            path='.',
            train=self.file_path,
            format='csv',
            skip_header=True,
            fields=fields
        )[0]

        DOCUMENT.build_vocab(train_data, vectors='glove.6B.100d')
        self.DOCUMENT = DOCUMENT
    
    def _get_word_vector(self, word):
        return self.DOCUMENT.vocab.vectors[self.DOCUMENT.vocab.stoi[word]]

    def get_vector(self, sentence):
        sentence = sentence.lower()
        words = self.tokenize(sentence)
        words_tensors = []
        for word in words:
            words_tensors.append(self._get_word_vector(word))
        return torch.stack(words_tensors, dim=0) 

class AgeDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, word_embedding):
        super().__init__()

        self.file_path = file_path
        self.word_embedding = word_embedding

        data = pd.read_csv(file_path)
        self.documents = data.loc[:, 'document']
        labels = data.loc[:, 'AGE']

        self.sentences = []
        self.labels = []
        for i in range(len(self.documents)):
            document = self.documents[i]
            label = labels[i]
            self.sentences.append(self.word_embedding.get_vector(document))
            self.labels.append(torch.tensor(label, dtype=torch.long))
            
    
    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.sentences)
    
    def collate_fn(self, batch_data):
        # batch_datas 是一个批次的数据 List类型
        datas = []
        tags = []
        batch_lens = []

        for data, tag in batch_data:
            datas.append(data)
            tags.append(tag)
            batch_lens.append(len(data))
        batch_max_len = max(batch_lens)

        # 填充至相同长度
        datas = [i.tolist() + [self.word_embedding._get_word_vector('<pad>').tolist()] * (batch_max_len - len(i)) for i in datas]  # i是，每句话 # i:[word_num, emb_dim]
        return torch.tensor(datas), torch.stack(tags)


class YelpWordEmbedding():

    def __init__(self, file_path):
        self.file_path = file_path
        self.spacy_en = spacy.load('en_core_web_sm')

        self.load_glove()

    def tokenize(self, text):
        return [tok.text for tok in self.spacy_en.tokenizer(text)]

    def load_glove(self):
        DOCUMENT = data.Field(sequential=True, tokenize=self.tokenize, lower=True, fix_length=200)
        LABEL = data.Field(sequential=False, use_vocab=False)
        fields = [("stars", LABEL), ("reviews", DOCUMENT)]

        train_data = TabularDataset.splits(
            path='.',
            train=self.file_path,
            format='csv',
            skip_header=True,
            fields=fields
        )[0]

        DOCUMENT.build_vocab(train_data, vectors='glove.6B.100d')
        self.DOCUMENT = DOCUMENT
    
        
    def _get_word_vector(self, word):
        return self.DOCUMENT.vocab.vectors[self.DOCUMENT.vocab.stoi[word]]

    def get_vector(self, sentence):
        sentence = sentence.lower()
        words = self.tokenize(sentence)
        words_tensors = []
        for word in words:
            words_tensors.append(self._get_word_vector(word))
        return torch.stack(words_tensors, dim=0)


class YelpDataset(torch.utils.data.Dataset):

    def __init__(self, file_path, word_embedding):
        super().__init__()

        self.file_path = file_path
        self.word_embedding = word_embedding

        data = pd.read_csv(file_path)
        self.documents = data.loc[:, 'reviews']
        labels = data.loc[:, 'stars']

        self.sentences = []
        self.labels = []
        for i in range(len(self.documents)):
            document = self.documents[i]
            label = labels[i]
            self.sentences.append(self.word_embedding.get_vector(document))
            self.labels.append(torch.tensor(label, dtype=torch.long))
    
    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.sentences)

    def collate_fn(self, batch_data):
        # batch_datas 是一个批次的数据 List类型
        datas = []
        tags = []
        batch_lens = []

        for data, tag in batch_data:
            datas.append(data)
            tags.append(tag)
            batch_lens.append(len(data))
        batch_max_len = max(batch_lens)

        datas = [i.tolist() + [self.word_embedding._get_word_vector('<pad>').tolist()] * (batch_max_len - len(i)) for i in datas]  # i是，每句话 # i:[word_num, emb_dim]
        return torch.tensor(datas), torch.stack(tags)