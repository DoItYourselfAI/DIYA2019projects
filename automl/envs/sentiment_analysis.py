import os
import sys

import torch
import torch.nn as nn
import torchtext
import torchtext.datasets as datasets

from settings import PROJECT_ROOT, DATA_DIR

__all__ = ['imdb']


class DataLoader:
    def __init__(self, data, vocab, **kwargs):
        self.data = torchtext.data.BucketIterator(data, repeat=False, **kwargs)
        self.iter = iter(self.data)

        self.embedding = nn.Embedding(vocab.size(0), vocab.size(1))
        self.embedding.weight.data.copy_(vocab)

    def __iter__(self):
        self.iter = iter(self.data)
        return self

    def __next__(self):
        batch = next(self.iter)
        data = self.embedding(batch.text)
        data = data.transpose(0,1).detach()
        label = batch.label.detach()
        return data, label


class SentimentAnalysis:
    def __init__(self, dataset, path, fix_length=1000, tokenize=str.split):
        self.text_field = torchtext.data.Field(lower=True, fix_length=fix_length, tokenize=tokenize)
        self.label_field = torchtext.data.LabelField(dtype=torch.long)
        self.train, self.val = dataset.splits(self.text_field, self.label_field, root=path)

    def vocab(self, vectors):
        self.text_field.build_vocab(self.train, vectors=vectors)
        self.label_field.build_vocab(self.train)
        return self.text_field.vocab.vectors


def imdb(args):
    ds = datasets.IMDB
    root = os.path.join(PROJECT_ROOT, DATA_DIR)

    path = os.path.join(root, 'glove')
    vectors = torchtext.vocab.Vectors('glove.6B.50d.txt', path)

    model = SentimentAnalysis(ds, root, fix_length=30)
    train = model.train
    val = model.val
    vocab = model.vocab(vectors)

    config = {
        'batch_size': args.batch_size,
        'shuffle': args.shuffle
    }
    return {
        'train': DataLoader(train, vocab, **config),
        'val': DataLoader(val, vocab, **config)
    }
