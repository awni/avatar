from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import json
import random
import numpy as np
import nltk.tokenize as nltok
import torch
import torch.autograd as autograd
import torch.utils.data as tud

class Preprocessor():

    END = "</s>"
    START = "<s>"
    UNK = "<unk>"

    def __init__(self, data_json, max_vocab_size):
        """
        Builds a preprocessor from a dataset.
        Arguments:
            data_json (string): A file containing a json representation
                of each example per line.
            max_vocab_size (int): Maximum vocabulary size, all other
                words get mapped to UNK.
        """
        data = read_data_json(data_json)

        # Build the vocabulary
        toks = [self.START, self.END, self.UNK]
        words = (word for conv in data for turn in conv
                    for word in turn['text'])
        vocab = collections.Counter(words)
        vocab = vocab.most_common(max_vocab_size - len(toks))
        vocab = [word for word, _ in vocab]
        vocab.extend(toks)

        self.int_to_word = dict(enumerate(vocab))
        self.word_to_int = {v : k for k, v in self.int_to_word.items()}
        self.unk_idx = self.word_to_int[self.UNK]

    @staticmethod
    def tokenize(text):
        return nltok.word_tokenize(text.lower())

    def encode(self, text):
        text = list(text)
        text = [self.START] + text + [self.END]
        return [self.word_to_int.get(t, self.unk_idx)
                for t in text]

    def decode(self, seq):
        text = [self.int_to_word[s] for s in seq]
        s = text[0] == self.START
        e = len(text)
        if text[-1] == self.END:
            e = text.index(self.END)
        return text[s:e]

    def preprocess(self, inputs, labels):
        inputs = self.encode(inputs)
        labels = self.encode(labels)
        return inputs, labels

    @property
    def vocab_size(self):
        return len(self.int_to_word)

class Dataset(tud.Dataset):

    def __init__(self, data_json, preproc,
                 batch_size, max_len=50):

        data = read_data_json(data_json)
        self.preproc = preproc

        # Make pairs, note some pairs are the same speaker,
        # not sure if we should merge / disallow this.
        pairs = [(a['text'], b['text'])
                 for conv in data
                 for a, b in zip(conv[:-1], conv[1:])]

        pairs = [preproc.preprocess(*pair) for pair in pairs]

        # Filter utterances that are too long.
        filt_fn = lambda x : all(len(i) < max_len for i in x)
        pairs = filter(filt_fn, pairs)

        bucket_diff = 4
        max_len = max(len(b) for _, b in pairs)
        num_buckets = (max_len // bucket_diff) ** 2
        buckets = [[] for _ in range(num_buckets)]
        for d in pairs:
            bid = min(len(d[1]) // bucket_diff, num_buckets - 1)
            buckets[bid].append(d)

        # Sort by input length followed by output length
        sort_fn = lambda x : (len(x[0]), len(x[1]))
        for b in buckets:
            b.sort(key=sort_fn)
        pairs = [d for b in buckets for d in b]

        self.data = pairs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class BatchRandomSampler(tud.sampler.Sampler):
    """
    Batches the data consecutively and randomly samples
    by batch without replacement.
    """

    def __init__(self, data_source, batch_size):
        it_end = len(data_source) - batch_size + 1
        self.batches = [range(i, i + batch_size)
                for i in range(0, it_end, batch_size)]
        self.data_source = data_source

    def __iter__(self):
        random.shuffle(self.batches)
        return (i for b in self.batches for i in b)

    def __len__(self):
        return len(self.data_source)

def end_pad_concat(examples):
    # Assumes last item in each example is the end token.
    batch_size = len(examples)
    end_tok = examples[0][-1]
    max_len = max(len(e) for e in examples)
    mat = np.full((batch_size, max_len),
                    fill_value=end_tok, dtype=np.int64)
    for e, l in enumerate(examples):
        mat[e, :len(l)] = l
    return mat

def collate(batch):
    inputs, labels = zip(*batch)
    inputs = end_pad_concat(inputs)
    labels = end_pad_concat(labels)
    inputs = autograd.Variable(torch.from_numpy(inputs))
    labels = autograd.Variable(torch.from_numpy(labels))
    return inputs, labels

def make_loader(dataset_json, preproc,
                batch_size, num_workers=0):
    dataset = Dataset(dataset_json, preproc, batch_size)
    sampler = BatchRandomSampler(dataset, batch_size)
    loader = tud.DataLoader(dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate,
                drop_last=True)
    return loader

def read_data_json(data_json):
    with open(data_json) as fid:
        return [json.loads(l) for l in fid]
