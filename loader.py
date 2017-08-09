from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import random
import numpy as np
import torch
import torch.autograd as autograd
import torch.utils.data as tud
import collections

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
        # TODO, should clean up text a bit, lowercase, punctuation, etc.
        toks = [self.START, self.END, self.UNK]
        words = (word for conv in data for turn in conv
                    for word in turn['text'].split())
        vocab = collections.Counter(words)
        vocab = vocab.most_common(max_vocab_size - len(toks))
        vocab = [word for word, _ in vocab]
        vocab.extend(toks)

        self.int_to_word = dict(enumerate(vocab))
        self.word_to_int = {v : k for k, v in self.int_to_word.items()}
        self.unk_idx = self.word_to_int[self.UNK]

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
        inputs = self.encode(inputs.split())
        labels = self.encode(labels.split())
        return inputs, labels

    @property
    def vocab_size(self):
        return len(self.int_to_word)

class Dataset(tud.Dataset):

    def __init__(self, data_json, preproc, batch_size):

        data = read_data_json(data_json)
        self.preproc = preproc

        # Make pairs, note some pairs are the same speaker,
        # not sure if we should merge / disallow this.
        pairs = [(a['text'], b['text'])
                 for conv in data
                 for a, b in zip(conv[:-1], conv[1:])]

        pairs = [preproc.preprocess(*pair) for pair in pairs]

        pairs.sort(key=lambda x : (len(x[0]), len(x[1])))
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

if __name__ == "__main__":
    max_vocab_size = 10000
    batch_size = 8
    data_json = "examples/cornell_mdc/data/train.json"

    preproc = Preprocessor(data_json, max_vocab_size)
    loader = make_loader(data_json, preproc, batch_size)
    for inputs, labels in loader:
        assert inputs.size()[0] == labels.size()[0]
        assert inputs.size()[0] == batch_size
