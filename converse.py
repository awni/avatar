from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import string

import utils

class Converser:

    def __init__(self, model_path, use_cuda=False):
        loaded = utils.load(model_path, tag="best")
        self.model, self.preproc = loaded
        model, preproc = self.model, self.preproc

        model.cuda() if use_cuda else model.cpu()
        self.use_cuda = use_cuda

        # For decoding
        self.start = preproc.word_to_int[preproc.START]
        self.end = preproc.word_to_int[preproc.END]
        self.beam_size = 1
        self.max_len = 50

    def reply(self, query):
        toks = self.preproc.tokenize(query.lower())
        toks = self.preproc.encode(toks)
        toks = torch.autograd.Variable(torch.LongTensor(toks))
        toks.volatile = True
        if self.use_cuda:
            toks = toks.cuda()
        toks = toks.unsqueeze(dim=0)
        response, score = self.model.beam_search(toks, self.start,
                            self.end, self.beam_size, self.max_len)
        response = self.preproc.decode(response)
        return self.untokenize(response)

    @staticmethod
    def space(tok):
        if "'" in tok:
            return False
        if tok in string.punctuation:
            return False
        return True

    @staticmethod
    def untokenize(toks):
        toks = (" " + t if Converser.space(t) else t
                for t in toks)
        return "".join(toks).strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Converse with a model.")

    parser.add_argument("model_path",
        help="A path to a saved model.")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    converser = Converser(args.model_path, use_cuda)
    while True:
        query = raw_input("You: ")
        reply = converser.reply(query)
        print("Avatar: " + reply)


