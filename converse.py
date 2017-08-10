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
        self.model.cuda() if use_cuda else self.model.cpu()
        self.use_cuda = use_cuda

    def reply(self, query):
        toks = self.preproc.tokenize(query.lower())
        toks = self.preproc.encode(toks)
        toks = torch.autograd.Variable(torch.LongTensor(toks))
        toks.volatile = True
        toks = toks.unsqueeze(dim=0)

        response, _ = self.model.decode_test(toks)
        response = response.cpu().data.numpy().squeeze()
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

    use_cuda = False #torch.cuda.is_available()
    converser = Converser(args.model_path, use_cuda)
    while True:
        query = raw_input("You: ")
        reply = converser.reply(query)
        print("Avatar: " + reply)


