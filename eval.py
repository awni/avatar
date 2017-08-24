from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
import tqdm

import loader
import utils

def eval_loop(model, ldr):
    losses = []
    for x, y in tqdm.tqdm(ldr):
        x.volatile = True
        y.volatile = True
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        out = model(x, y)
        loss = model.loss(out, y)
        losses.append(loss.data[0])
    avg_loss = sum(losses) / len(losses)
    print("Loss: {:.2f}".format(avg_loss))
    return avg_loss

def run(model_path, data_json, batch_size=32):
    model, preproc = utils.load(model_path, tag="best")
    model.cuda() if use_cuda else model.cpu()

    ldr = loader.make_loader(data_json, preproc, batch_size)

    eval_loop(model, ldr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Evaluate a model on a dataset.")

    parser.add_argument("model_path",
        help="A path to a saved model.")
    parser.add_argument("data_json",
        help="A path to data json file.")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    run(args.model_path, args.data_json)


