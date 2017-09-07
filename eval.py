from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import nltk
import torch
import tqdm

import loader
import utils

def eval_loop(model, ldr):
    losses = []
    all_labels = []
    all_preds = []
    for x, y in tqdm.tqdm(ldr):
        x.volatile = True
        y.volatile = True
        if use_cuda:
            x = x.cuda()
            y = y.cuda()

        out = model(x, y)
        loss = model.loss(out, y)
        losses.append(loss.data[0])
        labels = y.data.cpu().numpy().tolist()
        all_labels.extend(labels)
        preds = torch.max(out, dim=2)[1]
        preds = preds.data.cpu().numpy().tolist()
        all_preds.extend(preds)
    avg_loss = sum(losses) / len(losses)
    return avg_loss, (all_labels, all_preds)

def run(model_path, data_json, tag, batch_size=32):
    model, preproc = utils.load(model_path, tag=tag)
    model.cuda() if use_cuda else model.cpu()

    ldr = loader.make_loader(data_json, preproc, batch_size)

    avg_loss, (labels, preds) = eval_loop(model, ldr)

    refs = [[preproc.decode(l)] for l in labels]
    hyps = [preproc.decode(p) for p in preds]
    bleu = nltk.bleu_score.corpus_bleu(refs, hyps)
    print("Loss: {:.2f}, BLEU: {:.3f}".format(avg_loss, bleu))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Evaluate a model on a dataset.")

    parser.add_argument("model_path",
        help="A path to a saved model.")
    parser.add_argument("data_json",
        help="A path to data json file.")
    parser.add_argument("--overfit",
        action="store_true",
        help="Use the last saved model for eval.")
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    tag = "best"
    if args.overfit:
        tag = None

    run(args.model_path, args.data_json, tag)


