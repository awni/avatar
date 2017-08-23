from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, config):
        super(Seq2Seq, self).__init__()

        embed_dim = config["embedding_dim"]
        rnn_dim = config["rnn_dim"]
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Encoding
        self.enc_rnn = nn.GRU(input_size=embed_dim,
                          hidden_size=rnn_dim,
                          num_layers=config["encoder_layers"],
                          batch_first=True, dropout=False,
                          bidirectional=config["bidirectional"])

        # For decoding
        self.dec_rnn = nn.GRU(input_size=embed_dim,
                          hidden_size=rnn_dim,
                          num_layers=config["decoder_layers"],
                          batch_first=True, dropout=False,
                          bidirectional=False)
        self.attend = Attention()
        self.fc = nn.Linear(rnn_dim, vocab_size)

    def loss(self, out, y):
        batch_size, _, out_dim = out.size()
        out = out.view((-1, out_dim))
        y = y[:,1:].contiguous().view(-1)
        loss = nn.functional.cross_entropy(out, y,
                size_average=False)
        loss = loss / batch_size
        return loss

    def forward(self, x, y):
        x = self.encode(x)
        out, _ = self.decode(x, y)
        return out

    def encode(self, x):
        x = self.embedding(x)
        x, h = self.enc_rnn(x)
        return x

    def decode(self, x, y):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        inputs = self.embedding(y[:, :-1])

        out = []; aligns = []
        hx = None; sx = None
        for t in range(y.size()[1] - 1):
            ix = inputs[:, t:t+1, :]
            #ix = ix if sx is None else ix + sx

            ox, hx = self.dec_rnn(ix, hx=hx)
            sx, ax = self.attend(x, ox)

            aligns.append(ax)
            out.append(ox + sx)

        out = torch.cat(out, dim=1)
        b, t, h = out.size()
        out = out.view(b * t, h)
        out = self.fc(out)
        out = out.view(b, t, out.size()[1])

        aligns = torch.cat(aligns, dim=0)
        return out, aligns

    def decode_step(self, x, y, hx=None, softmax=False):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, 1)
        """
        ix = self.embedding(y)
        ox, hx = self.dec_rnn(ix, hx=hx)
        sx, _ = self.attend(x, ox)
        out = ox + sx
        out = self.fc(out.squeeze(dim=1))
        if softmax:
            out = nn.functional.log_softmax(out)

        return out, hx

    def beam_search(self, x, start_tok, end_tok,
                    beam_size, max_len):
        y = x[:, 0:1].clone()
        x = self.encode(x)
        beam = [((start_tok,), 0, None)];
        complete = []
        for _ in range(max_len):
            new_beam = []
            for hyp, score, hx in beam:

                y[0] = hyp[-1]
                out, hx = self.decode_step(x, y, hx=hx, softmax=True)
                out = out.cpu().data.numpy().squeeze(axis=0).tolist()
                for i, p in enumerate(out):
                    new_score = score + p
                    new_hyp = hyp + (i,)
                    new_beam.append((new_hyp, new_score, hx))
            new_beam = sorted(new_beam, key=lambda x: x[1], reverse=True)

            # Remove complete hypotheses
            for cand in new_beam[:beam_size]:
                if cand[0][-1] == end_tok:
                    complete.append(cand)
            if len(complete) >= beam_size:
                complete = complete[:beam_size]
                break
            beam = filter(lambda x : x[0][-1] != end_tok, new_beam)
            beam = beam[:beam_size]

        complete = sorted(complete, key=lambda x: x[1], reverse=True)
        hyp, score, _ = complete[0]
        return hyp, score

class Attention(nn.Module):

    def forward(self, eh, dhx):
        """
        Arguments:
            eh (FloatTensor): the encoder hidden state with
                shape (batch size, time, hidden dimension).
            dhx (FloatTensor): one time step of the decoder hidden
                state with shape (batch size, 1, hidden dimension).
                The hidden dimension must match that of the
                encoder state.

        Returns the summary of the encoded hidden state
        and the corresponding alignment.
        """
        # Compute inner product of decoder slice with every
        # encoder slice.
        ax = torch.sum(eh * dhx, dim=2)
        ax = nn.functional.softmax(ax)

        # At this point sx should have size (batch size, time).
        # Reduce the encoder state accross time weighting each
        # slice by its corresponding value in sx.
        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * ax.unsqueeze(dim=2), dim=1,
                       keepdim=True)
        return sx, ax
