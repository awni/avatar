from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, config):
        super(Seq2Seq, self).__init__()

        # TODO, add more decoder layers.
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
        self.dec_rnn = nn.GRU(input_size=embed_dim + rnn_dim,
                          hidden_size=rnn_dim,
                          num_layers=config["decoder_layers"],
                          batch_first=True, dropout=False,
                          bidirectional=False)
        self.h_init = nn.Parameter(data=torch.zeros(1, rnn_dim))
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

    def cpu(self):
        super(Model, self).cpu()
        self.dec_rnn.bias_hh.data.squeeze_()
        self.dec_rnn.bias_ih.data.squeeze_()

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
        batch_size, seq_len = y.size()
        inputs = self.embedding(y[:, :-1])
        ox = self.h_init.expand(batch_size, 1, self.h_init.size()[1])

        out = []; aligns = []
        hx = None; ax = None
        for t in range(seq_len - 1):
            ix = inputs[:, t:t+1, :]
            sx, ax = self.attend(x, ox)
            ix = torch.cat([ix, sx], dim=2)
            ox, hx = self.dec_rnn(ix, hx=hx)
            aligns.append(ax)
            out.append(ox + sx)

        out = torch.cat(out, dim=1)
        b, t, h = out.size()
        out = out.view(b * t, h)
        out = self.fc(out)
        out = out.view(b, t, out.size()[1])

        aligns = torch.cat(aligns, dim=0)
        return out, aligns

    def predict(self, probs):
        _, argmaxs = probs.max(dim=2)
        if argmaxs.is_cuda:
            argmaxs = argmaxs.cpu()
        argmaxs = argmaxs.data.numpy()
        return [seq.tolist() for seq in argmaxs]

class Attention(nn.Module):

    def __init__(self):
        super(Attention, self).__init__()

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

if __name__ == "__main__":
    vocab_size = 100
    batch_size = 4
    in_t = 10
    out_t = 8
    def gen_fake_data(shape):
        d = np.random.randint(0, vocab_size, shape)
        d = torch.from_numpy(d)
        d = torch.autograd.Variable(d)
        return d
    x = gen_fake_data((batch_size, out_t))
    y = gen_fake_data((batch_size, in_t))

    config = {
        "rnn_dim" : 128,
        "embedding_dim" : 128,
        "encoder_layers" : 2,
        "bidirectional" : False,
        "decoder_layers" : 2
    }

    model = Seq2Seq(vocab_size, config)
    out = model.forward(x, y)
    loss = model.loss(out, y)

