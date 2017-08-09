from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(self, vocab_size, config):
        super(Seq2Seq, self).__init__()

        # TODO, make this configurable.
        # TODO, add more decoder layers.
        embed_dim = 16
        rnn_dim = 128
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Encoding
        self.rnn = nn.GRU(input_size=embed_dim,
                          hidden_size=rnn_dim,
                          num_layers=2,
                          batch_first=True, dropout=False,
                          bidirectional=False)

        # For decoding
        self.dec_rnn = nn.GRUCell(embed_dim + rnn_dim, rnn_dim)
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
        x, h = self.rnn(x)
        return x

    def decode(self, x, y):
        """
        x should be shape (batch, time, hidden dimension)
        y should be shape (batch, label sequence length)
        """
        batch_size, seq_len = y.size()
        inputs = self.embedding(y[:, :-1])

        hx = self.h_init.expand(batch_size, self.h_init.size()[1])

        out = []; aligns = []
        ax = None
        for t in range(seq_len - 1):
            ix = inputs[:, t, :].squeeze(dim=1)
            sx, ax = self.attend(x, hx, ax)
            ix = torch.cat([ix, sx], dim=1)
            hx = self.dec_rnn(ix, hx)
            aligns.append(ax)
            out.append(hx + sx)

        out = torch.stack(out, dim=1)
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

    def __init__(self, kernel_size=11):
        """
        Module which Performs a single attention step along the
        second axis of a given encoded input. The module uses
        both 'content' and 'location' based attention.

        The 'content' based attention is an inner product of the
        decoder hidden state with each time-step of the encoder
        state.

        The 'location' based attention performs a 1D convollution
        on the previous attention vector and adds this into the
        next attention vector prior to normalization.

        *NB* Computes attention differently if using cuda or cpu
        based on performance. See
        https://gist.github.com/awni/9989dd31642d42405903dec8ab91d1f0
        """
        super(Attention, self).__init__()
        assert kernel_size % 2 == 1, \
            "Kernel size should be odd for 'same' conv."
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=padding)

    def forward(self, eh, dhx, ax=None):
        """
        Arguments:
            eh (FloatTensor): the encoder hidden state with
                shape (batch size, time, hidden dimension).
            dhx (FloatTensor): one time step of the decoder hidden
                state with shape (batch size, hidden dimension).
                The hidden dimension must match that of the
                encoder state.
            ax (FloatTensor): one time step of the attention
                vector.

        Returns the summary of the encoded hidden state
        and the corresponding alignment.
        """
        # Compute inner product of decoder slice with every
        # encoder slice.
        dhx = dhx.unsqueeze(1)
        pax = torch.sum(eh * dhx, dim=2)
        if ax is not None:
            ax = ax.unsqueeze(dim=1)
            ax = self.conv(ax).squeeze(dim=1)
            pax = pax + ax
        ax = nn.functional.softmax(pax)

        # At this point sx should have size (batch size, time).
        # Reduce the encoder state accross time weighting each
        # slice by its corresponding value in sx.
        sx = ax.unsqueeze(2)
        sx = torch.sum(eh * ax.unsqueeze(dim=2), dim=1)
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

    model = Seq2Seq(vocab_size)
    out = model.forward(x, y)
    loss = model.loss(out, y)
