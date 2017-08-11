"""
Run tests with `pytest tests.py -v`
"""

import numpy as np
import tempfile
import torch
import torch.autograd as autograd

import loader
import models
import utils

def load_model(vocab_size):
    config = {
        "rnn_dim" : 128,
        "embedding_dim" : 128,
        "encoder_layers" : 2,
        "bidirectional" : False,
        "decoder_layers" : 2
    }
    model = models.Seq2Seq(vocab_size, config)
    return model

def test_loader():
    max_vocab_size = 10000
    batch_size = 8
    data_json = "examples/cornell_mdc/data/train_small.json"

    preproc = loader.Preprocessor(data_json, max_vocab_size)
    ldr = loader.make_loader(data_json, preproc, batch_size)
    for inputs, labels in ldr:
        assert inputs.size()[0] == labels.size()[0]
        assert inputs.size()[0] == batch_size

def test_model():
    torch.manual_seed(2017)
    np.random.seed(2017)

    vocab_size = 100
    batch_size = 4
    in_t = 10
    out_t = 8
    def gen_fake_data(shape):
        d = np.random.randint(0, vocab_size, shape)
        d = torch.from_numpy(d)
        d = autograd.Variable(d)
        return d
    x = gen_fake_data((batch_size, out_t))
    y = gen_fake_data((batch_size, in_t))

    model = load_model(vocab_size)

    out = model(x, y)
    loss = model.loss(out, y)

    # Test the decode_test gives the same results as the regular
    # decoder with the infered labels.
    labels, acts = model.decode_test(x)
    expected = model(x, labels)
    assert expected.size() == acts.size(), "Size mismatch."
    acts = acts.data.numpy()
    exp = expected.data.numpy()
    assert np.allclose(acts, exp, rtol=1e-6, atol=1e-7), \
            "Results should be quite close."

def test_save():

    batch_size = 8
    vocab_size = 100
    model = load_model(vocab_size)

    # TODO, make a fake json for testing
    data_json = "examples/cornell_mdc/data/train_small.json"
    preproc = loader.Preprocessor(data_json, vocab_size)

    save_dir = tempfile.mkdtemp()
    utils.save(model, preproc, save_dir)

    s_model, s_preproc = utils.load(save_dir)
    assert hasattr(s_preproc, 'int_to_word')
    assert hasattr(s_preproc, 'word_to_int')
    assert hasattr(s_preproc, 'unk_idx')

    msd = model.state_dict()
    for k, v in s_model.state_dict().items():
        assert k in msd
