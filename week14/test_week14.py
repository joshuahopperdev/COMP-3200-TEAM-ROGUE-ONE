import numpy as np
from tensor import Tensor
from layers import Embedding, CrossEntropyLoss, LSTMCell
from exploding_gradients import exploding
import train_lstm as T

def test_index_select():
    x = Tensor(np.eye(5), autograd=True)
    x.index_select(Tensor([[1, 2, 3], [2, 3, 4]])).backward()
    assert np.all(x.grad.data[:, 0].flatten() == [0,1,2,2,1]), f"Rows selected incorrectly, expected [0,1,2,2,1], got {x.grad.data[:, 0].flatten()}"

def test_untrained_perplexity():
    np.random.seed(0)
    vocab_size = 62
    embed = Embedding(vocab_size, 512)
    model = LSTMCell(512, 512, vocab_size)
    model.w_ho.weight.data *= 0
    chars = Tensor(np.arange(16))
    hidden = model.init_hidden(batch_size=16)
    output, _ = model.forward(embed.forward(chars), hidden)
    loss = CrossEntropyLoss().forward(output, Tensor(np.arange(16)))
    assert np.isclose(np.exp(loss.data), vocab_size), f"Untrained perplexity must be close to vocab_size ({vocab_size}), got {np.exp(loss.data)}"

def test_one_step_decreases_loss():
    raw, vocab, w2i, indices = T.load_corpus()
    bi, bt = T.make_batches(indices)
    embed, model, crit, optim = T.build_model(len(vocab))
    bptt, batch_size = bi.shape[1], bi.shape[2]
    hidden = model.init_hidden(batch_size=batch_size)
    first_loss = train_step(hidden, bptt, bi, bt, model, embed, crit, optim)
    second_loss = train_step(hidden, bptt, bi, bt, model, embed, crit, optim)
    assert np.all(second_loss < first_loss), f"One optimizer step should lower the LSTM's loss, got first loss={first_loss}, second loss={second_loss}"

def test_exploding_gradients():
    step = np.random.randint(5, 20)
    ge = exploding(step)
    mags = [int(round(np.max(np.abs(g)))) for g in ge]
    assert mags[-1] == 5 ** step, f"Gradient must explode, expected {5 ** step}, got {mags[-1]}"

def train_step(hidden, bptt, bi, bt, model, embed, crit, optim):
    hidden = (Tensor(hidden[0].data, autograd=True),
              Tensor(hidden[1].data, autograd=True))

    loss = None
    for t in range(bptt):
        inp = Tensor(bi[0][t], autograd=True)
        output, hidden = model.forward(input=embed.forward(input=inp),
                                       hidden=hidden)
        target = Tensor(bt[0][t], autograd=True)
        step_loss = crit.forward(output, target)
        loss = step_loss if loss is None else step_loss + loss
    loss.backward()
    optim.step()
    return loss.data / bptt

if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")