import numpy as np
from tensor import Tensor
from layers import Embedding, LSTMCell, CrossEntropyLoss
from optim import SGD

def load_corpus(path="data/qa_oldtongue.txt"):
    raw = open(path, "r").read()
    # sorted: the same map on every run
    vocab = sorted(set(raw))
    word2index = {ch: i for i, ch in enumerate(vocab)}
    indices = np.array([word2index[ch] for ch in raw])
    return raw, vocab, word2index, indices

def make_batches(indices, batch_size=16, bptt=25):
    n = indices.shape[0] // batch_size # characters per column
    batched = indices[:n * batch_size].reshape(batch_size, n).transpose()
    n_bptt = (n - 1) // bptt # whole chunks of bptt steps
    inputs = batched[0:-1][:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)
    targets = batched[1:][:n_bptt * bptt].reshape(n_bptt, bptt, batch_size)
    # targets: the inputs shifted by one
    return inputs, targets

def build_model(vocab_size, alpha=0.032, seed=0):
    np.random.seed(seed)
    embed = Embedding(vocab_size=vocab_size, dim=512)
    model = LSTMCell(n_inputs=512, n_hidden=512, n_output=vocab_size)
    # every logit 0: a uniform guess
    model.w_ho.weight.data *= 0
    criterion = CrossEntropyLoss()
    optim = SGD(parameters=model.get_parameters() + embed.get_parameters(),
    alpha=alpha)
    return embed, model, criterion, optim

def train(embed, model, criterion, optim, inputs, targets, max_batches=None):
    bptt, batch_size = inputs.shape[1], inputs.shape[2]
    n = len(inputs) if max_batches is None else min(max_batches, len(inputs))
    hidden = model.init_hidden(batch_size=batch_size)
    total_loss, perplexities = 0, []

    for batch_i in range(n):
        # truncated BPTT: keep the state's numbers, cut its history
        hidden = (Tensor(hidden[0].data, autograd=True),
        Tensor(hidden[1].data, autograd=True))
        loss = None
        for t in range(bptt):
            inp = Tensor(inputs[batch_i][t], autograd=True)
            output, hidden = model.forward(input=embed.forward(input=inp),
            hidden=hidden)
            target = Tensor(targets[batch_i][t], autograd=True)
            step_loss = criterion.forward(output, target)
            loss = step_loss if loss is None else step_loss + loss
        loss.backward()
        optim.step()
        total_loss += loss.data / bptt # mean loss per character
        perplexities.append(float(np.exp(total_loss / (batch_i + 1))))
    return perplexities

if __name__ == "__main__":
    raw, vocab, word2index, indices = load_corpus("data/qa_oldtongue.txt")
    print("characters:", len(raw), " vocab size:", len(vocab))
    ib, tb = make_batches(indices)
    print("chunks per epoch:", len(ib), " each chunk:", ib[0].shape)
    # (a) train 30 batches (seed 0, sorted vocab, alpha 0.032) -> descent from 62
    embed, model, crit, optim = build_model(len(vocab))
    ppl = train(embed, model, crit, optim, ib, tb, max_batches=30)
    for b in (1, 5, 10, 15, 20, 25, 30):
        print(" batch {:2d}: {:.10f}".format(b, ppl[b - 1]))
    flat = all(ppl[k + 1] <= ppl[k] + 1e-9 for k in range(len(ppl) - 1))
    print("never rises:", flat)
    # (b) the thesis: gated cell state bounded, ungated runs away (50 steps)
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def lstm_cell_state_horizon(steps=50, forget_gate=True, seed=0):
        rng = np.random.RandomState(seed)
        c, norms = np.zeros(16), []
        for _ in range(steps):
            x = rng.randn(16)
            f_pre = rng.randn(16) # always drawn: same writes either way
            f = sigmoid(f_pre) if forget_gate else np.ones(16)
            i = sigmoid(rng.randn(16))
            c = f * c + i * np.tanh(x) # keep f of the old, write i of the new
            norms.append(float(np.max(np.abs(c))))
        return norms
    gated = lstm_cell_state_horizon(steps=50, forget_gate=True)
    ungated = lstm_cell_state_horizon(steps=50, forget_gate=False)
    row = "{:<16}: step1 |c|={:.4f} step50 |c|={:.4f}"
    print()
    print(row.format("forget gate ON", gated[0], gated[-1]))
    print(row.format("forget gate OFF", ungated[0], ungated[-1]))
    print("ratio at step 50: {:.1f}x".format(ungated[-1] / gated[-1]))
    top = np.tanh(ungated[-1])
    print("tanh of the ungated |c| at step 50: {:.7f}".format(top))
