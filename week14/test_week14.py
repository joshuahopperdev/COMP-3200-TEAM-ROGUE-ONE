import numpy as np
from tensor import Tensor
from layers import Embedding, CrossEntropyLoss, LSTMCell

def test_index_select():
    tensor = Tensor(np.eye(5), autograd=True)
    indices = Tensor([[1,2,3],[2,3,4]])
    rows = tensor.index_select(indices)
    assert np.all(rows == [0,1,2,2,1]), f"Rows selected incorrectly, expected [0,1,2,2,1], got {rows}"

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
    pass

def test_exploding_gradients():
    pass

if __name__ == "__main__":
    tests = [name for name in dir() if name.startswith("test_")]
    for test_name in sorted(tests):
        test_func = globals()[test_name]
        try:
            test_func()
            print(f" PASS: {test_name}")
        except AssertionError as e:
            print(f" FAIL (AssertionError): {test_name} -- {e}")