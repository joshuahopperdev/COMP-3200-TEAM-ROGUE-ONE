import numpy as np

sigmoid = lambda x: 1 / (1 + np.exp(-x))
relu = lambda x: (x > 0).astype(float) * x
W = np.array([[1, 4], [4, 1]]) 

def vanishing(steps = 10):
    activation = sigmoid(np.array([1, 0.01]))
    acts = []

    for _ in range(steps):
        activation = sigmoid(activation.dot(W)); acts.append(activation)
    grads, gradient = [], np.ones(2)

    for act in reversed(acts): # sigmoid derivative a*(1-a)
        gradient = (act * (1 - act) * gradient).dot(W.T)
        grads.append(gradient.copy())

    return grads

def exploding(steps=10):
    activation = sigmoid(np.array([1, 0.01]))
    acts = []

    for _ in range(steps):
        activation = relu(activation.dot(W)); acts.append(activation)
    grads, gradient = [], np.ones(2)
    
    for act in reversed(acts): # relu mask: nothing clamps
        gradient = ((act > 0) * gradient).dot(W.T)
        grads.append(gradient.copy())

    return grads

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

if __name__ == "__main__":
    gv = vanishing(10)
    print("VANISHING (sigmoid):")
    print(" |grad| step 1 = {:.4f}".format(np.max(np.abs(gv[0]))))
    print(" |grad| step 10 = {:.3e}".format(np.max(np.abs(gv[-1]))))
    
    ge = exploding(10)
    mags = [int(round(np.max(np.abs(g)))) for g in ge]
    print("EXPLODING (relu):")
    print(" |grad|:", mags)
    print(" 5**10 =", 5 ** 10, " -> match:", mags[-1] == 5 ** 10)

