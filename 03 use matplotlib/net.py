import chainer
import chainer.functions as F
import chainer.links as L


class MnistMLP(chainer.Chain):
    # n_hiddenlayer = 1
    def __init__(self, n_in, n_units, n_out):
        super(MnistMLP, self).__init__(
            l1=L.Linear(n_in, n_units),
            l2=L.Linear(n_units, n_out),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)
