import chainer
import chainer.functions as F
import chainer.links as L


class MnistMLP(chainer.Chain):
    def __init__(self, n_in, n_units, n_out, initW1=None, initb1=None, initW2=None, initb2=None):
        super(MnistMLP, self).__init__(
            l1=L.Linear(n_in, n_units, initialW=initW1, initial_bias=initb1),
            l2=L.Linear(n_units, n_out, initialW=initW2, initial_bias=initb2),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        return self.l2(h1)
