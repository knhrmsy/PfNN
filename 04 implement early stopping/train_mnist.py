import argparse
import numpy as np
import time
import matplotlib.pyplot as plt
import code

import chainer
from chainer import computational_graph
from chainer import cuda
import chainer.links as L
from chainer import optimizers
from chainer import serializers

import data
import net

starttime = time.time()

parser = argparse.ArgumentParser(description='Chainer example: MNIST')
parser.add_argument('--initmodel', '-m', default='',
                    help='Initialize the model from given file')
parser.add_argument('--resume', '-r', default='',
                    help='Resume the optimization from snapshot')
parser.add_argument('--net', '-n', choices=('simple', 'parallel'),
                    default='simple', help='Network type')
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

batchsize = 100
n_epoch = 10000
n_units = 1000

# Prepare dataset
print 'load MNIST dataset'
mnist = data.load_mnist_data()
mnist['data'] = mnist['data'].astype(np.float32)
mnist['data'] /= 255
mnist['target'] = mnist['target'].astype(np.int32)

N_train = 50000
N_validate = 10000
N_test = 10000
x_train, x_validate, x_test = np.split(
    mnist['data'], [N_train, N_train+N_validate])
y_train, y_validate, y_test = np.split(
    mnist['target'], [N_train, N_train+N_validate])
if N_test != y_test.size:
    print "dataset_size error"
    exit()

# Prepare multi-layer perceptron model, defined in net.py
if args.net == 'simple':
    model = L.Classifier(net.MnistMLP(784, n_units, 10))
    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()
    xp = np if args.gpu < 0 else cuda.cupy
elif args.net == 'parallel':
    cuda.check_cuda_available()
    model = L.Classifier(net.MnistMLPParallel(784, n_units, 10))
    xp = cuda.cupy

# Setup optimizer
optimizer = optimizers.AdaDelta()
optimizer.setup(model)

# Init/Resume
if args.initmodel:
    print 'Load model from {}'.format(args.initmodel)
    serializers.load_hdf5(args.initmodel, model)
if args.resume:
    print 'Load optimizer state from {}'.format(args.resume)
    serializers.load_hdf5(args.resume, optimizer)

# Plot init
plt.style.use('ggplot')
train_loss = []
train_acc  = []
test_loss = []
test_acc  = []

n_train_batches = N_train / batchsize
# early stopping
patience = 5000
patience_increase = 2
improvement_threshold = 0.995
validation_frequency = min(n_train_batches, patience / 2)
best_validation_loss = np.inf
test_score = 0
done_looping = False

# Learning loop
epoch = 0
while (epoch < n_epoch) and (not done_looping):
    epoch = epoch + 1
    print 'epoch {}'.format(epoch)

    # training
    perm = np.random.permutation(N_train)
    sum_train_accuracy = 0
    sum_train_loss = 0
    for i in xrange(0, N_train, batchsize):
        x = chainer.Variable(xp.asarray(x_train[perm[i:i + batchsize]]))
        t = chainer.Variable(xp.asarray(y_train[perm[i:i + batchsize]]))

        # Pass the loss function (Classifier defines it) and its arguments
        optimizer.update(model, x, t)
        
        sum_train_loss += float(model.loss.data) * len(t.data)
        sum_train_accuracy += float(model.accuracy.data) * len(t.data)
        
        # generate network graph
        # if epoch == 1 and i == 0:
            # with open('netgraph.dot', 'w') as o:
                # g = computational_graph.build_computational_graph(
                    # (model.loss, ), remove_split=True)
                # o.write(g.dump())
            # print 'net graph generated'
        
        # validation
        batch_index = (i / batchsize)
        iter = (epoch - 1) * n_train_batches + batch_index
        if (iter + 1) % validation_frequency == 0:
            sum_validate_accuracy = 0
            sum_validate_loss = 0
            for i in xrange(0, N_validate, batchsize):
                x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                                     volatile='on')
                t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                                     volatile='on')
                loss = model(x, t)
                sum_validate_loss += float(loss.data) * len(t.data)
                sum_validate_accuracy += float(model.accuracy.data) * len(t.data)

            this_validate_loss = sum_validate_loss / N_validate
            this_validate_accuracy = sum_validate_accuracy / N_validate
            
            print 'validation epoch{}, minibatch{}/{}'.format(epoch, batch_index + 1, n_train_batches)
            print '      mean loss={}, accuracy={}'.format(
                this_validate_loss, sum_validate_accuracy / N_validate)
            
            if this_validate_loss < best_validation_loss:
                if this_validate_loss < best_validation_loss * improvement_threshold:
                    patience = max(patience, iter * patience_increase)
                    print " iter {} / patience {}".format(iter+1, patience)
                
                best_validation_loss = this_validate_loss

        if patience <= iter:
            done_looping = True
            break

    train_loss.append(sum_train_loss / N_train)
    train_acc.append(sum_train_accuracy / N_train)
    print 'train mean loss={}, accuracy={}'.format(
        sum_train_loss / N_train, sum_train_accuracy / N_train)

    # evaluation
    sum_test_accuracy = 0
    sum_test_loss = 0
    for i in xrange(0, N_test, batchsize):
        x = chainer.Variable(xp.asarray(x_test[i:i + batchsize]),
                             volatile='on')
        t = chainer.Variable(xp.asarray(y_test[i:i + batchsize]),
                             volatile='on')
        loss = model(x, t)
        sum_test_loss += float(loss.data) * len(t.data)
        sum_test_accuracy += float(model.accuracy.data) * len(t.data)

    test_loss.append(sum_test_loss / N_test)
    test_acc.append(sum_test_accuracy / N_test)
    print 'test  mean loss={}, accuracy={}'.format(
        sum_test_loss / N_test, sum_test_accuracy / N_test)
    

print 'train finish'
print 'draw graph'
# draw graph
plt.figure(figsize=(8,6))
plt.xlim([0, epoch])
plt.ylim([0.975, 1.0])
def add1list(list):
    return map(lambda item: item+1, list)
plt.plot(add1list(xrange(len(train_acc))), train_acc)
plt.plot(add1list(xrange(len(test_acc))), test_acc)
plt.legend(["train_acc","test_acc"],loc=4)
plt.title("Accuracy of digit recognition.")
plt.plot()

# Save the model and the optimizer
print 'save the model'
model.to_cpu()
serializers.save_hdf5('mlp.model', model)
print 'save the optimizer'
serializers.save_hdf5('mlp.state', optimizer)

finishtime = time.time()
print 'execute time = {}'.format(finishtime - starttime)

plt.savefig("graph.png")
plt.show()

# 対話的コンソール グラフスケール変えたい時とかに
# code.InteractiveConsole(globals()).interact()
