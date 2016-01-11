import os
# os.environ["CHAINER_TYPE_CHECK"] = "0"

import argparse
import numpy as np
import cupy as cp
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
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')
args = parser.parse_args()

batchsize = 100
n_epoch = 10000
n_units = 10

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

def evolve(model):
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
            x = chainer.Variable(cp.asarray(x_train[perm[i:i + batchsize]]))
            t = chainer.Variable(cp.asarray(y_train[perm[i:i + batchsize]]))

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
                    x = chainer.Variable(cp.asarray(x_test[i:i + batchsize]),
                                         volatile='on')
                    t = chainer.Variable(cp.asarray(y_test[i:i + batchsize]),
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
            x = chainer.Variable(cp.asarray(x_test[i:i + batchsize]),
                                 volatile='on')
            t = chainer.Variable(cp.asarray(y_test[i:i + batchsize]),
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
    plt.ylim([0.95, 1.0])
    plt.plot(xrange(1,len(train_acc)+1), train_acc)
    plt.plot(xrange(1,len(test_acc)+1), test_acc)
    plt.legend(["train_acc","test_acc"],loc=4)
    plt.title("Accuracy of digit recognition.")
    plt.plot()

    # Save the model and the optimizer
    print 'save the model'
    model.to_cpu()
    serializers.save_hdf5("v%5d.model" % (version), model)
    print 'save the optimizer'
    serializers.save_hdf5('v%5d.state' % (version), optimizer)

    finishtime = time.time()
    print 'execute time = {}'.format(finishtime - starttime)

    plt.savefig("v%5d_graph.png" % (version))
    # plt.show()
    
    return sum_test_accuracy / N_test


# いろいろいじる
def initWeights(model, prev_n_units, n_units):
    model.to_cpu()
    if prev_n_units > n_units:
        print "Error: not implement!"
        exit()
    pW1 = model.predictor.l1.W.data
    pb1 = model.predictor.l1.b.data
    pW2 = model.predictor.l2.W.data
    pb2 = model.predictor.l2.b.data
    
    offset = n_units - prev_n_units
    W1 = np.append(pW1,
        np.random.normal(0, 1 * np.sqrt(1. / 784), (offset, 784)),
        axis=0)
    b1 = np.append(pb1,
        np.zeros(offset),
        axis=0)
    W2 = np.append(pW2,
        np.random.normal(0, 1 * np.sqrt(1. / offset), (10, offset)),
        axis=1)
    
    return W1, b1, W2, pb2


version = 0
while version < 100:
    version = version + 1
    
    if version == 1:
        model = L.Classifier(net.MnistMLP(784, n_units, 10))
    else:
        prev_model = L.Classifier(net.MnistMLP(784, prev_n_units, 10, ))
        cuda.get_device(args.gpu).use()
        model.to_gpu()

        serializers.load_hdf5("v%5d.model" % (version-1), prev_model)

        W1, b1, W2, b2 = initWeights(prev_model, prev_n_units, n_units)

        model = L.Classifier(net.MnistMLP(784, n_units, 10, initW1=W1, initb1=b1, initW2=W2, initb2=b2))
    cuda.get_device(args.gpu).use()
    model.to_gpu()

    optimizer = optimizers.AdaDelta()
    optimizer.setup(model)

    evolve(model)
    
    
    prev_n_units = n_units
    n_units = prev_n_units + 10
    



# 対話的コンソール グラフスケール変えたい時とかに
# code.InteractiveConsole(globals()).interact()
