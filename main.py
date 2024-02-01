# This is a sample Python script.
##交叉熵导数的快速求法：https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt
import numpy as np
import os
import gzip
from six.moves import cPickle as pickle
import platform
def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
def fc_forward(z, W, b):
    """
    The forward propagation of a fully connected layer
    :param z: Output of this layer, shape: (N,ln)
    :param W: Weight of this layer,
    :param b: Bias of this layer,
    :return:  Output of next layer,
    """
    return np.dot(z, W) + b
def fc_backward(next_dz, W, z):
    """
    The backward propagation of a fully connected layer
    :param next_dz: gradients of next layer,
    :param W: gradients of this layer,
    :param z: output of this layer,
    :return:
    """
    N = z.shape[0]
    dz = np.dot(next_dz, W.T)
    dw = np.dot(z.T, next_dz)
    db = np.sum(next_dz, axis=0)
    return dw / N, db / N, dz
def relu_forward(z):
    """
    ReLU forward propagation
    :param z: layer before activation
    :return: activated result
    """
    return np.maximum(0, z)
def relu_backward(next_dz, z):
    """
    ReLU backward propagation
    :param next_dz: activated gradient
    :param z: value before activation
    :return:
    """
    dz = np.where(np.greater(z, 0), next_dz, 0)
    return dz
def cross_entropy_loss(y_predict, y_true):
    """
    cross-entropy loss function
    :param y_predict: prediction,shape (N,d)，
    :param y_true: true value,shape(N,d)
    :return:
    """

    y_shift = y_predict - np.max(y_predict, axis=-1, keepdims=True)
    y_exp = np.exp(y_shift)
    y_probability = y_exp / np.sum(y_exp, axis=-1,keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # loss function
    dy = y_probability - y_true
    return loss, dy

# 定义前向过程
def forward(X):
    nuerons["z2"]=fc_forward(X,weights["W1"],weights["b1"])
    nuerons["z2_relu"]=relu_forward(nuerons["z2"])
    nuerons["z3"]=fc_forward(nuerons["z2_relu"],weights["W2"],weights["b2"])
    nuerons["z3_relu"]=relu_forward(nuerons["z3"])
    nuerons["y"]=fc_forward(nuerons["z3_relu"],weights["W3"],weights["b3"])
    return nuerons["y"]

def backward(X,y_true):
    loss,dy=cross_entropy_loss(nuerons["y"],y_true)
    gradients["W3"],gradients["b3"],gradients["z3_relu"]=fc_backward(dy,weights["W3"],nuerons["z3_relu"])
    gradients["z3"]=relu_backward(gradients["z3_relu"],nuerons["z3"])
    gradients["W2"],gradients["b2"],gradients["z2_relu"]=fc_backward(gradients["z3"],
                                                                     weights["W2"],nuerons["z2_relu"])
    gradients["z2"]=relu_backward(gradients["z2_relu"],nuerons["z2"])
    gradients["W1"],gradients["b1"],_=fc_backward(gradients["z2"],
                                                    weights["W1"],X)
    return loss
def get_accuracy(X,y_true):
    y_predict=forward(X)
    return np.mean(np.equal(np.argmax(y_predict,axis=-1),
                            np.argmax(y_true,axis=-1)))

# load pickle based on python version 2 or 3
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))
def load_mnist_datasets(path='mnist.pkl.gz'):
    if not os.path.exists(path):
        raise Exception('Cannot find %s' % path)
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = load_pickle(f)
        return train_set, val_set, test_set


def to_categorical(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix.

    E.g. for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=np.float32)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


# 定义权重、神经元、梯度
weights={}
weights_scale=1e-3
weights["W1"]=weights_scale*np.random.randn(28*28,256)
weights["b1"]=np.zeros(256)
weights["W2"]=weights_scale*np.random.randn(256,256)
weights["b2"]=np.zeros(256)
weights["W3"]=weights_scale*np.random.randn(256,10)
weights["b3"]=np.zeros(10)

nuerons={}
gradients={}
train_set, val_set, test_set = load_mnist_datasets('mnist.pkl.gz')
train_y,val_y,test_y=to_categorical(train_set[1]),to_categorical(val_set[1]),to_categorical(test_set[1])
#Randomly selecting training samples
train_num = train_set[0].shape[0]
def next_batch(batch_size):
    idx=np.random.choice(train_num,batch_size)
    return train_set[0][idx],train_y[idx]
x,y= next_batch(16)
print("x.shape:{},y.shape:{}".format(x.shape,y.shape))

# 初始化变量
batch_size = 32
epoch = 3
steps = train_num // batch_size
lr = 0.1

for e in range(epoch):
    for s in range(steps):
        X, y = next_batch(batch_size)

        # forward and backward
        forward(X)
        loss = backward(X, y)

        # undate gradients
        for k in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            weights[k] -= lr * gradients[k]

        if s % 500 == 0:
            print("\n epoch:{} step:{} ; loss:{}".format(e, s, loss))
            print(" train_acc:{};  val_acc:{}".format(get_accuracy(X, y), get_accuracy(val_set[0], val_y)))

print("\n final result test_acc:{};  val_acc:{}".
      format(get_accuracy(test_set[0], test_y), get_accuracy(val_set[0], val_y)))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    x, y = test_set[0][6], test_y[6]
    plt.imshow(np.reshape(x, (28, 28)))
    plt.show()

    y_predict = np.argmax(forward([x])[0])

    print("y_true:{},y_predict:{}".format(np.argmax(y), y_predict))
    print_hi('lyy')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
 
