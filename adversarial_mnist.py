from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import cPickle
import collections
from scipy.stats import norm
import random
# Import data
from tensorflow.examples.tutorials.mnist import input_data
import pylab
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
FLAGS = None
tf.set_random_seed(1234)
np.random.seed(1234)
random.seed(1234)
TRAIN_ITERS=1000000
M=20 # minibatch size
in_size=784 # Size of data in
out_size=1 # Size of label out (probability from data, a scalar)
mnist = input_data.read_data_sets('MNISt_Data', one_hot=True)

# MLP - used for D_pre, D_true, D_fake, G networks
def mlp(input, output_dim):
    hidden_1 = 300
    hidden_2 = 150
    # construct learnable parameters within local scope
    w1=tf.get_variable("w0", [input.get_shape()[1], hidden_1],
                       initializer=tf.random_normal_initializer())
    b1=tf.get_variable("b0", [hidden_1], initializer=tf.constant_initializer(0.0))
    w2=tf.get_variable("w1", [hidden_1, hidden_2], initializer=tf.random_normal_initializer())
    b2=tf.get_variable("b1", [hidden_2], initializer=tf.constant_initializer(0.0))
    w3=tf.get_variable("w2", [hidden_2,output_dim], initializer=tf.random_normal_initializer())
    b3=tf.get_variable("b2", [output_dim], initializer=tf.constant_initializer(0.0))
    # nn operators
    out_1=tf.nn.relu(tf.matmul(input,w1)+b1)
    out_2=tf.nn.relu(tf.matmul(out_1,w2)+b2)
    out_3=tf.nn.sigmoid(tf.matmul(out_2,w3)+b3)
    return out_3, [w1,b1,w2,b2,w3,b3]

# re-used for optimizing all networks
def momentum_optimizer(loss, weights):
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        0.001,                # Base learning rate.
        batch,  # Current index into the dataset.
        TRAIN_ITERS // 4,          # Decay step - this decays 4 times throughout training process.
        0.95,                # Decay rate.
        staircase=True)
    optimizer=tf.train.MomentumOptimizer(learning_rate,0.6).minimize(
        loss, global_step=batch, var_list=weights)
    return optimizer

# plot decision surface
def plot_d0(D,input_node):
    f,ax=plt.subplots(1)
    # p_data
    xs=np.linspace(-5,5,1000)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')
    # decision boundary
    r=1000 # resolution (number of points)
    xs=np.linspace(-5,5,r)
    ds=np.zeros((r,1)) # decision surface
    # process multiple points in parallel in a minibatch
    for i in range(int(r/M)):
        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M*i:M*(i+1)]=sess.run(D,{input_node: x})

    ax.plot(xs, ds, label='decision boundary')
    ax.set_ylim(0,1.1)
    plt.legend()
    plt.show()

def plot_fig():
    # plots pg, pdata, decision boundary 
    f,ax=plt.subplots(1)
    # p_data
    xs=np.linspace(-5,5,1000)
    ax.plot(xs, norm.pdf(xs,loc=mu,scale=sigma), label='p_data')

    # decision boundary
    r=5000 # resolution (number of points)
    xs=np.linspace(-5,5,r)
    ds=np.zeros((r,1)) # decision surface
    # process multiple points in parallel in same minibatch
    for i in range(int(r/M)):
        x=np.reshape(xs[M*i:M*(i+1)],(M,1))
        ds[M*i:M*(i+1)]=sess.run(D_true,{x_node: x})

    ax.plot(xs, ds, label='decision boundary')

    # distribution of inverse-mapped points
    zs=np.linspace(-5,5,r)
    gs=np.zeros((r,1)) # generator function
    for i in range(int(r/M)):
        z=np.reshape(zs[M*i:M*(i+1)],(M,1))
        gs[M*i:M*(i+1)]=sess.run(G,{z_node: z})
    histc, edges = np.histogram(gs, bins = 10)
    ax.plot(np.linspace(-5,5,10), histc/float(r), label='p_g')

    # ylim, legend
    ax.set_ylim(0,1.1)
    plt.legend()
    plt.show()

def plot_generated(generated, reconstruct_count=10):
    fig, ax = plt.subplots(1, reconstruct_count, figsize=(1.2*reconstruct_count, 3))
    for index, im in enumerate(random.sample(generated, reconstruct_count)):
        ax[index].imshow(np.reshape(im, (28, 28)))
    plt.show()


def noise_prior_sample(batch_size, in_size, scale=1.0):
    return np.array([np.random.random(in_size) * scale for _ in range(0, batch_size)])

def data_sample(batch_size, mnist):
    batch_x, _ = mnist.train.next_batch(batch_size)
    return batch_x

# Pretraining for faster convergence
with tf.variable_scope("D_pre"):
    input_node=tf.placeholder(tf.float32, shape=(M,in_size))
    train_labels=tf.placeholder(tf.float32,shape=(M,out_size))
    D_pre, weights_pre=mlp(input_node,out_size)
    loss=tf.reduce_mean(tf.square(D_pre - train_labels))

optimizer=momentum_optimizer(loss,None)
sess=tf.InteractiveSession()
tf.initialize_all_variables().run()
#plot_d0(D_pre, input_node)
#plt.title('Initial Decision Boundary')

lh=np.zeros(1000)
for i in range(1000):
    #d = (np.random.random(M)-0.5) * 10.0 # instead of sampling only from gaussian, want the domain to be covered as uniformly as possible
    #labels = norm.pdf(d,loc=mu,scale=sigma)
    d = data_sample(M, mnist)
    labels = np.ones(M)
    lh[i], _ = sess.run([loss,optimizer], {input_node: np.reshape(d,(M,in_size)),
                                           train_labels: np.reshape(labels,(M,out_size))})

#plt.plot(lh)
#plt.title('Training Loss')
#plot_d0(D_pre,input_node)
pretrained = sess.run(weights_pre)
sess.close()

with tf.variable_scope("Gen"):
    z_node=tf.placeholder(tf.float32, shape=(M,in_size)) # M uniform01 floats
    G, weights_gen=mlp(z_node, in_size) # generate normal transformation of Z - output size must match input size for D
    # G = tf.mul(5.0,G) # scale up by 5 to match range
with tf.variable_scope("Discr") as scope:
    # D(x)
    x_node = tf.placeholder(tf.float32, shape=(M,in_size)) # input M mnist samples
    out_prob, weights_discr=mlp(x_node, out_size) # output likelihood of being from data distrib
    D_true = tf.maximum(tf.minimum(out_prob, .99), 0.01) # clamp as a probability
    # make a copy of D that uses the same variables, but takes in G as input
    scope.reuse_variables()
    out_prob, weights_discr = mlp(G, out_size)
    D_fake = tf.maximum(tf.minimum(out_prob, .99), 0.01)
obj_discr=tf.reduce_mean(tf.log(D_true) + tf.log(1 - D_fake))
obj_gen=tf.reduce_mean(tf.log(D_fake))

# set up optimizer for G,D
opt_d = momentum_optimizer(1-obj_discr, weights_discr)
opt_g = momentum_optimizer(1-obj_gen, weights_gen) # maximize log(D(G(z)))

sess = tf.InteractiveSession()
tf.initialize_all_variables().run()
# copy weights from pre-training over to new D network
for i, w in enumerate(weights_discr):
    sess.run(w.assign(pretrained[i]))
#plot_fig()
#plt.title('Before Training')
# Algorithm 1 of Goodfellow et al 2014
k=1 # Iterations of discriminator for every generator iteration
histd, histg= np.zeros(TRAIN_ITERS), np.zeros(TRAIN_ITERS)
gen_samples = []
for i in range(TRAIN_ITERS):
    for j in range(k):
        x = data_sample(M, mnist)
        z = noise_prior_sample(M, in_size)  # sample m-batch from noise prior
        histd[i], _ =sess.run([obj_discr, opt_d], 
                              {x_node: np.reshape(x,(M,in_size)), z_node: np.reshape(z,(M,in_size))})
    z = noise_prior_sample(M, in_size)
    histg[i], _ =sess.run([obj_gen, opt_g], {z_node: np.reshape(z, (M, in_size))}) # update generator
    if i % (TRAIN_ITERS//10) == 0:
        print(float(i)/float(TRAIN_ITERS))
        samples = sess.run(G, {z_node: np.reshape(z, (M, in_size))})
        #plot_generated(samples)
        gen_samples.append(samples)
with open('/home/mifs/ds636/exps/mnist/gan/generated_samples', 'wb') as f_out:
    cPickle.dump(gen_samples, f_out)
'''
plt.plot(range(TRAIN_ITERS),histd, label='obj_discr')
plt.plot(range(TRAIN_ITERS), 1-histg, label='obj_gen')
plt.legend()
plt.show()
plot_fig()
with open('/home/mifs/ds636/exps/mnist/gan/generated_samples', 'rb') as f_in:
    gen_samples = cPickle.load(f_in)
    for sample_set in gen_samples:
        plot_generated(sample_set)
'''

