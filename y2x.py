# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 02:33:05 2022

@author: Jzh
"""

# y = 2 * x + 3

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

bucket_size = 300
sparse_len = 4
'''
with tf.variable_scope("scope", reuse=tf.AUTO_REUSE):
    s = tf.get_variable(
        #tf.random_normal([bucket_size ,sparse_len], mean=0, stddev=1),
        shape = [bucket_size, sparse_len],
        dtype=tf.float32, name="sparse_embedding")
    w = tf.get_variable(shape = [sparse_len, 1], dtype=tf.float32, name="weight")
    b = tf.get_variable(shape = [1], dtype=tf.float32, name="bias")
'''
with tf.variable_scope("self_define"):
    s = tf.Variable(
        tf.random_normal([bucket_size ,sparse_len], mean=0, stddev=1),
        dtype=tf.float32, name="sparse_embedding")
    w = tf.Variable(tf.random_normal([sparse_len, 1]), dtype=tf.float32, name="weight")
    b = tf.Variable(tf.random_normal([1]), dtype=tf.float32, name="bias")
    
    print(s, w, b)
    input = tf.placeholder(tf.int32, shape=(None,), name = "input")
    label = tf.placeholder(tf.float32, shape=(None, ), name = "label")
    emb = tf.nn.embedding_lookup(s, input)
    print("emb", emb)
    o = tf.add(tf.matmul(emb, w), b, name="logits")
    print("output", o, "label", label)
    loss = tf.reduce_sum(tf.square(o - label))
    
    
    optimizer = tf.train.GradientDescentOptimizer(0.001)
    #optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)
    
    g = tf.get_default_graph()
    write = tf.summary.FileWriter('./graph', graph= g)
    write.close()

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    ii = np.arange(10)
    for i in range(10000):
        x_train = np.random.random_sample((1)).astype(np.float32) * bucket_size % bucket_size
        x_train = x_train.astype(np.int32)
        y_train = x_train * 2 + 3
        y_train = y_train.astype(np.float32)
        #print(x_train, y_train)
        
        _, c_loss = sess.run([train, loss], feed_dict={input:x_train, label:y_train})
        if i % 100 == 0:
            print('iter step %d, loss %f' % (i, c_loss))
    # v = sess.run(o, feed_dict={input:[23]})
    # print(v, v.shape)
    # ss = sess.run(s)
    # ww = sess.run(w)
    # bb = sess.run(b)
    # print(ss, ss.shape, ww, ww.shape, bb, b.shape)
    
    ou = sess.run(o, feed_dict={input:ii})
    print(ou, ii * 2 + 3)

        