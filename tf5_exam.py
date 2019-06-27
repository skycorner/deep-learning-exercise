# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 13:34:28 2019

@author: Administrator
"""

import tensorflow as tf
import numpy as np

#creat data
x_data=np.random.rand(100).astype(np.float32)
y_data=x_data*0.1+0.3

### creat tensorflow structure start ###
W=tf.Variable(tf.random_uniform([1],-1.0,1.0))
b=tf.Variable(tf.zeros([1]))
 
y=W*x_data+b
 
loss=tf.reduce_mean(tf.square(y-y_data))
optimizer=tf.train.GradientDescentOptimizer(0.5)
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()
### creat tensorflow structure start ###

sess=tf.Session()
sess.run(init)           ###very important
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step,sess.run(W),sess.run(b))