# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 14:56:45 2019

@author: Administrator
"""

import tensorflow as tf
state=tf.Variable(0,name='counter')
print(state.name)
one=tf.constant(1)

new_value=tf.add(state,one) 
update = tf.assign(state,new_value)   ### new_value传给state
init=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)                  ####如果定义了Variable一定要有
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))