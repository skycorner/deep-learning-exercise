# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
m1=tf.constant([[3,3]])
m2=tf.constant([[2],[2]])
product = tf.matmul(m1,m2)     #matrix multiply  np.dot(m1,m2)


### method1
sess=tf.Session()
result = sess.run(product)
print(result)
sess.close()

### method2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)
