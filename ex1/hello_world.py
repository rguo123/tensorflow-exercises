'''
This file uses tf.print to print hello world.
'''

import tensorflow as tf

with tf.compat.v1.Session() as sess:
    f = tf.print("Hello World!")
    result = sess.run(f)