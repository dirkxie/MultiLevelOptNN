import tensorflow as tf
import numpy as np

def testfunc():
    graph1 = tf.Graph()
    with graph1.as_default():
        n1 = tf.constant(100, tf.float32, name='n1')
        n2 = tf.constant(50, tf.float32, name='n2')
        add = n1 + n2

        with tf.Session() as sess:
            return(sess.run(add))


print(testfunc())
