import tensorflow as tf
import numpy as np

graph2 = tf.Graph()

with graph2.as_default():
    n1 = tf.constant(100, tf.float32, name='n1')
    n2 = tf.constant(50, tf.float32, name='n2')
    add = tf.add(n1, n2, name='add')

with tf.Session(graph=graph2) as sess:
    add = sess.graph.get_operation_by_name('add').outputs[0]
    print(sess.run(add))

