import tensorflow as tf
import numpy as np
from test_functions import op_in_func
sys.dont_write_bytecode = True

test_graph = tf.Graph()

with test_graph.as_default():
    n1 = tf.constant(100, tf.float32, name='n1')
    n2 = tf.constant(50, tf.float32, name='n2')
    add = tf.add(n1, n2, name='add')

op_in_func(input_graph = test_graph, op_name = "add")
