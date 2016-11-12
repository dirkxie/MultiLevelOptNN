import tensorflow as tf
import numpy as np

def op_in_func(input_graph, op_name):
  with tf.Session(graph=input_graph) as sess:
      op = sess.graph.get_operation_by_name(op_name).outputs[0]
      print(sess.run(op))

