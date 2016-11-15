from datetime import datetime
import time
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def train_load_net(input_graph, params, train_dir, log_device_placement, max_steps, batch_size):    
    with tf.Session(graph=input_graph, config=tf.ConfigProto(
          log_device_placement=log_device_placement)) as sess:
      
      print('0. train_load_net')
      # initialize
      init = tf.initialize_all_variables()
      # start running operations on the graph
      sess.run(init)
      
      # Start the queue runners
      tf.train.start_queue_runners(sess=sess)
      summary_write = tf.train.SummaryWriter(train_dir, sess.graph)
              
      print('1. restore all variables from params to current graph')
      for var in tf.all_variables():
          var_name = var.name
          var_shape = var.get_shape()
          
          ndarray_in_params = params[var_name]
          assign_value_to_graph_variable_op = tf.assign(var, ndarray_in_params)
          sess.run(assign_value_to_graph_variable_op)
          
          print('----------')
          print(var_name)
          print(var_shape)
          print('restored')

      print('2. start/resume training')
      global_step = sess.graph.get_tensor_by_name("global_step:0")
      cur_step = sess.run(global_step)
      print('current step is %s' % cur_step)
      
      
      for step in xrange(cur_step, max_steps):
          start_time = time.time()
          
          train_op = sess.graph.get_operation_by_name('train')
          loss_op  = sess.graph.get_operation_by_name('total_loss').outputs[0]
          _ = sess.run(train_op)
          loss_value = sess.run(loss_op)
          #_, loss_value = sess.run([train_op, loss_op])
          duration = time.time() - start_time
          #print(type(loss_value))
          assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
      
          if step % 10 == 0:
            num_examples_per_step = batch_size
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = float(duration)
    
            format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print (format_str % (datetime.now(), step, loss_value,
                                 examples_per_sec, sec_per_batch))
    
          if step % 100 == 0:
            summary_str = sess.run(summary_op)
            summary_writer.add_summary(summary_str, step)
    
          # Save the model checkpoint periodically.
          if step % 1000 == 0 or (step + 1) == max_steps:
            checkpoint_path = os.path.join(train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)
