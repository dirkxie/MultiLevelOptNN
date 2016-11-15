import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

def extract_weights (ckpt_dir, saver):

  if ckpt_dir is not None:
    # Restoring from the checkpoint file
    print('1. Checkpoint path found.\n')
    print('Load checkpoint state:')
    ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
    print(ckpt_state)
  
  with tf.Session() as sess:  
    if ckpt_state is not None:
      saver.restore(sess, ckpt_state.model_checkpoint_path)
      print('2. Checkpoint restored.\n')
    else:
        print('2. No existing checkpoint.\n')
    
    params = {}
    
    print('3. Extracted weights info:')
    for var in tf.all_variables():
        var_name = var.name
        var_shape = var.get_shape()
        print(var_name)
        print(var_shape)
        params[var_name] = sess.run(var)
    
    sess.close()
    return params
