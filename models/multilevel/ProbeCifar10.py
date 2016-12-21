#%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import os
import cifar10_helper_func as cifar10
import prettytensor as pt
import sys
sys.dont_write_bytecode=True

# check tf version
print 'tensorflow version: ' + str(tf.__version__)

cifar10.data_path = "/curr/junyi/MultiLevelOptNN/dataset"
logs_path = "/curr/junyi/MultiLevelOptNN/models/multilevel/logs"

images_train, cls_train, labels_train = cifar10.load_training_data()
images_test, cls_test, labels_test = cifar10.load_test_data()

print("Size of:")
print("- Training-set:\t\t{}".format(len(images_train)))
print("- Test-set:\t\t{}".format(len(images_test)))

from cifar10_helper_func import img_size, num_channels, num_classes

img_size_cropped = 24
train_batch_size = 128

def pre_process_image(image, training):
    # This function takes a single image as input,
    # and a boolean whether to build the training or testing graph.
    
    if training:
        # For training, add the following to the TensorFlow graph.

        # Randomly crop the input image.
        image = tf.random_crop(image, size=[img_size_cropped, img_size_cropped, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)

        # Some of these functions may overflow and result in pixel
        # values beyond the [0, 1] range. It is unclear from the
        # documentation of TensorFlow 0.10.0rc0 whether this is
        # intended. A simple solution is to limit the range.

        # Limit the image pixels between [0, 1] in case of overflow.
        image = tf.minimum(image, 1.0)
        image = tf.maximum(image, 0.0)
    else:
        # For training, add the following to the TensorFlow graph.

        # Crop the input image around the centre so it is the same
        # size as images that are randomly cropped during training.
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=img_size_cropped,
                                                       target_width=img_size_cropped)

    return image

def pre_process(images, training):
    # Use TensorFlow to loop over all the input images and call
    # the function above which takes a single image as input.
    images = tf.map_fn(lambda image: pre_process_image(image, training), images)

    return images

def random_batch():
    # Number of images in the training-set.
    num_images = len(images_train)

    # Create a random index.
    idx = np.random.choice(num_images,
                           size=train_batch_size,
                           replace=False)

    # Use the random index to select random images and labels.
    x_batch = images_train[idx, :, :, :]
    y_batch = labels_train[idx, :]

    return x_batch, y_batch


def main():
    graph1 = tf.Graph()
    with graph1.as_default():
        global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
        starter_learning_rate = 0.002
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 30, 0.999, staircase=True)

        x = tf.placeholder('float',[None,32,32,3])
        y_true = tf.placeholder('float',[None,10])
        keep_prob = tf.placeholder('float')


        weights = {
            'wc1'  : tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=1e-4)), 
            'wp_c1': tf.Variable(tf.truncated_normal(shape=[16*16*64, 10], stddev=0.1)),
            'wc2'  : tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=1e-4)), 
            'wp_c2': tf.Variable(tf.truncated_normal(shape=[8*8*64, 10], stddev=0.1)),
            'wd1'  : tf.Variable(tf.truncated_normal(shape=[8*8*64, 384], stddev=0.04)),
            'wp_d1': tf.Variable(tf.truncated_normal(shape=[384,10], stddev=0.1)),
            'wd2'  : tf.Variable(tf.truncated_normal(shape=[384, 192], stddev=0.04)),
            'wp_d2': tf.Variable(tf.truncated_normal(shape=[192,10], stddev=0.1)),
            'wd3'  : tf.Variable(tf.truncated_normal(shape=[192, 10], stddev=1/192))
        }
        
        biases = {
            'bc1'  : tf.Variable(tf.constant(0.1, shape=[64])),
            'bp_c1': tf.Variable(tf.constant(0.1, shape=[10])),
            'bc2'  : tf.Variable(tf.constant(0.1, shape=[64])),
            'bp_c2': tf.Variable(tf.constant(0.1, shape=[10])),
            'bd1'  : tf.Variable(tf.constant(0.1, shape=[384])),
            'bp_d1': tf.Variable(tf.constant(0.1, shape=[10])),
            'bd2'  : tf.Variable(tf.constant(0.1, shape=[192])),
            'bp_d2': tf.Variable(tf.constant(0.1, shape=[10])),
            'bd3'  : tf.Variable(tf.constant(0.1, shape=[10]))
        }
        
        # conv1
        conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1,1,1,1], padding='SAME', name='conv1')
        bias1 = tf.nn.bias_add(conv1, biases['bc1'], name='bias1')
        relu1 = tf.nn.relu(bias1, name='relu1')
        pool1 = tf.nn.max_pool(relu1, ksize=[1,3,3,1], strides = [1,2,2,1], padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
	
        norm1_flatten = tf.reshape(norm1, [-1, 16*16*64])

        # probe at conv1
        probe_conv1_stop      = tf.stop_gradient(norm1_flatten, name='probe_conv1_stop')
        probe_conv1_fc        = tf.matmul(probe_conv1_stop, weights['wp_c1'], name='probe_conv1_fc')
        probe_conv1_bias      = tf.add(probe_conv1_fc, biases['bp_c1'], name='probe_conv1_bias')
        probe_conv1_softmax   = tf.nn.softmax_cross_entropy_with_logits(probe_conv1_bias, y_true, name='probe_conv1_softmax')
        probe_conv1_cost      = tf.reduce_mean(probe_conv1_softmax, name='probe_conv1_cost')
        probe_conv1_correct_prediction = tf.equal(tf.argmax(probe_conv1_bias, 1), tf.argmax(y_true, 1))
        probe_conv1_accuracy  = tf.reduce_mean(tf.cast(probe_conv1_correct_prediction, 'float'))
        probe_conv1_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(probe_conv1_cost, global_step=global_step)
        tf.scalar_summary("probe_conv1_loss", probe_conv1_cost)
        tf.scalar_summary("probe_conv1_accuracy", probe_conv1_accuracy)

        # conv2
        conv2 = tf.nn.conv2d(norm1, weights['wc2'], strides=[1,1,1,1], padding='SAME', name='conv2')
        bias2 = tf.nn.bias_add(conv2, biases['bc2'], name='bias2')
        relu2 = tf.nn.relu(bias2, name='relu2')
        pool2 = tf.nn.max_pool(relu2, ksize=[1,3,3,1], strides = [1,2,2,1], padding='SAME', name='pool2')
        
        pool2_flatten = tf.reshape(pool2, [-1, 8*8*64])
        
        # probe at conv2
        probe_conv2_stop      = tf.stop_gradient(pool2_flatten, name='probe_conv2_stop')
        probe_conv2_fc        = tf.matmul(probe_conv2_stop, weights['wp_c2'], name='probe_conv2_fc')
        probe_conv2_bias      = tf.add(probe_conv2_fc, biases['bp_c2'], name='probe_conv2_bias')
        probe_conv2_softmax   = tf.nn.softmax_cross_entropy_with_logits(probe_conv2_bias, y_true, name='probe_conv2_softmax')
        probe_conv2_cost      = tf.reduce_mean(probe_conv2_softmax, name='probe_conv2_cost')
        probe_conv2_correct_prediction = tf.equal(tf.argmax(probe_conv2_bias, 1), tf.argmax(y_true, 1))
        probe_conv2_accuracy  = tf.reduce_mean(tf.cast(probe_conv2_correct_prediction, 'float'))
        probe_conv2_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(probe_conv2_cost, global_step=global_step)
        tf.scalar_summary("probe_conv2_loss", probe_conv2_cost)
        tf.scalar_summary("probe_conv2_accuracy", probe_conv2_accuracy)

        # fc1
        fc1   = tf.matmul(pool2_flatten, weights['wd1'], name='fc1')
        bias1 = tf.add(fc1, biases['bd1'], name='bias1')
        relu1 = tf.nn.relu(bias1, name='relu1')
       
        # probe at relu1
        probe_fc1_stop      = tf.stop_gradient(relu1, name='probe_fc1_stop')
        probe_fc1_fc        = tf.matmul(probe_fc1_stop, weights['wp_d1'], name='probe_fc1_fc')
        probe_fc1_bias      = tf.add(probe_fc1_fc, biases['bp_d1'], name='probe_fc1_bias')
        probe_fc1_softmax   = tf.nn.softmax_cross_entropy_with_logits(probe_fc1_bias, y_true, name='probe_fc1_softmax')
        probe_fc1_cost      = tf.reduce_mean(probe_fc1_softmax, name='probe_fc1_cost')
        probe_fc1_correct_prediction = tf.equal(tf.argmax(probe_fc1_bias, 1), tf.argmax(y_true, 1))
        probe_fc1_accuracy  = tf.reduce_mean(tf.cast(probe_fc1_correct_prediction, 'float'))
        probe_fc1_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(probe_fc1_cost, global_step=global_step)
        tf.scalar_summary("probe_fc1_loss", probe_fc1_cost)
        tf.scalar_summary("probe_fc1_accuracy", probe_fc1_accuracy)

        # fc2
        fc2      = tf.matmul(relu1, weights['wd2'], name='fc2')
        bias2    = tf.add(fc2, biases['bd2'], name='bias2')
        fc2_drop = tf.nn.dropout(bias2, keep_prob)
        
        # probe at bias2
        probe_fc2_stop      = tf.stop_gradient(bias2, name='probe_fc2_stop')
        probe_fc2_fc        = tf.matmul(probe_fc2_stop, weights['wp_d2'], name='probe_fc2_fc')
        probe_fc2_bias      = tf.add(probe_fc2_fc, biases['bp_d2'], name='probe_fc2_bias')
        probe_fc2_softmax   = tf.nn.softmax_cross_entropy_with_logits(probe_fc2_bias, y_true, name='probe_fc2_softmax')
        probe_fc2_cost      = tf.reduce_mean(probe_fc2_softmax, name='probe_fc2_cost')
        probe_fc2_correct_prediction = tf.equal(tf.argmax(probe_fc2_bias, 1), tf.argmax(y_true, 1))
        probe_fc2_accuracy  = tf.reduce_mean(tf.cast(probe_fc2_correct_prediction, 'float'))
        probe_fc2_optimizer = tf.train.AdamOptimizer(learning_rate).minimize(probe_fc2_cost, global_step=global_step)
        tf.scalar_summary("probe_fc2_loss", probe_fc2_cost)
        tf.scalar_summary("probe_fc2_accuracy", probe_fc2_accuracy)

        # fc3
        fc3   = tf.matmul(fc2_drop, weights['wd3'], name='fc3')
        bias3 = tf.add(fc3, biases['bd3'], name='bias3')
        
        softmax_bias3 = tf.nn.softmax_cross_entropy_with_logits(bias3, y_true, name='softmax')
        cost_bias3    = tf.reduce_mean(softmax_bias3, name='cost_bias3')
        
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost_bias3, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(bias3, 1), tf.argmax(y_true, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        
        summary_op = tf.merge_all_summaries()

    #sess = tf.InteractiveSession()
    with tf.Session(graph=graph1) as sess:
        saver = tf.train.Saver()
        save_dir = 'checkpoints_cifar10/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = save_dir + 'cifar10_cnn'

        try:
            print("Trying to restore last checkpoint ...")
        
            # Use TensorFlow to find the latest checkpoint - if any.
            last_chk_path = tf.train.latest_checkpoint(checkpoint_dir=save_dir)
        
            # Try and load the data in the checkpoint.
            saver.restore(sess, save_path=last_chk_path)
        
            # If we get to this point, the checkpoint was successfully loaded.
            print("Restored checkpoint from:", last_chk_path)
        except:
            # If the above failed for some reason, simply
            # initialize all the variables for the TensorFlow graph.
            print("Failed to restore checkpoint. Initializing variables instead.")
            sess.run(tf.initialize_all_variables())
    
        
        writer = tf.train.SummaryWriter(logs_path, graph=tf.get_default_graph())
        start_time = time.time()
        for i in xrange(200000):
                batch_xs, batch_ys = random_batch()
       
                # training steps
        	"""
                if i % 50 == 0:
                    # probe_conv1
                    probe_conv1_train_accuracy = probe_conv1_accuracy.eval(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob:1.0})
                    probe_conv1_train_cost = probe_conv1_cost.eval(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 1.0})
                    # probe_conv2
                    probe_conv2_train_accuracy = probe_conv2_accuracy.eval(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob:1.0})
                    probe_conv2_train_cost = probe_conv2_cost.eval(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 1.0})

                    # probe_fc1
                    probe_fc1_train_accuracy = probe_fc1_accuracy.eval(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob:1.0})
                    probe_fc1_train_cost = probe_fc1_cost.eval(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 1.0})
                    # probe_fc2
                    probe_fc2_train_accuracy = probe_fc2_accuracy.eval(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob:1.0})
                    probe_fc2_train_cost = probe_fc2_cost.eval(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 1.0})
                    # final layer
                    train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_true: batch_ys, keep_prob: 1.0})
                    train_cost = cost_bias3.eval(feed_dict={x:batch_xs, y_true: batch_ys, keep_prob: 1.0})

                    print "=====================step %d=====================" % (sess.run(global_step))
                    print "conv1  : accuracy %g, loss %g" % (probe_conv1_train_accuracy, probe_conv1_train_cost)
                    print "conv2  : accuracy %g, loss %g" % (probe_conv2_train_accuracy, probe_conv2_train_cost)
                    print "fc1    : accuracy %g, loss %g" % (probe_fc1_train_accuracy, probe_fc1_train_cost)
                    print "fc2    : accuracy %g, loss %g" % (probe_fc2_train_accuracy, probe_fc2_train_cost)
                    print "training accuracy %g, loss %g" % (train_accuracy, train_cost)

                    end_time = time.time()
        	    print 'time: ',(end_time - start_time)
        	    start_time = end_time
                    print 'global_step ', sess.run(global_step), 'learning_rate ', sess.run(learning_rate)
                """
                # testing
                if (i+1) % 150 == 0:
                    conv1_avg = 0
                    conv2_avg = 0
                    fc1_avg = 0
                    fc2_avg = 0
                    avg = 0
        	    """
                    for j in xrange(20):
                        conv1_avg += probe_conv1_accuracy.eval(feed_dict={x: images_test[j*50:j*50+50], y_true: labels_test[j*50:j*50+50], keep_prob: 1.0})
                        conv2_avg += probe_conv2_accuracy.eval(feed_dict={x: images_test[j*50:j*50+50], y_true: labels_test[j*50:j*50+50], keep_prob: 1.0})
                        fc1_avg += probe_fc1_accuracy.eval(feed_dict={x: images_test[j*50:j*50+50], y_true: labels_test[j*50:j*50+50], keep_prob: 1.0})
                        fc2_avg += probe_fc2_accuracy.eval(feed_dict={x: images_test[j*50:j*50+50], y_true: labels_test[j*50:j*50+50], keep_prob: 1.0})
                        avg+=accuracy.eval(feed_dict={x: images_test[j*50:j*50+50], y_true: labels_test[j*50:j*50+50], keep_prob: 1.0})
        	    conv1_avg/=20
                    conv2_avg/=20
                    fc1_avg/=20
                    fc2_avg/=20
                    avg/=20
		    """
                    conv1_loss = probe_conv1_cost.eval(feed_dict={x: images_test, y_true: labels_test, keep_prob: 1.0})
                    conv1_avg  = probe_conv1_accuracy.eval(feed_dict={x: images_test, y_true: labels_test, keep_prob: 1.0})
                    conv2_loss = probe_conv2_cost.eval(feed_dict={x: images_test, y_true: labels_test, keep_prob: 1.0})
                    conv2_avg  = probe_conv2_accuracy.eval(feed_dict={x: images_test, y_true: labels_test, keep_prob: 1.0})
                    fc1_loss   = probe_fc1_cost.eval(feed_dict={x: images_test, y_true: labels_test, keep_prob: 1.0})
                    fc1_avg    = probe_fc1_accuracy.eval(feed_dict={x: images_test, y_true: labels_test, keep_prob: 1.0})
                    fc2_loss   = probe_fc2_cost.eval(feed_dict={x: images_test, y_true: labels_test, keep_prob: 1.0})
                    fc2_avg    = probe_fc2_accuracy.eval(feed_dict={x: images_test, y_true: labels_test, keep_prob: 1.0})
                    avg        = accuracy.eval(feed_dict={x: images_test, y_true: labels_test, keep_prob: 1.0})
                    loss       = cost_bias3.eval(feed_dict={x: images_test, y_true: labels_test, keep_prob: 1.0})

        	    print "--------------------test----------------------"
                    print "conv1: accuracy %g, loss %g" % (conv1_avg, conv1_loss)
                    print "conv2: accuracy %g, loss %g" % (conv2_avg, conv2_loss)
                    print "fc1  : accuracy %g, loss %g" % (fc1_avg, fc1_loss)
                    print "fc2  : accuracy %g, loss %g" % (fc2_avg, fc2_loss)
                    print "test : accuracy %g, loss %g" % (avg, loss)
        	    
        	    saver.save(sess, save_path=save_path, global_step=global_step)
                    print 'global_step ', sess.run(global_step), 'learning_rate ', sess.run(learning_rate)
                    print "Model saved in file: ", save_path
                
                # optimization
        	optimizer.run(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 0.5})
                probe_conv1_optimizer.run(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 0.5})
                probe_conv2_optimizer.run(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 0.5})
                probe_fc1_optimizer.run(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 0.5})
                probe_fc2_optimizer.run(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 0.5})
                summary = sess.run(summary_op, feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 0.5})
                # tensorboard summary
                writer.add_summary(summary, i)
        avg = 0
        for i in xrange(200):
        	avg+=accuracy.eval(feed_dict={x: images_test[i*50:i*50+50], y_true: labels_test[i*50:i*50+50], keep_prob: 1.0})
        avg/=200
        print "test accuracy %g"%avg

if __name__ == '__main__':
        main()
