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
            'wc1': tf.Variable(tf.truncated_normal(shape=[5, 5, 3, 64], stddev=0.1)), 
            'wc2': tf.Variable(tf.truncated_normal(shape=[5, 5, 64, 64], stddev=0.1)), 
            'wd1': tf.Variable(tf.truncated_normal(shape=[8*8*64, 384], stddev=0.1)),
            'wd2': tf.Variable(tf.truncated_normal(shape=[384, 192], stddev=0.1)),
            'wd3': tf.Variable(tf.truncated_normal(shape=[192, 10], stddev=0.1))
        }
        
        biases = {
            'bc1': tf.Variable(tf.constant(0.1, shape=[64])),
            'bc2': tf.Variable(tf.constant(0.1, shape=[64])),
            'bd1': tf.Variable(tf.constant(0.1, shape=[384])),
            'bd2': tf.Variable(tf.constant(0.1, shape=[192])),
            'bd3': tf.Variable(tf.constant(0.1, shape=[10]))
        }
        
        conv1 = tf.nn.conv2d(x, weights['wc1'], strides=[1,1,1,1], padding='SAME', name='conv1')
        bias1 = tf.nn.bias_add(conv1, biases['bc1'], name='bias1')
        relu1 = tf.nn.relu(bias1, name='relu1')
        pool1 = tf.nn.max_pool(relu1, ksize=[1,3,3,1], strides = [1,2,2,1], padding='SAME', name='pool1')
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')
        
        conv2 = tf.nn.conv2d(norm1, weights['wc2'], strides=[1,1,1,1], padding='SAME', name='conv2')
        bias2 = tf.nn.bias_add(conv2, biases['bc2'], name='bias2')
        relu2 = tf.nn.relu(bias2, name='relu2')
        pool2 = tf.nn.max_pool(relu2, ksize=[1,3,3,1], strides = [1,2,2,1], padding='SAME', name='pool2')
        
        pool2_flatten = tf.reshape(pool2, [-1, 8*8*64])
        
        fc1   = tf.matmul(pool2_flatten, weights['wd1'], name='fc1')
        bias1 = tf.add(fc1, biases['bd1'], name='bias1')
        relu1 = tf.nn.relu(bias1, name='relu1')
        
        fc2   = tf.matmul(relu1, weights['wd2'], name='fc2')
        bias2 = tf.add(fc2, biases['bd2'], name='bias2')
        
        fc2_drop = tf.nn.dropout(bias2, keep_prob)
        
        fc3   = tf.matmul(fc2_drop, weights['wd3'], name='fc3')
        bias3 = tf.add(fc3, biases['bd3'], name='bias3')
        
        softmax = tf.nn.softmax_cross_entropy_with_logits(bias3, y_true, name='softmax')
        cost  = tf.reduce_mean(softmax, name='cost')
        
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
        correct_prediction = tf.equal(tf.argmax(bias3, 1), tf.argmax(y_true, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

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
    
        
        #current_loss = 0.0
        #previous_loss = 100000000.0
        #global_step_np = global_step.eval()
        start_time = time.time()
        for i in xrange(200000):
        #while (np.absolute(current_loss - previous_loss) / previous_loss > 0.00000005):
                batch_xs, batch_ys = random_batch()
        
        	if i % 1000 == 0:
                #if global_step_np % 1000 == 0:
        	    train_accuracy = accuracy.eval(feed_dict={x:batch_xs, y_true: batch_ys, keep_prob: 1.0})
                    train_cost = cost.eval(feed_dict={x:batch_xs, y_true: batch_ys, keep_prob: 1.0})
                    print "step %d, training accuracy %g, loss %g" % (sess.run(global_step), train_accuracy, train_cost)
        	    
                    #previous_loss = current_loss
                    #current_loss = train_cost

                    end_time = time.time()
        	    print 'time: ',(end_time - start_time)
        	    start_time = end_time
                    print 'global_step ', sess.run(global_step), 'learning_rate ', sess.run(learning_rate)
            	
                if (i+1) % 3000 == 0:
                #if (global_step_np+1)%3000 == 0:
                    avg = 0
        	    for j in xrange(20):
        	    	avg+=accuracy.eval(feed_dict={x: images_test[j*50:j*50+50], y_true: labels_test[j*50:j*50+50], keep_prob: 1.0})
        	    avg/=20
        	    print "--------------------"
                    print "test accuracy %g"%avg
        	    
        	    saver.save(sess, save_path=save_path, global_step=global_step)
                    print "Model saved in file: ", save_path
                    print "--------------------"
                
        	optimizer.run(feed_dict={x: batch_xs, y_true: batch_ys, keep_prob: 0.5})

        avg = 0
        for i in xrange(200):
        	avg+=accuracy.eval(feed_dict={x: images_test[i*50:i*50+50], y_true: labels_test[i*50:i*50+50], keep_prob: 1.0})
        avg/=200
        print "test accuracy %g"%avg

if __name__ == '__main__':
        main()
