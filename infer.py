import io
import os

import tensorflow as tf
import convolutional_autoencoder
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

def image_segmentation(input_image,model_dir,out,network):

    

    checkpoint = model_dir

    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise IOError('No model found in {}.'.format(checkpoint))


        image = np.array(ndimage.imread(input_image))  # load grayscale
        image = cv2.resize(image, (network.IMAGE_HEIGHT, network.IMAGE_WIDTH))
        image = np.multiply(image, 1.0/255)
        # cv2.imshow('image', image)
        # cv2.waitKey(0)

        print(image.shape)

        segmentation = sess.run(network.segmentation_result, feed_dict={
            network.inputs: np.reshape(image, [1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1])})

        segmented_image = np.dot(segmentation[0], 255)

        # print(segmentation[0].shape)
        # fig, axs = plt.subplots(2, 1, figsize=(1 * 3, 10))
        # axs[0].imshow(image, cmap='gray')
        # axs[1].imshow(np.reshape(segmentation[0],(128,128)), cmap='gray')

        # plt.show()
        name=input_image.split(".")[0]
        name=name[len(name)-10:]

        cv2.imwrite(os.path.join(out, name+'.jpg'), segmented_image)
        # plt.savefig(os.path.join(args.out, 'result.jpg'))
        # plt.waitforbuttonpress()
