from tensorflow.contrib.layers import conv2d,batch_norm
from tensorflow.contrib.slim import flatten

from utils import *
import numpy as np
import scipy.io

def generator(input,flags):
    with tf.variable_scope('generator_unit'):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2d(input,num_outputs=64,kernel_size=9,stride=1, scope='conv')
            net = prelu_tf(net)

        stage1_output = net

        # The residual block parts
        for i in range(1, flags.num_resblock+1 , 1):
            name_scope = 'resblock_%d'%(i)
            net = residual_block(net, 64, 1, name_scope)

        with tf.variable_scope('resblock_output'):
            net = conv2d(net, kernel_size=3, num_outputs=64, stride=1 , scope='conv')
            net = batch_norm(net)

        net = net + stage1_output

        with tf.variable_scope('output_stage'):
            net = conv2d(net,num_outputs=3,kernel_size=3, scope='conv')

    return net

def discriminator(dis_inputs, FLAGS=None):
    if FLAGS is None:
        raise ValueError('No FLAGS is provided for generator')

    # Define the discriminator block
    def discriminator_block(inputs, output_channel, kernel_size, stride, scope):
        with tf.variable_scope(scope):
            net = conv2d(inputs, kernel_size, output_channel, stride, scope='conv1')
            net = batch_norm(net)
            net = lrelu_tf(net, 0.2)

        return net


    with tf.variable_scope('discriminator_unit'):
        # The input layer
        with tf.variable_scope('input_stage'):
            net = conv2d(dis_inputs, 3, 64, 1, scope='conv')
            net = lrelu_tf(net, 0.2)

        # The discriminator block part
        # block 1
        net = discriminator_block(net, 64, 3, 2, 'disblock_1')

        # block 2
        net = discriminator_block(net, 128, 3, 1, 'disblock_2')

        # block 3
        net = discriminator_block(net, 128, 3, 2, 'disblock_3')

        # block 4
        net = discriminator_block(net, 256, 3, 1, 'disblock_4')

        # block 5
        net = discriminator_block(net, 256, 3, 2, 'disblock_5')

        # block 6
        net = discriminator_block(net, 512, 3, 1, 'disblock_6')

        # block_7
        net = discriminator_block(net, 512, 3, 2, 'disblock_7')

        # The dense layer 1
        with tf.variable_scope('dense_layer_1'):
            net = flatten(net)
            net = tf.layers.dense(net, 1024)
            net = lrelu_tf(net, 0.2)

        # The dense layer 2
        with tf.variable_scope('dense_layer_2'):
            net = tf.layers.dense(net, 1)
            net = tf.nn.sigmoid(net)

    return net

def residual_block(inputs, output_channel, stride, scope):
    with tf.variable_scope(scope):
        net = conv2d(inputs, kernel_size=3, num_outputs=output_channel, stride = stride,scope='conv_1')
        net = batch_norm(net)
        net = prelu_tf(net)
        net = conv2d(net, kernel_size=3,num_outputs=output_channel, stride = stride, scope='conv_2')
        net = batch_norm(net)
        net = net + inputs

    return net



def build_vgg19(input_image,model_path='./imagenet-vgg-verydeep-19.mat'):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )
    data = scipy.io.loadmat(model_path)

    # mean = data['normalization'][0][0][0]
    # mean_pixel = np.mean(mean, axis=(0, 1))

    weights = data['layers'][0]

    net = {}
    current = input_image
    for i, name in enumerate(layers):

        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]

            kernels = np.transpose(kernels, (1, 0, 2, 3))

            bias = bias.reshape(-1)
            current = conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = pool_layer(current)
        net[name] = current
    assert len(net) == len(layers)
    print("Functions for VGG ready")
    return net

