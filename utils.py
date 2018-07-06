import tensorflow as tf
from tensorflow import keras

def conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),
                        padding='SAME')
    return tf.nn.bias_add(conv, bias)


def pool_layer(input):
    return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                          padding='SAME')

def prelu_tf(inputs, name='Prelu'):
    with tf.variable_scope(name):
        alphas = tf.get_variable('alpha', inputs.get_shape()[-1], initializer=tf.zeros_initializer(), dtype=tf.float32)
    pos = tf.nn.relu(inputs)
    neg = alphas * (inputs - abs(inputs)) * 0.5

    return pos + neg

def lrelu_tf(inputs,alpha,name='lrelu'):

    pos = tf.nn.relu(inputs)
    neg = alpha * (inputs - abs(inputs)) * 0.5

    return pos + neg

def imgs_loder(list,flags):
    '''load many jpg images'''
    list = tf.convert_to_tensor(list,tf.string)

    queue = tf.train.slice_input_producer([list],num_epochs=flags.epochs)

    imgs_ = tf.read_file(queue[0])
    imgs_ = tf.image.decode_jpeg(imgs_)
    imgs_ = tf.reshape(imgs_, [flags.input_size,flags.input_size,flags.channels])
    imgs =tf.cast(imgs_,tf.float32)

    batch = tf.train.shuffle_batch([imgs],shapes=[[flags.input_size,flags.input_size,flags.channels]],
                                     batch_size=flags.batch_size,enqueue_many=False,
                                     capacity=flags.batch_size*10,min_after_dequeue=flags.batch_size)

    return batch