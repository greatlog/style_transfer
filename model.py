import collections
from network import *


def style_transfer(input1,input2,flags):
    Network = collections.namedtuple('Network', 'discrim_real_output, discrim_fake_output, discrim_loss, \
            adversarial_loss, content_loss, gen_output, train, global_step, learning_rate')

    with tf.variable_scope('generator'):
        gen_input = tf.concat([input1,input2],axis=-1)
        gen_output = generator(gen_input,flags)

    with tf.variable_scope('vgg19'):
        vgg19_features_1 = build_vgg19(gen_output)
        vgg19_features_2 = build_vgg19(input2)


    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=False):
            discrim_fake_output = discriminator(gen_output, FLAGS=flags)

    # Build the real discriminator
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            discrim_real_output = discriminator(input2,FLAGS=flags)

    with tf.variable_scope('generator_loss'):
        with tf.variable_scope('adversarial_loss'):
            adversarial_loss = tf.reduce_mean(-tf.log(discrim_fake_output + flags.EPS))

        with tf.variable_scope('content_loss'):
            content_loss = flags.vgg_scaling*tf.reduce_mean(tf.square(tf.subtract(vgg19_features_1[flags.vgg19_layer],
                                                            vgg19_features_2[flags.vgg19_layer])))

        gen_loss = content_loss + (flags.ratio) * adversarial_loss

    with tf.variable_scope('discriminator_loss'):
        discrim_fake_loss = tf.log(1 - discrim_fake_output + flags.EPS)
        discrim_real_loss = tf.log(discrim_real_output + flags.EPS)

        discrim_loss = tf.reduce_mean(-(discrim_fake_loss + discrim_real_loss))



    with tf.variable_scope('get_learning_rate_and_global_step'):
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(flags.learning_rate, global_step, flags.decay_step, flags.decay_rate,
                                                   staircase=flags.stair)
        incr_global_step = tf.assign(global_step, global_step + 1)

    with tf.variable_scope('discrim_train'):

        dis_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=flags.beta)
        dis_train_op = dis_optimizer.minimize(discrim_loss)

    with tf.variable_scope('gen_train'):
        with tf.control_dependencies([dis_train_op]):
            gen_optimizer = tf.train.AdamOptimizer(learning_rate,beta1=flags.beta)
            gen_train_op = gen_optimizer.minimize(gen_loss)


    exp_averager = tf.train.ExponentialMovingAverage(decay=0.99)
    update_loss = exp_averager.apply([discrim_loss, adversarial_loss, content_loss])

    return Network(
        discrim_real_output=discrim_real_output,
        discrim_fake_output=discrim_fake_output,
        discrim_loss=exp_averager.average(discrim_loss),
        adversarial_loss=exp_averager.average(adversarial_loss),
        content_loss=exp_averager.average(content_loss),
        gen_output=gen_output,
        train=tf.group(update_loss,incr_global_step, gen_train_op),
        global_step=global_step,
        learning_rate=learning_rate
    )

