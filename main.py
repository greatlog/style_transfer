from model import style_transfer
import tensorflow as tf
from utils import imgs_loder
from PIL import Image
import numpy as np
import os

Flags = tf.app.flags

# The system parameter
Flags.DEFINE_string('output_dir', None, 'The output directory of the checkpoint')
Flags.DEFINE_string('summary_dir', None, 'The dirctory to output the summary')
Flags.DEFINE_string('mode', 'train', 'The mode of the model train, test.')
Flags.DEFINE_string('train_dir1','my_pic','the path of data')
Flags.DEFINE_string('train_dir2','gakii_pic','the path of data')
Flags.DEFINE_string('checkpoint', None, 'If provided, the weight will be restored from the provided checkpoint')
Flags.DEFINE_boolean('pre_trained_model', False, 'If set True, the weight will be loaded but the global_step will still '
                                                 'be 0. If set False, you are going to continue the training. That is, '
                                                 'the global_step will be initiallized from the checkpoint, too')

Flags.DEFINE_integer('epochs',None,'number of epochs')
Flags.DEFINE_boolean('is_training', True, 'Training => True, Testing => False')

Flags.DEFINE_integer('num_resblock', 16, 'How many residual blocks are there in the generator')

Flags.DEFINE_string('vgg19_layer','conv5_4','layer in vgg19 to extract feature')
Flags.DEFINE_string('task', None, 'The task: discriminator, combined_model')

Flags.DEFINE_integer('batch_size',4 , 'Batch size of the input batch')
Flags.DEFINE_integer('input_size',512,'input size of the picture')
Flags.DEFINE_integer('channels',3,'number of image channels')

Flags.DEFINE_float('EPS', 1e-12, 'The eps added to prevent nan')
Flags.DEFINE_float('ratio', 0.001, 'The ratio between content loss and adversarial loss')
Flags.DEFINE_float('vgg_scaling', 0.0061, 'The scaling factor for the perceptual loss if using vgg perceptual loss')

Flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the network')
Flags.DEFINE_integer('decay_step', 500000, 'The steps needed to decay the learning rate')
Flags.DEFINE_float('decay_rate', 0.1, 'The decay rate of each decay step')
Flags.DEFINE_boolean('stair', False, 'Whether perform staircase decay. True => decay in discrete interval.')
Flags.DEFINE_float('beta', 0.9, 'The beta1 parameter for the Adam optimizer')
Flags.DEFINE_integer('max_epoch', None, 'The max epoch for the training')
Flags.DEFINE_integer('max_iter', 1000000, 'The max iteration of the training')
Flags.DEFINE_integer('display_freq', 20, 'The diplay frequency of the training process')
Flags.DEFINE_integer('summary_freq', 100, 'The frequency of writing summary')
Flags.DEFINE_integer('save_freq', 10000, 'The frequency of saving images')


flags = Flags.FLAGS

if flags.output_dir is None:
    raise ValueError('The output directory is needed')

# Check the output directory to save the checkpoint
if not os.path.exists(flags.output_dir):
    os.mkdir(flags.output_dir)

# Check the summary directory to save the event
if not os.path.exists(flags.summary_dir):
    os.mkdir(flags.summary_dir)


if flags.mode == 'train':
    with tf.device('/gpu:0'):

        list1_ = os.listdir(flags.train_dir1)
        list1 = [os.path.join(flags.train_dir1,_) for _ in list1_ if _.split('.')[1]=='jpg']

        list2_ = os.listdir(flags.train_dir2)
        list2 = [os.path.join(flags.train_dir2, _) for _ in list2_ if _.split('.')[1] == 'jpg']

        batch1 = imgs_loder(list1,flags)
        batch2 = imgs_loder(list2,flags)

        net = style_transfer(batch1,batch2,flags)

        tf.summary.scalar('discrim_loss',net.discrim_loss)
        tf.summary.scalar('content_loss', net.discrim_loss)
        tf.summary.scalar('adversarial_loss', net.adversarial_loss)

        tf.summary.scalar('learning_rate', net.learning_rate)

        saver = tf.train.Saver(max_to_keep=10)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        weight_initiallizer = tf.train.Saver(var_list2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Use superviser to coordinate all queue and summary writer
        sv = tf.train.Supervisor(logdir=flags.summary_dir, save_summaries_secs=0, saver=None)
        with sv.managed_session(config=config) as sess:

            if (flags.checkpoint is not None) and (flags.pre_trained_model is False):
                print('Loading model from the checkpoint...')
                checkpoint = tf.train.latest_checkpoint(flags.checkpoint)
                saver.restore(sess, checkpoint)

            elif (flags.checkpoint is not None) and (flags.pre_trained_model is True):
                print('Loading weights from the pre-trained model')
                weight_initiallizer.restore(sess, flags.checkpoint)

            print('Optimization starts!!!')
            acc_txt = os.path.join(flags.output_dir,'acc.txt')
            if os.path.exists(acc_txt):
                os.remove(acc_txt)
            txt_result = open(acc_txt,'a+')
            step = 0
            while True:
                try:
                    fetches = {
                        "train": net.train,
                        "content_loss":net.content_loss,
                        "adversarial_loss":net.adversarial_loss,
                        "discrim_loss":net.discrim_loss,
                        "global_step": sv.global_step,
                        "learning_rate":net.learning_rate
                    }

                    if ((step + 1) %flags.summary_freq) == 0:
                        fetches["summary"] = sv.summary_op


                    results = sess.run(fetches)

                    if ((step + 1) % flags.summary_freq) == 0:
                        print('Recording summary!!')
                        sv.summary_writer.add_summary(results['summary'], results['global_step'])

                    if ((step + 1) % flags.display_freq) == 0:


                        print("global_step", results["global_step"])
                        print("content_loss", results["content_loss"])
                        print("adversarial_loss", results["adversarial_loss"])
                        print("discrim_loss", results["discrim_loss"])
                        print("learning_rate", results['learning_rate'])

                        txt_result.write("global_step:%d,content_loss:%.4f,adversarial_loss:%.4f,\
                        discrim_loss:%.4f,learning_rate:%.4f\n" %
                                         (results["global_step"],results["content_loss"],results["adversarial_loss"],
                                          results["discrim_loss"],results["learning_rate"]))

                    if ((step + 1) % flags.save_freq) == 0:
                        print('Save the checkpoint')
                        saver.save(sess, os.path.join(flags.output_dir, 'model'), global_step=sv.global_step)
                    step = step+1
                except tf.errors.OutOfRangeError as e:
                    break

            print('Optimization done!!!!!!!!!!!!')


if flags.mode == 'inference':
    with tf.device('/gpu:0'):

        list1_ = os.listdir(flags.train_dir1)
        list1 = [os.path.join(flags.train_dir1, _) for _ in list1_ if _.split('.')[1] == 'jpg']

        list2_ = os.listdir(flags.train_dir2)
        list2 = [os.path.join(flags.train_dir2, _) for _ in list2_ if _.split('.')[1] == 'jpg']

        batch1 = imgs_loder(list1, flags)
        batch2 = imgs_loder(list2, flags)

        net = style_transfer(batch1, batch2, flags)

        tf.summary.scalar('per_loss', net.per_loss)
        tf.summary.scalar('learning_rate', net.learning_rate)

        saver = tf.train.Saver(max_to_keep=10)

        var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        var_list2 = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

        weight_initiallizer = tf.train.Saver(var_list2)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # Use superviser to coordinate all queue and summary writer
        sv = tf.train.Supervisor(logdir=flags.summary_dir, save_summaries_secs=0, saver=None)
        with sv.managed_session(config=config) as sess:
            img_dir = os.path.join(flags.output_dir,'images')
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            count = 0
            while True:
                try:
                    fetches = {
                        "train": net.train,
                        "content_loss": net.content_loss,
                        "adversarial_loss": net.adversarial_loss,
                        "discrim_loss": net.discrim_loss,
                        "global_step": sv.global_step,
                        "learning_rate": net.learning_rate
                    }

                    results = sess.run(fetches)

                    for i in range(results["output"].shape[0]):
                        img_ = results["output"][i].astype(np.uint8)
                        print(img_.shape)
                        img = Image.fromarray(img_)
                        img.save(os.path.join(img_dir,'image%d.jpg'%count))
                        count = count+1

                    print("global_step:%d,loss:%.4f\n" %
                          (results["global_step"], results["per_loss"]))

                except tf.errors.OutOfRangeError as e:

                    break

            print('Evaluation done!!!!!!!!!!!!')