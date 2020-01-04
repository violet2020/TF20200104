import tensorflow as tf
import tensorflow.contrib.slim as slim
from . import resnet_v1, resnet_utils
from tensorflow.contrib.slim import arg_scope
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib.layers.python.layers import regularizers, initializers, layers
from config import cfg
import numpy as np

from . import mobilenet_v2 as mobv

def resnet_arg_scope(bn_is_training,
                     bn_trainable,
                     trainable=True,
                     weight_decay=cfg.weight_decay,
                     weight_init = initializers.variance_scaling_initializer(),
                     batch_norm_decay=0.99,
                     batch_norm_epsilon=1e-9,
                     batch_norm_scale=True):
    batch_norm_params = {
        'is_training': bn_is_training,
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'trainable': bn_trainable,
        'updates_collections': ops.GraphKeys.UPDATE_OPS
    }

    with arg_scope(
            [slim.conv2d, slim.conv2d_transpose],
            weights_regularizer=regularizers.l2_regularizer(weight_decay),
            weights_initializer=weight_init,
            trainable=trainable,
            activation_fn=nn_ops.relu,
            normalizer_fn=layers.batch_norm,
            normalizer_params=batch_norm_params):
        with arg_scope([layers.batch_norm], **batch_norm_params) as arg_sc:
            return arg_sc

def resnet50(inp, bn_is_training, bn_trainable):
    bottleneck = resnet_v1.bottleneck
    blocks = [
        resnet_utils.Block('block1', bottleneck,
                           [(256, 64, 1)] * 2 + [(256, 64, 1)]),
        resnet_utils.Block('block2', bottleneck,
                           [(512, 128, 2)] + [(512, 128, 1)] * 3),
        resnet_utils.Block('block3', bottleneck,
                           [(1024, 256, 2)] + [(1024, 256, 1)] * 5),
        resnet_utils.Block('block4', bottleneck,
                           [(2048, 512, 2)] + [(2048, 512, 1)] * 2)
    ]   
    
    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):

        with tf.variable_scope('resnet_v1_50', 'resnet_v1_50'):
            net = resnet_utils.conv2d_same(
                    tf.concat(inp,axis=3), 64, 7, stride=2, scope='conv1')
            
            net = tf.pad(net, [[0, 0], [1, 1], [1, 1], [0, 0]])
            net = slim.max_pool2d(
                net, [3, 3], stride=2, padding='VALID', scope='pool1')
        net, _ = resnet_v1.resnet_v1(                                  # trainable ?????
            net, blocks[0:1],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')
    
    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net2, _ = resnet_v1.resnet_v1(
            net, blocks[1:2],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')

    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net3, _ = resnet_v1.resnet_v1(
            net2, blocks[2:3],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')

    with slim.arg_scope(resnet_arg_scope(bn_is_training=bn_is_training, bn_trainable=bn_trainable)):
        net4, _ = resnet_v1.resnet_v1(
            net3, blocks[3:4],
            global_pool=False, include_root_block=False,
            scope='resnet_v1_50')

    resnet_features = [net, net2, net3, net4]
    print(resnet_features.shape)
    return resnet_features



####mobilenet_v2

def mobilenet_v2_arg_scope(weight_decay, is_training=True, depth_multiplier=1.0, regularize_depthwise=False,
                           dropout_keep_prob=1.0):

    regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    if regularize_depthwise:
        depthwise_regularizer = regularizer
    else:
        depthwise_regularizer = None

    with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                        activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'center': True, 'scale': True }):

        with slim.arg_scope([slim.conv2d], weights_regularizer=regularizer):

            with slim.arg_scope([slim.separable_conv2d],
                                weights_regularizer=depthwise_regularizer, depth_multiplier=depth_multiplier):

                with slim.arg_scope([slim.dropout], is_training=is_training, keep_prob=dropout_keep_prob) as sc:

                    return sc

def mobilenet_v2(inp, conv_width=1.4):
  with tf.contrib.slim.arg_scope(mobv.training_scope()):
    net, endpoints = mobv.mobilenet_base(inp, conv_width)
    return net

'''
def block(net, input_filters, output_filters, expansion, stride):
    res_block = net
    res_block = slim.conv2d(inputs=res_block, num_outputs=input_filters * expansion, kernel_size=[1, 1])
    res_block = slim.separable_conv2d(inputs=res_block, num_outputs=None, kernel_size=[3, 3], stride=stride)
    res_block = slim.conv2d(inputs=res_block, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
    if stride == 2:
        return res_block
    else:
        if input_filters != output_filters:
            net = slim.conv2d(inputs=net, num_outputs=output_filters, kernel_size=[1, 1], activation_fn=None)
        return tf.add(res_block, net)


def blocks(net, expansion, output_filters, repeat, stride):
    input_filters = net.shape[3].value

    # first layer should take stride into account
    net = block(net, input_filters, output_filters, expansion, stride)

    for _ in range(1, repeat):
        net = block(net, input_filters, output_filters, expansion, 1)

    return net


def mobilenet_v2(inputs,
                 num_classes=1000,
                 dropout_keep_prob=0.999,
                 is_training=True,
                 depth_multiplier=1.0,
                 prediction_fn=tf.contrib.layers.softmax,
                 spatial_squeeze=True,
                 scope='MobilenetV2'):

    endpoints = dict()

    expansion = 6

    with tf.variable_scope(scope):

        with slim.arg_scope(mobilenet_v2_arg_scope(0.0004, is_training=is_training, depth_multiplier=depth_multiplier,
                                                   dropout_keep_prob=dropout_keep_prob)):
            net = tf.identity(inputs)

            net = slim.conv2d(net, 32, [3, 3], scope='conv11', stride=2)

            net1 = blocks(net=net, expansion=1, output_filters=16, repeat=1, stride=1)

            net2 = blocks(net=net1, expansion=expansion, output_filters=24, repeat=2, stride=2)

            net3 = blocks(net=net2, expansion=expansion, output_filters=32, repeat=3, stride=2)

            net4 = blocks(net=net3, expansion=expansion, output_filters=64, repeat=4, stride=2)

            net5 = blocks(net=net4, expansion=expansion, output_filters=96, repeat=3, stride=1)

            net6 = blocks(net=net5, expansion=expansion, output_filters=160, repeat=3, stride=2)

            net7 = blocks(net=net6, expansion=expansion, output_filters=320, repeat=1, stride=1)
            mobilenetv2_features = [net1, net2, net3, net4, net5, net6, net7]
    return mobilenetv2_features

            #net = slim.conv2d(net, 1280, [1, 1], scope='last_bottleneck')

            #net = slim.avg_pool2d(net, [7, 7])

            #logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='features')

            #if spatial_squeeze:
                #logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')

            #endpoints['Logits'] = logits

            #if prediction_fn:
                 #endpoints['Predictions'] = prediction_fn(logits, scope='Predictions')

    #return logits, endpoints
    #return logits

mobilenet_v2.default_image_size = 224


'''
