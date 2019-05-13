import tensorflow as tf
from math import ceil
import math
import numpy as np

def _variable_on_gpu(name, shape, initializer):
    """
    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable
    Returns:
      Variable Tensor
    """
    with tf.device('/gpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def msra_initializer(kl, dl):
    """
    kl for kernel size, dl for filter number
    """
    stddev = math.sqrt(2. / (kl ** 2 * dl))
    return tf.truncated_normal_initializer(stddev=stddev)

def conv_with_bn(inputT, shape, train_phase, activation=True, name=None):
    Cin = inputT.shape[3]
    Cout = shape[2]
    k1 = shape[0]
    k2 = shape[1]
    with tf.variable_scope(name) as scope:
        kernel = _variable_on_gpu('ort_weights', shape=(k1,k2,Cin,Cout), initializer=tf.initializers.orthogonal())
        conv = tf.nn.conv2d(inputT, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_gpu('biases', [Cout], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def separable_conv_with_bn(inputT, shape, train_phase, activation=True, name=None):
    with tf.variable_scope(name) as scope:
        depth_filter = _variable_on_gpu('depth_weights', shape=shape[:-1], initializer=tf.initializers.orthogonal())
        point_filter = _variable_on_gpu('point_weights', shape=[1,1,shape[2]*shape[3],shape[4]], initializer=tf.initializers.orthogonal())
        conv = tf.nn.separable_conv2d(inputT,depthwise_filter=depth_filter,pointwise_filter=point_filter,strides=[1,1,1,1],padding='SAME')
        biases = _variable_on_gpu('biases', [shape[4]], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def atrous_conv2d(inputT,shape,train_phase,rate=2,activation=True,name=None):
    Cin = inputT.shape[3]
    Cout = shape[2]
    k1 = shape[0]
    k2 = shape[1]
    with tf.variable_scope(name) as scope:
        kernel = _variable_on_gpu('ort_weights', shape=(k1,k2,Cin,Cout), initializer=tf.initializers.orthogonal())
        conv = tf.nn.atrous_conv2d(inputT, kernel,rate=rate,padding='SAME')
        biases = _variable_on_gpu('biases', [Cout], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        if activation is True:
            conv_out = tf.nn.relu(batch_norm_layer(bias, train_phase, scope.name))
        else:
            conv_out = batch_norm_layer(bias, train_phase, scope.name)
    return conv_out

def get_deconv_filter(f_shape):
    """
      reference: https://github.com/MarvinTeichmann/tensorflow-fcn
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init,
                           shape=weights.shape)


def deconv_layer(inputT, f_shape, output_shape, stride=2, name=None):
    # output_shape = [b, w, h, c]
    # sess_temp = tf.InteractiveSession()
    strides = [1, stride, stride, 1]
    with tf.variable_scope(name):
        weights = get_deconv_filter(f_shape)
        deconv = tf.nn.conv2d_transpose(inputT, weights, output_shape,strides=strides, padding='SAME')
    return deconv

def batch_norm_layer(inputT, is_training, scope):
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                                                        center=False, updates_collections=None, scope=scope + "_bn"),
                   lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                                                        updates_collections=None, center=False, scope=scope + "_bn",
                                                        reuse=True))

# def conv_concate(x, concate_layer,conv_size, growth_rate,phase_train, name):
#     if not concate_layer:
#         l = conv_with_bn(x, [conv_size[0], conv_size[1], growth_rate], phase_train, name=name)
#     else:
#         l = tf.concat(concate_layer, axis=3)
#         l = tf.concat([x, l], axis=3)
#         l = conv_with_bn(l, [conv_size[0], conv_size[1], growth_rate], phase_train, name=name)
#     return l

def conv_concate(x,conv_size, growth_rate,phase_train, name):
    l=conv_with_bn(x, [conv_size[0], conv_size[1], growth_rate], phase_train, name=name)
    l=tf.concat([l,x],3)
    return l

def dense_block(l,layers=4,growth_rate=12,conv_size=(3,3),phase_train=True):
    for i in range(layers):
        l = conv_concate(l,conv_size=conv_size, growth_rate=growth_rate,phase_train=phase_train, name='conv_{}.'.format(i))
    return l

# def dense_block(l,layers=4,growth_rate=12,conv_size=(3,3),phase_train=True):
#     a = []
#     a.append(l)
#     for i in range(layers):
#         l = conv_concate(l, concate_layer=a[:i],conv_size=conv_size, growth_rate=growth_rate,phase_train=phase_train, name='conv_{}.'.format(i))
#         a.append(l)
#     l = tf.concat(a, axis=3)
#     return l

def crack_refine(l,conv_size=(7,7),phase_train=True):
    """ a crack refinement module using two depth-wise separable convolution k*1 followed by 1*k """
    # cr = conv_with_bn(l,[1,7,l.get_shape().as_list()[3],64],phase_train,False,name='CR_0')
    # cr = conv_with_bn(cr,[7,1,cr.get_shape().as_list()[3],64],phase_train,False,name='CR_1')
    cr = separable_conv_with_bn(l,[conv_size[0],1,l.get_shape().as_list()[3],2,32],phase_train,False,name='CR_0')
    cr = separable_conv_with_bn(cr,[1,conv_size[1],cr.get_shape().as_list()[3],2,32],phase_train,True,name='CR_1')
    L = tf.nn.dropout(cr,rate=0.4)  # dropout rate 0.4
    return L

def crack_refine_new(l,conv_size=(3,3,7,7,11,11),phase_train=True):
    branch_1 = separable_conv_with_bn(l,[conv_size[0],1,l.get_shape().as_list()[3],2,32],phase_train,False,name='CR_00')
    branch_1 = separable_conv_with_bn(branch_1,[1,conv_size[1],branch_1.get_shape().as_list()[3],2,32],phase_train,True,name='CR_01')

    branch_2 = separable_conv_with_bn(l,[conv_size[2],1,l.get_shape().as_list()[3],2,32],phase_train,False,name='CR_10')
    branch_2 = separable_conv_with_bn(branch_2,[1,conv_size[3],branch_2.get_shape().as_list()[3],2,32],phase_train,True,name='CR_11')

    branch_3 = separable_conv_with_bn(l,[conv_size[4],1,l.get_shape().as_list()[3],2,32],phase_train,False,name='CR_20')
    branch_3 = separable_conv_with_bn(branch_3, [1, conv_size[5], branch_2.get_shape().as_list()[3], 2, 32],
                                      phase_train, True, name='CR_11')
    max = tf.math.maximum(branch_1,branch_2)
    L = tf.math.maximum(max,branch_3)
    return L

def atrous_SPP(l,phase_train=True):
    L1 = conv_with_bn(l,[3,3,64],phase_train,name='conv_1x1')
    L2 = atrous_conv2d(l,[3,3,64], rate=3,train_phase=phase_train, name='atrous_6')
    L3 = atrous_conv2d(l, [3,3,64], rate=6,train_phase=phase_train, name='atrous_12')
    L4 = atrous_conv2d(l, [3,3,64], rate=9,train_phase=phase_train, name='atrous_18')
    L = tf.concat([L1, L2, L3, L4], axis=3)
    return L


def max_pool(inputs,name):
    return tf.nn.max_pool(inputs, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)