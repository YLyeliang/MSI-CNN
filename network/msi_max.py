import tensorflow as tf
import numpy as np
from core.utilities import conv_with_bn, dense_block, \
    atrous_SPP, deconv_layer, crack_refine, max_pool, msra_initializer, _variable_on_gpu


class msi_cnn(object):
    """Multi-scale input CNN framework for feature extraction"""

    def __init__(self, inputs, phase_train):
        self.outputs = self.forward(inputs, phase_train)

    def forward(self, inputs, phase_train):
        batch_size = tf.shape(inputs)[0]
        norm1 = inputs / 255.
        # norm1 = batch_norm_layer(inputs, phase_train, scope='norm1')
        norm2 = tf.image.resize_bilinear(norm1, (128, 128))
        norm3 = tf.image.resize_bilinear(norm1, (64, 64))
        norm4 = tf.image.resize_bilinear(norm1, (32, 32))
        # input_dense_block_1
        with tf.variable_scope('input_block_1'):
            self.conv1 = dense_block(norm1, layers=4, growth_rate=24, phase_train=phase_train)  # 256x256
            self.conv1 = conv_with_bn(self.conv1, [1, 1, 64], phase_train, True,
                                      name='transition')  # trainsition
            # pool1
            self.pool1 = max_pool(self.conv1, 'pool1')  # 128x128
        # input_dense_block_2
        with tf.variable_scope('input_block_2'):
            self.conv2 = dense_block(norm2, layers=4, growth_rate=24, phase_train=phase_train)  # 128x128
            self.conv2 = conv_with_bn(self.conv2, [1, 1, 64], phase_train, True,
                                      name='transition')  # trainsition

        # dense_block_1
        with tf.variable_scope('dense_block_1'):
            self.dense1 = tf.concat([self.pool1, self.conv2], axis=3)
            self.dense1 = dense_block(self.dense1, layers=4, growth_rate=24, phase_train=phase_train)
            self.dense1 = conv_with_bn(self.dense1, [1, 1, 64], phase_train, True,
                                       name='transition')
            # pool2
            self.pool2 = max_pool(self.dense1, 'pool2')  # 64x64
        # input_dense_block_3
        with tf.variable_scope('input_block_3'):
            self.conv3 = dense_block(norm3, layers=4, growth_rate=24, conv_size=(3, 3),
                                     phase_train=phase_train)  # 64x64
            self.conv3 = conv_with_bn(self.conv3, [1, 1, 64], phase_train, True,
                                      name='transition')  # trainsition
        # dense_block_2
        with tf.variable_scope('dense_block_2'):
            self.dense2 = tf.concat([self.pool2, self.conv3], axis=3)
            self.dense2 = dense_block(self.dense2, layers=4, growth_rate=24, phase_train=phase_train)
            self.dense2 = conv_with_bn(self.dense2, [1, 1, 64], phase_train, True,
                                       name='transition')
            # pool3
            self.pool3 = max_pool(self.dense2, 'pool3')
        # input_dense_block_4
        with tf.variable_scope('input_block_4'):
            self.conv4 = dense_block(norm4, layers=4, growth_rate=24, conv_size=(3, 3),
                                     phase_train=phase_train)  # 32x32
            self.conv4 = conv_with_bn(self.conv4, [1, 1, 64], phase_train, True,
                                      name='transition')  # trainsition
        # dense_block_3
        with tf.variable_scope('dense_block_3'):
            self.dense3 = tf.concat([self.pool3, self.conv4], axis=3)
            self.dense3 = dense_block(self.dense3, layers=4, growth_rate=24, phase_train=phase_train)
            self.dense3 = conv_with_bn(self.dense3, [1, 1, 64], phase_train, True,
                                       name='transition')
            # pool4
            self.pool4 = max_pool(self.dense3, 'pool4')  # 16x16
        with tf.variable_scope('atrous_SPP'):
            self.pool4 = atrous_SPP(self.pool4, phase_train)
            self.pool4 = conv_with_bn(self.pool4, [1, 1, 256], phase_train, True, name='trainsition')

        """ End of encoder """
        """ start upsample """
        # upsample4
        # Need to change when using different dataset out_w, out_h
        self.upsample4 = deconv_layer(self.pool4, [3, 3, 256, 256], [batch_size, 32, 32, 256], 2, "up4")
        # decode 4
        with tf.variable_scope('decode_4'):
            self.cr4 = crack_refine(self.dense3, (3, 3), phase_train=phase_train)
            self.concat4 = tf.concat([self.upsample4, self.cr4], axis=3)
            self.conv_decode4 = conv_with_bn(self.concat4, [3, 3, 128], phase_train,
                                             False, name="conv_decode4")

        # upsample 3
        self.upsample3 = deconv_layer(self.conv_decode4, [3, 3, 128, 128], [batch_size, 64, 64, 128], 2, "up3")
        # decode 3
        with tf.variable_scope('decode_3'):
            self.cr3 = crack_refine(self.dense2, (5, 5), phase_train=phase_train)
            self.concat3 = tf.concat([self.upsample3, self.cr3], axis=3)
            self.conv_decode3 = conv_with_bn(self.concat3, [3, 3, 64], phase_train, False,
                                             name="conv_decode3")

        # upsample2
        self.upsample2 = deconv_layer(self.conv_decode3, [3, 3, 64, 64], [batch_size, 128, 128, 64], 2, "up2")
        # decode 2
        with tf.variable_scope('decode_2'):
            self.cr2 = crack_refine(self.dense1, (7, 7), phase_train=phase_train)
            self.concat2 = tf.concat([self.upsample2, self.cr2], axis=3)
            self.conv_decode2 = conv_with_bn(self.concat2, [3, 3, 64], phase_train, False,
                                             name="conv_decode2")

        # upsample1
        self.upsample1 = deconv_layer(self.conv_decode2, [3, 3, 64, 64], [batch_size, 256, 256, 64], 2, "up1")
        # decode4
        with tf.variable_scope('decode_1'):
            self.cr1 = crack_refine(self.conv1, (9, 9), phase_train=phase_train)
            self.concat1 = tf.concat([self.upsample1, self.cr1], axis=3)
            self.conv_decode1 = conv_with_bn(self.concat1, [3, 3, 64], phase_train, False,
                                             name="conv_decode1")
        """ end of Decode """

        return self.conv_decode1


class msi_net(object):
    def __init__(self, num_classes):
        self.NUM_CLASSES = num_classes

    def inference(self, inputs, phase_train):
        conv_decode1 = msi_cnn(inputs, phase_train).outputs
        with tf.variable_scope('conv_classifier') as scope:
            kernel = _variable_on_gpu('weights', shape=[1, 1, 64, self.NUM_CLASSES],
                                      initializer=msra_initializer(1, 64))
            conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = _variable_on_gpu('biases', [self.NUM_CLASSES], tf.constant_initializer(0.0))
            logit = tf.nn.bias_add(conv, biases, name=scope.name)
        return logit

    def weighted_loss(self, logits, labels, num_classes, head=None):
        """ median-frequency re-weighting """
        with tf.name_scope('loss'):
            logits = tf.reshape(logits, (-1, num_classes))

            epsilon = tf.constant(value=1e-10)

            logits = logits + epsilon

            # consturct one-hot label array
            label_flat = tf.reshape(labels, (-1, 1))

            # should be [batch ,num_classes]
            labels = tf.reshape(tf.one_hot(label_flat, depth=num_classes), (-1, num_classes))

            softmax = tf.nn.softmax(logits)

            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), head), axis=[1])

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

            tf.add_to_collection('losses', cross_entropy_mean)

            loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

        return loss

    def loss_layer(self, logits, labels):
        loss_weight = np.array([
            0.2595,
            13.3899])  # class 0~11

        labels = tf.cast(labels, tf.int32)
        # return loss(logits, labels)
        return self.weighted_loss(logits, labels, num_classes=self.NUM_CLASSES, head=loss_weight)


x = tf.placeholder(tf.float32, shape=[2, 256, 256, 3])
phase_train = tf.placeholder(tf.bool, name='phase_train')
model = msi_cnn(x, phase_train).outputs
