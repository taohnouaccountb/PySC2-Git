import tensorflow as tf


def dqn(state_input, name, training=None):

    with tf.variable_scope(name) as scope:

        conv_1 = tf.layers.conv2d(state_input, 32, 8, strides=4, padding='same', activation=tf.nn.relu, name='conv_1')

        conv_2 = tf.layers.conv2d(conv_1, 64, 4, strides=2, padding='same', activation=tf.nn.relu, name='conv_2')

        conv_3 = tf.layers.conv2d(conv_2, 64, 3, strides=1, padding='same', activation=tf.nn.relu, name='conv_3')

        hidden1 = tf.layers.dense(conv_3, 512, activation=tf.nn.relu, name='hidden1')

        op_output = tf.layers.dense(hidden1, 30, activation=None, name='hidden2')  # 30 is the number of actions

    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    trainable_vars_by_name = {var.name[len(name):]: var for var in trainable_vars}

    return op_output, trainable_vars_by_name