import tensorflow as tf


def dqn(state_input, name, training=None):
    with tf.variable_scope(name) as scope:
        op_output = tf.layers.dense(state_input, 3)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=name)
    trainable_vars_by_name = {var.name[len(name):]: var for var in trainable_vars}
    return op_output, trainable_vars_by_name
