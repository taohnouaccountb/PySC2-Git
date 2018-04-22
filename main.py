import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util import create_update_op, Loader
from model import dqn

# ENVIRONMENT ARGUMENT
RUN_ON_CRANE = True

flags = tf.app.flags
if RUN_ON_CRANE:
    flags.DEFINE_string('data_dir', '/work/cse496dl/shared/hackathon/08/ptbdata/', '')
    flags.DEFINE_string('save_dir', '/work/cse496dl/tyao/output/', '')
else:
    flags.DEFINE_string('GPU_ID', '1', '')
    flags.DEFINE_string('data_dir', 'E:\\cse496_hw2\\PTB-Git\\data\\', '')
    flags.DEFINE_string('save_dir', '.\\output\\', 'directory where model graph and weights are saved')

flags.DEFINE_bool('RUN_ON_CRANE', RUN_ON_CRANE, '')
flags.DEFINE_integer('batch_size', 20, '')
flags.DEFINE_integer('max_epoch_num', 20, '')
flags.DEFINE_integer('patience', 4, '')
flags.DEFINE_float('REG_COEFF', 0.00001, '')
flags.DEFINE_float('LEARNING_RATE', 1e-3, '')

flags.DEFINE_float('GAMMA', 0.9, '')
FLAGS = flags.FLAGS

early_count_list = []


def main(argv):
    # Load data
    loader = Loader("ep_test")

    # Define Variables and Layers
    train_flag = tf.Variable(True)
    x = tf.placeholder(tf.int32, [None, 17, 84, 84], name='input_state')
    y = tf.placeholder(tf.int32, [None], name='input_action')
    yy = tf.placeholder(tf.int32, [None, 2], name='input_action_axis')
    z = tf.placeholder(tf.int32, [None], name='input_reward')
    xx = tf.placeholder(tf.int32, [None, 17, 84, 84], name='input_new_state')

    # online_op: select, select_0, select_1, select_2
    # online_att: attack operation, 20*20
    # move_op: 3*3
    online_op, online_vars = dqn(x, name='online', training=train_flag)
    target_op, target_vars = dqn(xx, name='target', training=train_flag)
    copy_online_to_target = create_update_op(online_vars, target_vars)

    op = online_op[:, y]
    label_op = z + FLAGS.GAMMA * tf.reduce_max(target_op, axis=1)

    # Set up training and saving functionality
    global_step_tensor = tf.get_variable('global_step', trainable=False, shape=[], initializer=tf.zeros_initializer)
    optimizer = tf.train.RMSPropOptimizer(FLAGS.LEARNING_RATE)
    train_op = optimizer.minimize(loss, global_step=global_step_tensor)

    # Start Training
    config = tf.ConfigProto()
    if not RUN_ON_CRANE:
        config.gpu_options.visible_device_list = FLAGS.GPU_ID
    with tf.Session(config=config) as session:
        # print the sum of the numbers of the parameters in the graph
        print('Number of parameters in the graph: ',
              np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
        session.run(tf.global_variables_initializer())
        batch_size = FLAGS.batch_size

        for epoch in range(FLAGS.max_epoch_num):
            print('\nEpoch: ' + str(epoch))
            update_count = 0
            # train
            for t in tqdm(range(loader.max_count), ncols=70):
                train_state, train_action, train_action_axis, train_reward, train_nstate, num_train = loader()
                train_flag.assign(True)
                ce_vals = []
                for i in range(num_train // batch_size):
                    batch_x = train_state[i * batch_size:(i + 1) * batch_size, :]
                    batch_y = train_action[i * batch_size:(i + 1) * batch_size, :]
                    batch_yy = train_action_axis[i * batch_size:(i + 1) * batch_size, :]
                    batch_z = train_reward[i * batch_size:(i + 1) * batch_size, :]
                    batch_xx = train_nstate[i * batch_size:(i + 1) * batch_size, :]
                    _, train_ce = session.run([train_op, loss],
                                              {x: batch_x, y: batch_y, yy: batch_yy, z: batch_z, xx: batch_xx})
                    ce_vals.append(train_ce)

                    # target network
                    update_count = update_count + 1
                    if update_count % 500 == 0:
                        session.run(copy_online_to_target)
                avg_train_ce = sum(ce_vals) / len(ce_vals)  # after each fold , train err store in kfd_train_err


if __name__ == "__main__":
    tf.app.run()
