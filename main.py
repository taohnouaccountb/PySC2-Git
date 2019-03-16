import numpy as np
import tensorflow as tf
from tqdm import tqdm
from util import create_update_op, Loader
from model_update import dqn

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

# FLAGS = flags.FLAGS

early_count_list = []


def main(argv):
    # parameters

    n_steps = 1000000  # total number of training steps
    n_outputs = 50       # the number of actions
    save_steps = 100000  # save the model every 1,000 training steps
    copy_steps = 10000  # copy online DQN to target DQN every 10,000 training steps
    discount_rate = 0.99
    batch_size = 50
    checkpoint_path = "./my_dqn.ckpt"

    replay_memory_size = 10000000                       #    这个你改一下
    learning_rate = 0.001
    momentum = 0.95
    epsilon = 0.1
    loss_val = np.infty

#################################



    # Load data
    loader = Loader("ep_test")

    # Define Variables and Layers
    train_flag = tf.Variable(True)

    x = tf.placeholder(tf.int32, [None, 17, 84, 84], name='input_state')    #  84, 84, 17  ???
    y = tf.placeholder(tf.int32, [None], name='input_action')
    z = tf.placeholder(tf.int32, [None], name='input_reward')
    xx = tf.placeholder(tf.int32, [None, 17, 84, 84], name='input_new_state')
    c=tf.placeholder(tf.int32, [None], name='index_endgame')
########################

    online_q_values, online_vars = dqn(x, name="q_networks/online", training=train_flag)
    target_q_values, target_vars = dqn(xx, name="q_networks/target", training=train_flag)

    copy_ops = [target_var.assign(online_vars[var_name])
                for var_name, target_var in target_vars.items()]
    copy_online_to_target = tf.group(*copy_ops)

##

    # Set up training and saving functionality

    with tf.variable_scope("train"):

        q_value = tf.reduce_sum(online_q_values * tf.one_hot(y, n_outputs),
                                axis=1, keep_dims=True)
        error = tf.abs(target_q_values - q_value)

        clipped_error = tf.clip_by_value(error, 0.0, 1.0)
        linear_error = 2 * (error - clipped_error)
        loss = tf.reduce_mean(tf.square(clipped_error) + linear_error)

        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True)
        training_op = optimizer.minimize(loss, global_step=global_step)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


#####################################################################
    def sample_memories(batch_size):

        #  replay_memory format:  deque([], maxlen=replay_memory_size)

        indices = np.random.permutation(len(replay_memory))[:batch_size]
        cols = [[], [], [], [], []]  # state, action, reward, next_state, continue
        for idx in indices:
            memory = replay_memory[idx]
            for col, value in zip(cols, memory):
                col.append(value)
        cols = [np.array(col) for col in cols]
        return cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)

    def epsilon_greedy(q_values):
        if np.random.rand() < epsilon:
            return np.random.randint(n_outputs)  # random action
        else:
            return np.argmax(q_values)  # optimal action
#######################################################################

    with tf.Session() as sess:
        if os.path.isfile(checkpoint_path + ".index"):
            saver.restore(sess, checkpoint_path)
        else:
            init.run()
            copy_online_to_target.run()

        train_state, train_action, train_reward, train_nstate, num_train = loader()

        while True:
            step = global_step.eval()
            if step >= n_steps:
                break

            print("\rTraining step {}/{} ({:.1f})%\tLoss {:5f}".format(step, n_steps, step * 100 / n_steps, loss_val))


            # Sample memories and use the target DQN to produce the target Q-Value
            X_state_val, X_action_val, rewards, X_next_state_val, continues = (
                sample_memories(batch_size))


            next_q_values = target_q_values.eval(
                feed_dict={X_state: X_next_state_val})

            max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
            y_val = rewards + continues * discount_rate * max_next_q_values

            # Train the online DQN
            _, loss_val = sess.run([training_op, loss], feed_dict={
                X_state: X_state_val, X_action: X_action_val, y: y_val})

            # Regularly copy the online DQN to the target DQN
            if step % copy_steps == 0:
                copy_online_to_target.run()

            # And save regularly
            if step % save_steps == 0:
                saver.save(sess, checkpoint_path)


if __name__ == "__main__":
    tf.app.run()
