import numpy as np
import os
import tensorflow as tf


def soft_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def create_update_op(online_vars, target_vars):
    copy_ops = [target_var.assign(online_vars[var_name])
                for var_name, target_var in target_vars.items()]
    return tf.group(*copy_ops)


class Loader(object):
    def __init__(self, filename, dataset_num=10):
        self.filename = filename
        self._count = 0
        self.max_count = dataset_num

    def __call__(self, *args, **kwargs):
        data = np.load(self.filename + '_' + str(self._count) + '.npz')
        self.size = data[data.keys()[0]].shape[0]
        data = self.random_shuffle(data)

        self._count = self._count + 1
        if self._count > self.max_count:
            self._count = 0
            raise tf.errors.OutOfRangeError()
        return data['state'], data['action'], data['action_axis'], data['reward'], data['nstate'], self.size

    @staticmethod
    def random_shuffle(npz):
        data_list = [npz[i] for i in npz.keys()]
        size = data_list[0].shape
        s = np.random.permutation(size)
        shuffled_data = [i[s, :] for i in data_list]
        ret = {}
        for i in range(len(npz.keys())):
            ret[npz.keys[i]] = shuffled_data[i]
        return ret
