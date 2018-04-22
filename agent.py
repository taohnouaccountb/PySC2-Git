import os
import pprint
import tensorflow as tf
from util import soft_mkdir

pp = pprint.PrettyPrinter().pprint
FLAGS = tf.flags.FLAGS


class BaseModel(object):
    """Abstract object representing an Reader model."""

    def __init__(self, gpu_id):
        config = tf.ConfigProto()
        if not FLAGS.RUN_ON_CRANE:
            config.gpu_options.visible_device_list = gpu_id
        self._sess = tf.Session(config=config)
        self._saver = tf.train.Saver()

    def save_model(self, step=None):
        print(" [*] Saving checkpoints...")
        soft_mkdir(self.checkpoint_dir)
        fname = os.path.join(self.checkpoint_dir, 'sc2')
        self._saver.save(self._sess, self._fname, global_step=step)

    def load_model(self):
        print(" [*] Loading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(self._sess, fname)
            print(" [*] Load SUCCESS: %s" % fname)
            return True
        else:
            print(" [!] Load FAILED: %s" % self.checkpoint_dir)
            return False

    @property
    def checkpoint_dir(self):
        return 'checkpoints'

    @property
    def saver(self):
        if self._saver is None:
            self._saver = tf.train.Saver(max_to_keep=1)
        return self._saver
