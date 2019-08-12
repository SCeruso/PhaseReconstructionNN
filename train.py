import numpy as np
import argparse
import tensorflow as tf
import os

from zernike_generator import ZernikeGenerator
from model import model

DEFAULT_IMG_SIZE = 21
DEFAULT_N_ZERNIKES = 120
DEFAULT_ITS = 500000
DEFAULT_OUT_PATH = "logdir/"

def train(args):
    size = args.size
    nzernikes = args.nzernikes
    out_path = args.chkp_out
    train_its = args.its

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    gen = ZernikeGenerator(size, np.arange(nzernikes), seed=None)
    pupil = gen.pupil[np.newaxis,:,:,np.newaxis]
    
    z_t, sx_t, sy_t = gen.data_batch()
    z_t = tf.expand_dims(z_t, -1)

    res = model(sx_t, sy_t)
    
    loss_t = tf.losses.mean_squared_error(z_t * pupil, res * pupil)

    global_step_t = tf.train.get_or_create_global_step()

    with tf.name_scope('optimizer'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) 
        updates = tf.group(*update_ops, name='update_ops') 
        opt = tf.train.AdamOptimizer(0.0001)
        with tf.control_dependencies([updates]): 
            train_op = opt.minimize(loss_t, name='optimizer',
                                global_step=global_step_t)

    saver = tf.train.Saver()
    
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        while True:
            try:
                _, step, loss = sess.run([train_op, global_step_t, loss_t])

                if step % 100 == 0:
                    loss = sess.run(loss_t)
                    print("Loss at step {}: {}".format(step, loss))
            except KeyboardInterrupt:
                break
            if step > train_its:
                break
        saver.save(sess, os.path.join(
                        out_path, "{}".format(size)), global_step=global_step_t)
    print("Checkpoint saved at: {}".format(out_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int,
                        default=DEFAULT_IMG_SIZE, help="Image size.")
    parser.add_argument("--its", type=int,
                        default=DEFAULT_ITS, help="Image size.")
    parser.add_argument("--nzernikes", type=int, default=DEFAULT_N_ZERNIKES,
                        help="Number of zernike terms to use to train.")
    parser.add_argument("--chkp_out", type=str,
                        default=DEFAULT_OUT_PATH, help="Path to save checkpoint.")

    args = parser.parse_args()
    train(args)
