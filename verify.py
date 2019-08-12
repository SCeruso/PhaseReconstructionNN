import numpy as np
import argparse
import tensorflow as tf

from zernike_generator import ZernikeGenerator
from model import model

DEFAULT_IMG_SIZE = 21
DEFAULT_N_ZERNIKES = 120
DEFAULT_N_TEST = 10

def compute_rse(pred, target, mask):
    targetm = target - np.mean(target[mask != 0])

    diff = pred - target - \
        np.mean(pred[mask != 0]) + np.mean(target[mask != 0])
    diff = diff * diff

    target2 = targetm * targetm
    target2mean = np.mean(target2[mask != 0])

    return 100.0 * np.sqrt(np.mean(diff[mask != 0]) / target2mean)

def verify(args):
    nzernikes = args.nzernikes
    ntests = args.ntests
    chkp_path = args.chkp_path
    size = args.size

    errors = []

    gen = ZernikeGenerator(size, np.arange(nzernikes), seed=None)
    pupil = gen.pupil
    z_t, sx_t, sy_t = gen.data_batch(1)

    reconstucted_t = model(sx_t, sy_t)
    loader = tf.train.Saver()

    sess = tf.Session()
    loader.restore(sess, chkp_path)

    for i in range(ntests):
        reconstructed, z = sess.run([reconstucted_t, z_t])
        
        error = compute_rse(reconstructed.squeeze(), z.squeeze(), pupil)
        errors.append(error)

    errors = np.asarray(errors)

    print("RSE: mean: {}, std: {}".format(errors.mean(), errors.std()))


def main(args):
    verify(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int,
                        default=DEFAULT_IMG_SIZE, help="Image size.")
    parser.add_argument("--nzernikes", type=int, default=DEFAULT_N_ZERNIKES,
                        help="Number of zernike terms to use to validate.")
    parser.add_argument("--ntests", type=int,
                        default=DEFAULT_N_TEST, help="Numbers of tests.")
    parser.add_argument("-chkp_path", type=str, help="Checkpoint to restore.")

    args = parser.parse_args()
    main(args)
