import tensorflow as tf
import random
import numpy as np
import math


class ZernikeGenerator(object):
    def __init__(self, size, nzernikes=np.arange(120), seed=None):
        self.seed = seed
        self.zernikes = []
        self.nzernikes = nzernikes
        self.size = size
        self.xx, self.yy = tf.meshgrid(np.arange(size), np.arange(size))
        self.xx = tf.cast(self.xx, tf.float32) - (size / 2.0)
        self.yy = tf.cast(self.yy, tf.float32) - (size / 2.0)
        self.pupil = self._circular_mask(self.size)

        indexes = [self._compute_indexes(i+1) for i in self.nzernikes]
        ro = tf.sqrt(self.xx*self.xx + self.yy*self.yy) / ((size / 2.0) - 1.0)
        theta = tf.atan2(self.yy, self.xx)

        for j, (n, m) in enumerate(indexes):
            j = j+1
            if m == 0:
                mode = tf.sqrt(n + 1.0) * self._zer_rad(ro, n, m)
            if m != 0:
                if self._is_even(j):
                    mode = tf.sqrt((2.0*n)+2.0) * \
                        self._zer_rad(ro, n, m) * tf.cos(m * theta)
                else:
                    mode = tf.sqrt((2.0*n)+2.0) * \
                        self._zer_rad(ro, n, m) * tf.sin(m * theta)
            self.zernikes.append(mode * self.pupil)

        self.sxs = []
        self.sys = []

        for idx, zer in enumerate(self.zernikes):
            dx, dy = tf.gradients(zer, [self.xx, self.yy])
            self.sxs.append(dx)
            self.sys.append(dy)

        self.sxs = tf.transpose(
            tf.convert_to_tensor(self.sxs), [1, 2, 0])
        self.sys = tf.transpose(
            tf.convert_to_tensor(self.sys), [1, 2, 0])

        self.zernikes = tf.transpose(
            tf.convert_to_tensor(self.zernikes), [1, 2, 0])

        self.zernikes = tf.expand_dims(self.zernikes, 0)
        self.sxs = tf.expand_dims(self.sxs, 0)
        self.sys = tf.expand_dims(self.sys, 0)

    def data_batch(self, batch_size=16):
        weights = tf.random_uniform(
            [batch_size, len(self.nzernikes)], -1.0, maxval=1.0, dtype=tf.float32, seed=self.seed)
        weights = tf.expand_dims(weights, 1)
        weights = tf.expand_dims(weights, 1)
        phase = tf.reduce_sum(self.zernikes * weights, axis=(-1))
        sx = tf.reduce_sum(self.sxs * weights, axis=(-1))
        sy = tf.reduce_sum(self.sys * weights, axis=(-1))
        phase = phase * self.pupil[np.newaxis, :, :]
        return phase, sx, sy

    def _zer_rad(self, ro, n, m):
        rr = np.zeros(np.shape(ro))
        ddif = int(np.round((n-m)/2))
        dsum = int(round((n+m)/2))
        for s in range(ddif+1):
            numer = ((-1.0)**s)*math.factorial(n-s)
            denom = math.factorial(
                s)*math.factorial(dsum-s)*math.factorial(ddif-s)
            rr = rr+((ro**(n-(2*s))*numer)/denom)
        return rr

    def _circular_mask(self, resolucion):
        xp = np.linspace(-1, 1, resolucion)
        X, Y = np.meshgrid(xp, xp)
        rho = np.sqrt(X**2 + Y**2).astype('float64')
        pupil = np.ones(np.shape(rho))

        pupil[rho > 1] = 0
        return pupil

    def _is_even(self, number):
        return (number % 2) == 0

    def _compute_indexes(self, jend):
        n_gra = np.zeros(jend+100)
        m_azi = np.zeros(jend + 100)
        n = 1
        j = 0
        while j <= jend:
            for mm in range(n+1):
                j = int(((n*(n+1))/2) + mm + 1)
                if self._is_even(n) != self._is_even(mm):
                    m = mm+1
                else:
                    m = mm

                n_gra[j-1] = n
                m_azi[j-1] = m
            n = n+1

        ngra = n_gra[jend-1]
        mazi = m_azi[jend-1]
        return int(ngra), int(mazi)
