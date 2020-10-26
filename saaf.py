
from config import config
cfg = config()

from tensorflow.keras.layers import Layer
from tensorflow import matmul
from tensorflow.math import greater_equal, greater
import tensorflow as tf

import numpy as np
import math
#=============================================
# WARSTWA SAAF
#===========================================
def p_fcn(c,x):
    return tf.math.pow(x, c) / math.factorial(c)

def P_fcn(eta,x):
    arrays = [p_fcn(c,x) for c in range(eta)]
    return tf.stack(arrays, axis=0)

def suma_p_fcn(eta,x1,x2):
    result = tf.zeros(tf.shape(x1), dtype=tf.dtypes.float32)
    for c in range(eta):
        result += p_fcn(c,x1)*p_fcn(eta-c,x2)
    return result

def I_fcn(a1,a2,x):
    return (greater_equal(x,a1) * greater(a2,x)) * tf.ones(tf.shape(x))

def b_fcn(eta,l,a,x):
    al1 = tf.ones(tf.shape(x)) * a[l+1]
    al = tf.ones(tf.shape(x)) * a[l]
    greaterAll = tf.cast(greater(x,al1), tf.float32)
    lowerAl = tf.cast(greater(al,x), tf.float32)
    w_przedziale = tf.cast(greater_equal(al1,x), tf.float32) * tf.cast(greater_equal(x,al), tf.float32)
    #zeroTF = tf.zeros(tf.shape(x), dtype=tf.dtypes.float64)
    if 0 < a[l] and 0 < a[l+1]:
        return suma_p_fcn(eta, a[l] - a[l+1], x - a[l]) * lowerAl + p_fcn(eta, x - a[l+1]) * w_przedziale
    if 0 > a[l] and 0 > a[l+1]:
        return suma_p_fcn(eta, a[l+1] - a[l], x - a[l+1]) * greaterAll + p_fcn(eta, x - a[l]) * w_przedziale
    return suma_p_fcn(eta, a[l+1], x - a[l+1]) * greaterAll + p_fcn(eta, x) * w_przedziale + suma_p_fcn(eta, a[l], x - a[l]) * lowerAl

def B_fcn(ksi,eta,a,x):
    arrays = [b_fcn(eta,l,a,x) for l in range(ksi)]
    return tf.stack(arrays, axis=0)


class DenseSAAF(Layer):

  def __init__(self, units=cfg.m, ksi=cfg.ksi, eta=cfg.eta, **kwargs):
    super(DenseSAAF, self).__init__(**kwargs)
    self.units = units
    self.ksi = ksi
    self.eta = eta
    self.a = 2.2 * np.arange(ksi+1, dtype=np.float32)/ksi - 1.1
    #print(self.a)

  def get_config(self):
    config = super().get_config().copy()
    config.update({
      'units': self.units,
      'ksi': self.ksi,
      'eta': self.eta,
    })
    return config

  def build(self, input_shape):
    self.w = self.add_variable("w", shape=[int(input_shape[-1]),self.units])
    self.b = self.add_variable("b", shape=[1, self.units])
    self.alpha = self.add_variable("alpha", shape=[self.eta, 1, self.units])
    self.beta = self.add_variable("beta", shape=[self.ksi, 1, self.units])
    
  def call(self, inputs):
    x = matmul(inputs, self.w) + self.b
    #print(x)
    P = P_fcn(self.eta, x)
    #print(P)
    B = B_fcn(self.ksi, self.eta, self.a, x)
    #print(B)
    P_alpha = tf.multiply(P, self.alpha)
    #print(P_alpha)
    B_beta = tf.multiply(B, self.beta)
    #print(B_beta)
    output = tf.reduce_sum(P_alpha, axis=0) + tf.reduce_sum(B_beta, axis=0)
    return output