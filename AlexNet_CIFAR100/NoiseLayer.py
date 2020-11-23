#import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
#from tensorflow.python.ops import math_ops
#from tensorflow.python.util.tf_export import keras_export

#@keras_export('keras.layers.GaussianNoise_new')
class GaussianNoise_new(Layer):

  def __init__(self, **kwargs):
    super(GaussianNoise_new, self).__init__(**kwargs)
    self.supports_masking = True

  def call(self, inputs, training=None):
    stddev = K.sqrt(K.mean(K.square(inputs)))*0.05
    def noised():
      return inputs + K.random_normal(shape=array_ops.shape(inputs), mean=0., stddev=stddev)

    return K.in_train_phase(noised, noised, training=training)

  def get_config(self):
    base_config = super(GaussianNoise_new, self).get_config()
    return dict(list(base_config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape