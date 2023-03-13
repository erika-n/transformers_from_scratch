import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding

import math

# results:
# train: accuracy: 0.3394
# test: accuracy: 0.3286

class MinimalTransformer(tf.keras.Model):
  def __init__(self, input_length, vocab_size):
    super(MinimalTransformer, self).__init__()
    self.embedding_dim = 300
    self.dk = 64
    self.input_length = input_length
    
    self.embedding = Embedding(vocab_size, self.embedding_dim, input_length=input_length)
    self.wq = Dense(self.dk)
    self.wk = Dense(self.dk)
    self.wv = Dense(self.dk)
    self.out = Dense(vocab_size, activation="relu")

  def call(self, x):
    e = self.embedding(x)
    q = self.wq(e)
    k = self.wk(e)
    v = self.wv(e)
    w = (tf.matmul(q, tf.transpose(k, perm=[0, 2, 1])))/math.sqrt(self.dk)
    #todo: masked attention
    z = w @ v
    z = self.out(z)
    return tf.nn.softmax(z)
