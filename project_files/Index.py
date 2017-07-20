import tensorflow as tf
import numpy as np





def main():
  with tf.device('./cpu:0'):
    x = tf.placeholder(tf.float32, [32, 32])
    y = tf.placeholder(tf.float32, [9])
    
    W_conv1 = tf.Variable([3





if __name__ == '__main__':
  main()
