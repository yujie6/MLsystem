#!/bin/python3.7
import tensorflow as tf
import numpy as np
from vector import *

a = Vector(1, 2)
b = Vector(3, 4)
print(a + b)

x = tf.ones((2, 2))

# 需要计算梯度的操作
with tf.GradientTape() as t:
    t.watch(x)
    y = tf.reduce_sum(x)
    z = tf.multiply(y, y)
# 计算z关于x的梯度
dz_dx = t.gradient(z, x)
print(dz_dx)
