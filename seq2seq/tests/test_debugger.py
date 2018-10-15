#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 14:52:22 2018

@author: rgarzon
"""

from __future__ import print_function
from tensorflow.python import debug as tf_debug


import tensorflow as tf

# Simple hello world using TensorFlow

# Create a Constant op
# The op is added as a node to the default graph.
#
# The value returned by the constructor represents the output
# of the Constant op.
hello = tf.constant('Hello, TensorFlow!')

# Start tf session
sess = tf.Session()
sess = tf_debug.LocalCLIDebugWrapperSession(sess)


# Run the op
print(sess.run(hello))

