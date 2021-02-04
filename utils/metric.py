from keras.metrics import *
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import numpy as np


def TruePositives(true_y,pred_y):
    m = tf.keras.metrics.TruePositives()
    m.update_state(true_y,pred_y)
    TP=m.result().numpy()
    return TP

def TrueNegatives(true_y,pred_y):
    m = tf.keras.metrics.TrueNegatives()
    m.update_state(true_y,pred_y)
    TN=m.result().numpy()
    return TN

def FalsePositives(true_y,pred_y):
    m = tf.keras.metrics.FalsePositives()
    m.update_state(true_y,pred_y)
    FP=m.result().numpy()
    return FP

def FalseNegatives(true_y,pred_y):
    m = tf.keras.metrics.FalseNegatives()
    m.update_state(true_y,pred_y)
    FN=m.result().numpy()
    return FN

