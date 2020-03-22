import csv
import sys
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from input_utils import get_field_names
from numpy import genfromtxt
from sklearn import preprocessing

matplotlib.use('Agg')

def find_theta(filename, percent_theta):
    """
    finding the theta used in adversarial perturbation, as different data types
    handle different different theta. For symbolic data theta is set to difference between each of
    its level (e.g. for boolean attribute theta is 1). For continuous data it is set to
    percent_theta. For discrete data it changes by percent_theta of the range rounded evenly to
    nearest integer value before scaling, if data_range is too small treat like symbolic data.

    Args:
        filename (type): metadata file containing filename and dtypes
        percent_theta (float): how much continuous attribute change as percentage (0~1).

    Returns:
        array: the theta values for each feature.

    """
    f = open(filename, "r")
    data_range = scaler.data_range_
    thetas = np.zeros((len(data_range),))
    index = 0
    for line in f.readlines():
        if data_range[index] == 0:
            thetas[index] = 0
            index += 1
            continue
        type = line.rstrip().split(",")[-1]
        # if symbolic theta is set to change by 1 level
        if type == "symbolic":
            theta = 1 / data_range[index]
            thetas[index] = theta
        # if continous theta is set to percent theta
        if type == "continuous":
            thetas[index] = percent_theta
        # if discrete theta is set to percent theta of the data range rounded evenly
        if type == "discrete":
            theta = np.around(
                percent_theta * data_range[index]) / data_range[index]
            if theta == 0:
                theta = 1 / data_range[index]
            thetas[index] = theta
        index += 1

    return thetas