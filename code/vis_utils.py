import csv
import json
import logging
import os
import pprint
from itertools import product

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical


def vis_original_input(filename,attack_type):
    """
    visualizes the original data in time sequential order.

    Args:
        filename (string): file name of the attack file.
        attack_type (string): attack type to include.

    Returns:
        None: plot of the attack types

    """
    with open(filename) as file:
        p=[]
        for line in file.readlines()[1:]:
            attack=line.rstrip().split(",")[-1]
            if attack in attack_type:
                p.append(attack)
        print(len(p))
        fig, ax = plt.subplots(figsize=(len(p)//2000,5))
        ax.plot(p)
        fig.tight_layout()
        fig.savefig("../experiment/attack_pcap/real_attack.png")

def vis_attack_distribution(data_path, out_file, ignore_label=True, attack=None):
    """
    visualizes the distribution of a set of data with histograms for each field. The axis is logarithmic
    to account for drastic variations in attribute data ranges.

    Args:
        data_path (string): path to csv file.
        out_file (string): output histogram file path
        ignore_label (boolean): whether to ignore label set to true if using real data. Defaults to True.
        attack (int): attacks index to include, if None all attacks are included, cannot use this if ignore_label is True. Defaults to None.

    Returns:
        None: a histogram file produced at out_file.

    """
    df=pd.read_csv(data_path)

    if ignore_label:
        df=df.drop(["Label"],axis=1)

    if attack is not None:
        df=df.loc[df["Label"]==attack]

    axes=df.hist(figsize=(25, 25),log=True)

    plt.savefig(out_file)
