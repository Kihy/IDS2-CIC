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

def vis_diff(file1,file2,out_file):
    df1=pd.read_csv(file1)
    df1 = df1.drop(labels='Label', axis=1)
    df2=pd.read_csv(file2)
    df2 = df2.drop(labels='Label', axis=1)
    diff=df1-df2

    axes=diff.hist(figsize=(25, 25),log=True)

    plt.savefig(out_file)

def vis_clusters(metadata, label_name, datafile, field_names):
    metadata_df=pd.read_csv(metadata, sep='\t', usecols=["label","idx"])
    data_df = pd.read_csv(datafile, usecols=field_names+["idx"])
    metadata_df=metadata_df[metadata_df["label"].isin(label_name)]
    merge_df=metadata_df.merge(data_df,how="left", on="idx")
    n_cols=6
    n_rows=np.math.ceil(len(field_names)/n_cols)
    f,ax=plt.subplots(figsize=(20, 20),nrows=n_rows,ncols=n_cols)

    for i in label_name:
        tmp=merge_df[merge_df["label"]==i]
        for j in range(len(field_names)):
            r=j//n_cols
            c=j-n_cols*r
            ax[r][c].hist(tmp[field_names[j]], alpha=0.5, label=i)
            ax[r][c].set_title(field_names[j])

    plt.tight_layout()
    plt.savefig("../experiment/aae_vis/cluster_vis_{}.png".format(label_name))
