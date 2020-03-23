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

matplotlib.use('Agg')
pd.options.mode.use_inf_as_na = True

class PackNumericFeatures(object):
    def __init__(self, names, num_classes=None):
        self.names = names
        self.num_classes = num_classes

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32)
                            for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features
        if self.num_classes!=None:
            labels=tf.one_hot(labels, self.num_classes)
        return features, labels 

def read_maps(filename):
    """
    reads the generated map file, useful for metadata for embedding layer weights.
    Not really a map, but a list where index maps to the element at that index.

    Args:
        filename (string): the filepath to the map file.

    Returns:
        list: elements contained in map file.

    """
    f = open(filename, "r")
    map = []
    for i in f.readlines():
        _, value = i.rstrip().split(",")
        map.append(value)
    return map


def check_float(x):
    """
    convert value to float. used because column 14 and 15 have Infinity as its value.

    Args:
        x (string): input `x`.

    Returns:
        float: float version of x.

    """
    x = float(x)
    return x


def save_map(filename, map_list):
    """saves the map_list into file called filename.

    Args:
        filename (string): where to save the file.
        map_list (list): a list where index is key and element at the index is value.

    Returns:
        nothing

    """
    f = open(filename, "w")
    for index, value in enumerate(map_list):
        f.write("{},{}\n".format(index, value))
    f.close()


def get_field_names(filename):
    """
    simple function to get field names

    Args:
        filename (string): path to file containing fieldnames, note not the NSL_KDD file.

    Returns:
        list: list of field names.

    """
    f = open(filename)
    field_names = []
    for i in f.readlines():
        field_names.append(i.rstrip().split(",")[-1])
    return field_names


def load_dataset(dataset_name, sets=["train", "test", "val"], label_name="Label", batch_size=128):
    """returns various samples of datasets. the samples are defined by prefix_suffix,
    e.g. train_x

    Args:
        dataset_name (string): name of the dataset.
        sets (array): a list of the sample name. defaults to ["train","test","val"]
        label_name (string): column name of the label. defaults to "Label"

    Returns:
        list of array: a list of array where indices correspond to prefix[0]_suffix[0], prefix[0]_suffix[1] ...

    """
    return_sets = []
    print("loading dataset:", dataset_name)
    for set in sets:
        print("loading sample set:", set)

        data = tf.data.experimental.make_csv_dataset(
            "../data/{}/{}.csv".format(dataset_name, set),
            batch_size=batch_size, label_name=label_name)

        return_sets.append(data)
    return return_sets


def save_to_csv(filename, content):
    """write content to filename as csv file.

    Args:
        filename (string): the location to of the saved file.
        content (array): the data that is intended to be saved.

    Returns:
        Nothing.

    """
    with open(filename, "w") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(content)
    csv_file.close()


def get_attack_types(filename):
    """Returns the attack types as dictionary.

    Args:
        filename (string): filename of the file to read attack types from.

    Returns:
        dictionary: dictionary of the attack types described in filename.

    """
    type_file = open(filename, "r")
    type_dict = {}
    for line in type_file.readlines()[1:]:
        child_type, parent_type = line.rstrip().split(",")
        if parent_type not in type_dict.keys():
            type_dict[parent_type] = []
        type_dict[parent_type].append(child_type)
    return type_dict


def split_dataframe(df, split_percentage, random_state=0):
    """Splits dataframe given percentage. Returns 2 dataframes with the second
    dataframe containing split_percentage% of df

    Args:
        df (pandas dataframe): the dataframe to split.
        split_percentage (int): the split percentage in integer.
        random_state (type): random state for error checking. Defaults to 0.

    Returns:
        dataframes: 2 dataframes with the first containing (100-split_percentage) of df

    """
    df1, df2 = train_test_split(
        df, test_size=split_percentage, random_state=random_state)
    return df1, df2


class DataReader:
    def __init__(self, data_directory, num_features, train_test_split, test_val_split, attack_type=None, dataset_name=None):
        """initializes the data reader for CIC-IDS datasets.

        Args:
            dataset_name (string): name of the dataset generated, the dataset will be saved in ../data/{dataset_name}
            data_directory (string list): list of locations to look for csv data.
            num_features (int): number of features excluding the label
            train_test_split (float): percentage of all files in test.
            test_val_split (float): percentage of test files in validation.
            attack_type (string list): list of attack_types to include

        Returns:
            nothing
        """
        # if dataset name is not specified, use concatenation of attack types.
        if dataset_name == None:
            self.dataset_name = "+".join(attack_type)
        else:
            self.dataset_name = dataset_name

        self.data_directory = data_directory
        self.train_test_split = train_test_split
        self.test_val_split = test_val_split
        self.attack_type = attack_type
        self.num_features = num_features

    def generate_dataframes(self):
        """
        generates train test and val pandas dataframes.
        unlike NSK KDD the datasets contains headers and labels in one csv file
        Also saves statistics and meta data about the dataset
        Returns:
            nothing.

        """
        pp = pprint.PrettyPrinter(indent=4, compact=True)

        # get dataframes
        dataframe, attack_label = self.generate_dataframe()

        # save metadata about the data for processing later
        metadata = {}
        metadata["num_classes"] = len(attack_label)
        metadata["col_max"] = dataframe.max(axis=0).tolist()
        metadata["col_min"] = dataframe.min(axis=0).tolist()
        metadata["col_mean"] = dataframe.mean(axis=0).tolist()
        metadata["col_std"] = dataframe.std(axis=0).tolist()
        metadata["field_names"]=dataframe.columns.tolist()
        # dtype object not serializable so turn into string first
        dtypes=[str(x) for x in dataframe.dtypes]
        metadata["dtypes"]=dtypes
        

        # create dataset folder if it doesnt exist
        if not os.path.exists("../data/{}".format(self.dataset_name)):
            os.mkdir("../data/{}".format(self.dataset_name))
            os.makedirs("../data/{}/maps".format(self.dataset_name))
            os.makedirs("../data/{}/stats".format(self.dataset_name))

        # split data into train, val, test and save
        self.train_data, test_data = split_dataframe(
            dataframe, self.train_test_split)
        self.test_data, self.val_data = split_dataframe(
            test_data, self.test_val_split)

        num_train = len(self.train_data.index)
        num_val = len(self.val_data.index)
        num_test = len(self.test_data.index)

        print("test_data splitted into train:{}, test:{}, val:{}".format(
            num_train, num_test, num_val))

        metadata["num_train"] = num_train
        metadata["num_val"] = num_val
        metadata["num_test"] = num_test

        self.dataframe = dataframe
        # save the maps
        save_map("../data/{}/maps/{}.csv".format(self.dataset_name,
                                                 "attack label"), attack_label)

        with open('../data/{}/metadata.txt'.format(self.dataset_name), 'w') as outfile:
            json.dump(metadata, outfile, indent=True)

    def dataset_statistics(self):
        counts_file = open(
            "../data/{}/stats/counts.txt".format(self.dataset_name), "w")
        counts_file.write("all samples:\n{}\n".format(
            self.dataframe["Label"].value_counts()))
        counts_file.write("train samples:\n{}\n".format(
            self.train_data["Label"].value_counts()))
        counts_file.write("test samples:\n{}\n".format(
            self.test_data["Label"].value_counts()))
        counts_file.write("val samples:\n{}\n".format(
            self.val_data["Label"].value_counts()))
        counts_file.close()

        # draw distributions of each attribute for all data
        axes = self.dataframe.hist(figsize=(50, 50))

        plt.savefig("../data/{}/stats/hist_all.png".format(self.dataset_name))

    def write_to_csv(self):
        """writes train, val and test data to csv file. This is a expensive operation.

        Returns:
            None: file written to data/{datasetname}/

        """
        self.train_data.to_csv(
            "../data/{}/train.csv".format(self.dataset_name), index=False)
        self.val_data.to_csv(
            "../data/{}/val.csv".format(self.dataset_name), index=False)
        self.test_data.to_csv(
            "../data/{}/test.csv".format(self.dataset_name), index=False)

    def generate_dataframe(self):
        """
        Generates Pandas dataframe from self.data_directory.
        converts Label to numerical data.

        Returns:
            dataframe, array: dataframe and the attack label mapping.

        """

        datasets = []
        # get all files under data_directory
        for file in os.listdir(self.data_directory):
            # ignore .gitignore
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(
                    self.data_directory, file), header=0,
                    converters={14: check_float, 15: check_float}, encoding="utf-8")
                datasets.append(df)
        all_data = pd.concat(datasets)

        # some headers have spaces in front
        all_data = all_data.rename(columns=lambda x: x.lstrip())

        # filter attacks
        if self.attack_type is not None:
            all_data = all_data[all_data["Label"].isin(self.attack_type)]

        # convert label to categorical
        label_map = list(all_data["Label"].astype(
            "category").cat.categories)
        all_data["Label"] = all_data["Label"].astype(
            "category").cat.codes

        # remove negative and nan values
        all_data[all_data < 0] = np.nan
        all_data = all_data.dropna()

        return all_data, label_map

    def start(self):
        """
        runs the whole thing:
        generate_dataframes
        write_to_csv
        dataset_statistics

        Returns:
            None

        """
        self.generate_dataframes()
        self.write_to_csv()
        self.dataset_statistics()
