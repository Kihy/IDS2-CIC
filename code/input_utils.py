import csv
import json
import logging
import os
import pprint
from itertools import product

import numpy as np
import pandas as pd
from numpy import genfromtxt

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

matplotlib.use('Agg')
pd.options.mode.use_inf_as_na = True


def min_max_scaler_gen(min, max):
    def min_max_scaler(data):
        """
        scales the input according to metadata.

        Args:
            feature (ordered dict): feature from tf.dataset.
            label (ordered dict): labels.

        Returns:
            ordered dict, ordered dict: the scaled input and label with same size as input.

        """
        data_range = max - min
        # replace 0 with 1 so it does not produce nan
        data_range = np.where(data_range != 0, data_range, 1)

        x_std = (data - min) / data_range

        return x_std

    def min_max_unscaler(data):
        data_range = max - min

        data_range = np.where(data_range != 0, data_range, 1)

        unscaled=data*data_range+min

        return unscaled

    return min_max_scaler, min_max_unscaler


class PackNumericFeatures(object):
    """
    packs the features from tensorflow's csv dataset pipeline.

    Args:
        names (string list): feature names.
        num_classes (int): number of classes, used for one hot encoding of labels. Defaults to None.
        vae (boolean): Whether the return value is in vae format, i.e. label = features. Defaults to False.
        scaler (func): a scaling function if you decide to do normalization at this stage. Defaults to None.

    Attributes:
        names
        num_classes
        vae
        scaler

    """

    def __init__(self, names, num_classes=None, scaler=None):
        self.names = names
        self.num_classes = num_classes
        self.scaler = scaler

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32)
                            for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)

        if self.scaler != None:
            numeric_features = self.scaler(numeric_features)

        features['numeric'] = numeric_features
        if self.num_classes != None:
            labels = tf.one_hot(labels, self.num_classes)

        return features, labels


def show_batch(dataset):
    for batch, label in dataset.take(1):
        print(label)
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


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
    convert value to float. used because column 14 and 15 have Infinity, Nan, '' as its value.

    Args:
        x (string): input `x`.

    Returns:
        float: float version of x.

    """
    if x == "":
        x = 0
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


def load_dataset(dataset_name, sets=["train", "test", "val"], **kwargs):
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
            "../data/{}/{}.csv".format(dataset_name, set), **kwargs)

        return_sets.append(data)
    print("finished loading dataset")
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


def get_column_map(path):
    """
    get the column mapping from raw extracted data to ml data.

    Args:
        path (string): path to mapping file

    Returns:
        type: Description of returned object.

    """
    map_file = open(path)
    dict = {}
    for i in map_file.readlines():
        key, value = i.rstrip().split(",")
        if value == "same":
            dict[key] = key
        elif value == "None":
            dict[key] = "remove"
        else:
            dict[key] = value
    map_file.close()
    return dict


def add_label_col(file):
    df = pd.read_csv("../experiment/attack_pcap/" + file)
    protocol_map = {}
    with open("../data/dos_pyflowmeter/maps/protocol.csv") as f:
        for i in f.readlines():
            key, val = i.rstrip().split(",")
            protocol_map[val] = key

    df["Label"] = file.split(".")[0]
    df["protocol"] = df["protocol"].apply(lambda x: protocol_map[x])
    df.to_csv("../experiment/attack_pcap/" + file, index=False)


def format_converter(data_directory, column_map_path, **kwargs):
    """
    converts raw extracted data(flows) to machine learning format, also produces a
    metadata file with number of samples and field names

    Args:
        data_directory (string): directory containing raw data flows or file
        column_map_path (string): path to column mapping file.

    Returns:
        None: converted file and metadata file is stored at experiment/attack_pcap/.

    """
    dict = get_column_map(column_map_path)
    if os.path.isfile(data_directory):
        convert_file(data_directory, dict, **kwargs)
    else:
        for file in os.listdir(data_directory):
            if file.endswith(".csv"):
                convert_file(os.path.join(
                    data_directory, file), dict, **kwargs)


def convert_file(file, col_map, out_dir, metadata=False, use_filename_as_label=False):
    """
    converts single file from Flow format to ml format.

    Args:
        file (string): path to file.
        col_map (dict): column name map.

    Returns:
        None: output files saved at experiment/attack_pcap.

    """

    print("processing file: {}".format(file))

    df = pd.read_csv(file, header=0, encoding="utf-8")
    df = df.rename(columns=col_map)
    df = df.drop(columns=['remove'])

    file_name = file.split("/")[-1]

    if use_filename_as_label:
        df = df.replace("No Label", file_name.split(".")[0])
    print(metadata)
    if metadata:
        meta_dict = {}
        print("generating metadata")
        meta_dict["field_names"] = df.columns.tolist()
        meta_dict["num_samples"] = len(df.index)
        with open('{}metadata_{}'.format(out_dir, file_name), 'w') as outfile:
            json.dump(meta_dict, outfile, indent=True)
    df.to_csv("{}{}".format(out_dir, file_name), index=False)


class DataReader:
    def __init__(self, data_directory, train_test_split, test_val_split, files=[], protocols=[], columns=[], label_col="Label", ignore=False, attack_type=None, dataset_name=None, use_filename_as_label=False):
        """initializes the data reader for CIC-IDS datasets.

        Args:
            dataset_name (string): name of the dataset generated, the dataset will be saved in ../data/{dataset_name}
            data_directory (string list): list of locations to look for csv data.
            num_features (int): number of features excluding the label
            files(list): a list of file to process, depends on ignore
            protocols (string list): list of protocols to include
            columns(string list): list of columns to includes
            label_col(string): name of the label column
            ignore(boolean): if set to True, only the files in the files list are processed.
            if set to False, the files in files are ignored. to process all files set files to [] and ignore to False.
            train_test_split (float): percentage of all files in test.
            test_val_split (float): percentage of test files in validation.
            attack_type (string list): list of attack_types to include
            use_filename_as_label(boolean): whether to use filename as labels. defaults to False

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
        self.files = files
        self.ignore = ignore
        self.use_filename_as_label = use_filename_as_label
        self.protocols = protocols
        self.columns = columns
        self.label_col=label_col

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
        dataframe, maps, num_classes = self.generate_dataframe()

        # save metadata about the data for processing later
        metadata = {}
        metadata["num_classes"] = num_classes
        metadata["col_max"] = dataframe.max(axis=0).tolist()
        metadata["col_min"] = dataframe.min(axis=0).tolist()
        metadata["col_mean"] = dataframe.mean(axis=0).tolist()
        metadata["col_std"] = dataframe.std(axis=0).tolist()
        metadata["field_names"] = dataframe.columns.tolist()
        # dtype object not serializable so turn into string first
        dtypes = [str(x) for x in dataframe.dtypes]
        metadata["dtypes"] = dtypes

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
        for key, val in maps.items():
            save_map("../data/{}/maps/{}.csv".format(self.dataset_name,
                                                     key), val)

        with open('../data/{}/metadata.txt'.format(self.dataset_name), 'w') as outfile:
            json.dump(metadata, outfile, indent=True)

    def dataset_statistics(self):
        counts_file = open(
            "../data/{}/stats/counts.txt".format(self.dataset_name), "w")
        counts_file.write("all samples:\n{}\n".format(
            self.dataframe[self.label_col].value_counts()))
        counts_file.write("train samples:\n{}\n".format(
            self.train_data[self.label_col].value_counts()))
        counts_file.write("test samples:\n{}\n".format(
            self.test_data[self.label_col].value_counts()))
        counts_file.write("val samples:\n{}\n".format(
            self.val_data[self.label_col].value_counts()))
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
            is_in_files=False
            for i in self.files:
                if i in file:
                    is_in_files=True
            if file.endswith(".csv") and (is_in_files == self.ignore):
                print("processing file", file)
                df_chunk = pd.read_csv(os.path.join(
                    self.data_directory, file), header=0, chunksize=100000,
                    usecols=self.columns, encoding="utf-8")

                for chunk in df_chunk:
                    if self.use_filename_as_label:
                        chunk[self.label_col] = file.split(".")[0]

                    if len(self.protocols) > 0:
                        chunk = chunk[chunk["protocol_type"].isin(
                            self.protocols)]
                    datasets.append(chunk)

        print("finished loading datasets")
        all_data = pd.concat(datasets)
        # some headers have spaces in front
        all_data = all_data.rename(columns=lambda x: x.lstrip())

        # drop duplicate since duplicate columns ends with .n
        for colname in all_data.columns:
            if colname[-1].isdigit():
                all_data = all_data.drop([colname], axis=1)

        # filter attacks
        if self.attack_type is not None:
            all_data = all_data[all_data[self.label_col].isin(self.attack_type)]
            if all_data.empty:
                raise ValueError("Specified attack type results in empty dataframe")
        # convert label to categorical
        maps = {}
        cat_col = all_data.select_dtypes(['object']).columns
        all_data[cat_col] = all_data[cat_col].astype("category")

        for i in cat_col:
            maps[i] = list(all_data[i].cat.categories)
        all_data[cat_col] = all_data[cat_col].apply(lambda x: x.cat.codes)

        num_classes = all_data[self.label_col].nunique()

        # # remove negative and nan values
        # all_data[all_data < 0] = np.nan
        # all_data = all_data.fillna(0)

        return all_data, maps, num_classes

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
