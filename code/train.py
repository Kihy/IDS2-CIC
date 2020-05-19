import functools
import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from input_utils import *
from matplotlib import cm
from matplotlib.colors import Normalize, to_hex
from scipy.spatial import ConvexHull
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
# from stats_utils import *
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (Dense, Embedding, Flatten, Input,
                                     concatenate)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model

matplotlib.use('Agg')



def min_max_scaler_gen(min,max):
    def min_max_scaler(data):
        """
        scales the input according to metadata.

        Args:
            feature (ordered dict): feature from tf.dataset.
            label (ordered dict): labels.

        Returns:
            ordered dict, ordered dict: the scaled input and label with same size as input.

        """
        data_range=max-min
        # replace 0 with 1 so it does not produce nan
        data_range=np.where(data_range!=0, data_range, 1)

        x_std = (data - min) /data_range


        return x_std
    return min_max_scaler


def train_normal_network(dataset_name, save_path, batch_size=128, epochs=50, label_name="Label"):
    """
    trains a normal 3 fully connected layer network. uses train and val subset of
    dataset_name.

    Args:
        dataset_name (string): name of the dataset to be trained on.
        save_path (string): the path to where the model is saved.
        batch_size (int): batch size for training. Defaults to 128.
        epochs (int): number of epochs for training. Defaults to 50.

    Returns:
        None: the model is saved at save_path.

    """

    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)
    print(metadata["dtypes"])
    train, val = load_dataset(
        dataset_name, sets=["train", "val"],
        label_name=label_name, batch_size=batch_size)

    num_classes = metadata["num_classes"]
    field_names=metadata["field_names"][:-1]

    packed_train_data = train.map(
        PackNumericFeatures(field_names,num_classes))
    packed_val_data = val.map(PackNumericFeatures(field_names, num_classes))

    # example_batch, labels_batch = next(iter(packed_train_data))

    min = np.array(metadata["col_min"][:-1])
    max = np.array(metadata["col_max"][:-1])

    input_dim=len(field_names)

    normalizer = min_max_scaler_gen(min, max)
    numeric_column = tf.feature_column.numeric_column(
        'numeric', normalizer_fn=normalizer, shape=(input_dim,))
    numeric_columns = [numeric_column]
    numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns, name='scaler')

    inputs = {
        'numeric': tf.keras.layers.Input(name='numeric', shape=(input_dim,), dtype='float32')
    }
    # inputs=tf.keras.layers.Input(name='input', shape=(input_dim,), dtype='float32')
    dense_input = numeric_layer(inputs)
    dense = Dense(41, activation='relu')(dense_input)
    dense1 = Dense(41, activation='relu')(dense)
    dense2 = Dense(41, activation='relu')(dense1)
    output = Dense(num_classes, activation='sigmoid')(dense)

    # return
    model = Model(inputs=inputs, outputs=output)
    dense_layer_weights=model.layers[2].get_weights()[0]
    # print(numeric_layer(example_batch))
    # print(np.matmul(numeric_layer(example_batch),dense_layer_weights))

    model.compile(optimizer='adam',
                  loss="categorical_crossentropy", metrics=["accuracy"])
    # plot_model(model, "network_graphs/dense_feature_network.png",show_shapes=True)
    # model.summary()

    model.fit(packed_train_data,
              steps_per_epoch=metadata["num_train"]//batch_size,
              epochs=epochs,
              validation_data=packed_val_data,
              validation_steps=metadata["num_val"]//batch_size
              )
    tf.saved_model.save(model, save_path)
    # model.save(save_path)
