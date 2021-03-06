import json

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from input_utils import load_dataset, read_maps
from matplotlib.colors import Normalize, to_hex
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (Dense, Embedding, Flatten, Input,
                                     concatenate)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from train import PackNumericFeatures

matplotlib.use('Agg')


def predict_sample(data_name, model_path, dataset_name, batch_size=1024):
    data = tf.data.experimental.make_csv_dataset("../experiment/attack_pcap/{}".format(
        data_name), batch_size=batch_size, label_name="Label", shuffle=False)
    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)
    field_names = metadata["field_names"][:-1]
    packed_data = data.map(PackNumericFeatures(field_names))

    model = load_model(model_path)

    predictions = model.predict(packed_data, steps=1)

    p = []
    for batch in predictions:
        print(batch)
        p.append(np.argmax(batch))
    fig, ax = plt.subplots()
    ax.plot(p)
    fig.savefig("../experiment/attack_pcap/{}.png".format(data_name))
