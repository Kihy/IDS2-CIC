import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from input_utils import load_dataset, read_maps
from sklearn import preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from tensorflow.keras import optimizers
from tensorflow.keras.layers import (Dense, Embedding, Flatten, Input,
                                     concatenate)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from matplotlib.colors import to_hex, Normalize
from input_utils import *
import json
from aae_dim_reduce import WcLayer
matplotlib.use('Agg')

def generate_fake_data(model_path, fieldnames, n_samples, dataset_name):
    #last fieldname is label and first column is always 0
    samples=np.random.rand(n_samples, len(fieldnames)-2)
    samples=np.c_[np.zeros(n_samples),samples ]
    print(samples)
    np.savetxt("../experiment/aae_vis/fake.csv",samples, delimiter=",",header=",".join(fieldnames))
    inputs={'numeric':samples.astype('float32')}
    model = tf.saved_model.load(model_path)
    pred=model(inputs)
    pred=np.argmax(pred,axis=1)
    np.savetxt("../experiment/aae_vis/fake_pred.csv", pred, delimiter=",")

    custom_objects = {'WcLayer': WcLayer}
    prefix="{}_{}_{}".format(dataset_name, 3, False)
    aae = tf.keras.models.load_model("../models/aae/{}_aae.h5".format(prefix), custom_objects=custom_objects)
    encoder = tf.keras.models.load_model("../models/aae/{}_encoder.h5".format(prefix))
    decoder = tf.keras.models.load_model("../models/aae/{}_decoder.h5".format(prefix))
    encoded,label=encoder(samples)
    np.savetxt("../experiment/aae_vis/fake_encode.csv",encoded,delimiter="\t")



def evaluate_network(dataset_name, model_path, output_name, batch_size=1024, label_name="Label"):
    """
    evaluates the a model. generates classification report (including precision recall and accuracy) and a graphical
    version of the classification report. classification report has name output_name.txt and
    the heatmap have name output_name_heatmap.png.
    The evaluation takes place on the first batch of the test set.

    Args:
        dataset_name (string): the name of the dataset.
        model_path (string): path to some saved model.
        output_name (string): name of the output file JUST THE NAME NOT PATH. the
        path is experiment/reports/

    Returns:
        None: the classification report and its graphical representation is saved
        at output_name.

    """
    subset="train"

    test = load_dataset(
        dataset_name, sets=[subset], label_name=label_name, batch_size=batch_size)[0]

    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)


    field_names = metadata["field_names"][:-1]
    min = np.array(metadata["col_min"][:-1])
    max = np.array(metadata["col_max"][:-1])
    normalizer,_ = min_max_scaler_gen(min, max)

    packed_test=test.map(PackNumericFeatures(field_names, metadata["num_classes"], scaler=normalizer))

    print("evaluated on batch_size:",batch_size)





    model = load_model(model_path)

    y_pred_all=[]
    labels_all=[]

    for features, labels in packed_test.take(metadata["num_{}".format(subset)]//batch_size):

        # generate predictions and decode them
        y_pred = model(features)

        y_pred = np.argmax(y_pred, axis=1)
        labels=np.argmax(labels, axis=1)
        y_pred_all.extend(y_pred)
        labels_all.extend(labels)


    attack_label = read_maps(
        "../data/{}/maps/{}.csv".format(dataset_name,label_name))


    # draws a heat_map of the classification report
    report = classification_report(labels_all, y_pred_all, target_names=attack_label, labels=list(
        range(len(attack_label))), output_dict=True)
    draw_heatmap(report, output_name + "_heatmap")

    # save a text version of the report as well
    report = classification_report(
        labels_all, y_pred_all, target_names=attack_label, labels=list(range(len(attack_label))))
    report_file = open("../experiment/reports/{}.txt".format(output_name), "w")
    report_file.write(report)
    report_file.close()


def draw_heatmap(report, filename):
    """
    draws a heatmap representation of classification report.

    Args:
        report (dict): the classification report dictionary.
        filename (string): the name of the output graph JUST THE NAME NOT PATH.

    Returns:
        None: the graph is saved at experiments/reports/.

    """

    attr = ["precision", "recall", "f1-score", "support"]
    # dont really want the last 3 rows of averages
    samples = list(report.keys())[:-3]
    heat_map = []

    for key in samples:
        heat_map.append(np.array(list(report[key].values())))

    heat_map = np.array(heat_map)
    fig, ax = plt.subplots(figsize=(9, len(samples)*2))

    m = np.zeros_like(heat_map)
    m[:, 3] = 1

    im1 = ax.imshow(np.ma.masked_array(heat_map, m), cmap="YlGn")
    im2 = ax.imshow(np.ma.masked_array(
        heat_map, np.logical_not(m)), cmap="GnBu")

    # Create colorbar
    cbarlabel = "percentage"
    cbar = ax.figure.colorbar(im1, ax=ax, aspect=50)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    cbarlabel = "number"
    cbar = ax.figure.colorbar(im2, ax=ax, aspect=50)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(attr)))
    ax.set_yticks(np.arange(len(samples)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(attr)
    ax.set_yticklabels(samples)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    textcolors = ["black", "white"]
    threshold1 = im1.norm(im1.get_array().max()) / 2.
    threshold2 = im2.norm(im2.get_array().max()) / 2.
    # Loop over data dimensions and create text annotations.
    for i in range(len(samples)):
        for j in range(len(attr)):
            text_val = heat_map[i, j]
            # change text color if background color is too light
            if j == 3:
                color = textcolors[int(im2.norm(text_val) > threshold2)]
            else:
                color = textcolors[int(im1.norm(text_val) > threshold1)]
            text = ax.text(j, i, "{:.3f}".format(text_val),
                           ha="center", va="center", color=color)

    ax.set_title("Classification Report")
    fig.tight_layout()
    fig.savefig("../experiment/reports/{}.png".format(filename))
