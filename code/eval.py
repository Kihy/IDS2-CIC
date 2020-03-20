import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import progressbar
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
matplotlib.use('Agg')

def evaluate_network(dataset_name, model_path, output_name, embed=False, batch_size=1024):
    """
    evaluates the a model. generates classification report (including precision recall and accuracy) and a graphical
    version of the classification report. classification report has name output_name.txt and
    the heatmap have name output_name_heatmap.png

    Args:
        dataset_name (string): the name of the dataset.
        model_path (string): path to some saved model.
        output_name (string): name of the output file JUST THE NAME NOT PATH. the
        path is experiment/reports/
        embed (boolean): whether the network is embedded. Defaults to False.

    Returns:
        None: the classification report and its graphical representation is saved
        at output_name.

    """
    test = load_dataset(
        dataset_name, prefix=["test"],label_name="Label", batch_size=batch_size)

    model = load_model(model_path)

    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)


    # generate predictions and decode them
    y_pred = model.predict(test, steps=1)

    y_pred = np.argmax(y_pred, axis=1)

    y_true=list(test[0][1])

    attack_label = read_maps(
        "../data/{}/maps/attack_type.csv".format(dataset_name))

    # draws a heat_map of the classification report
    report = classification_report(y_true, y_pred, target_names=attack_label, labels=list(
        range(len(attack_label))), output_dict=True)
    draw_heatmap(report, output_name + "_heatmap")

    # save a text version of the report as well
    report = classification_report(
        y_true, y_pred, target_names=attack_label, labels=list(range(len(attack_label))))
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
    fig, ax = plt.subplots(figsize=(10, 30))

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
