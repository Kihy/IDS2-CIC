import csv
import sys
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from input_utils import *
from numpy import genfromtxt
from sklearn import preprocessing
import json
from input_utils import load_dataset
matplotlib.use('Agg')

def find_theta(metadata, percent_theta):
    """
    finding the theta used in adversarial perturbation, as different data types
    handle different different theta. For symbolic data theta is set to difference between each of
    its level (e.g. for boolean attribute theta is 1). For continuous data it is set to
    percent_theta. For discrete data it changes by percent_theta of the range rounded evenly to
    nearest integer value before scaling, if data_range is too small treat like symbolic data.

    Args:
        metadata (dictionary): metadata for the dataset.
        percent_theta (float): how much continuous attribute change as percentage (0~1).

    Returns:
        array: the theta values for each feature.

    """
        
    data_range = np.array(metadata["col_max"])-np.array(metadata["col_min"])
    
    # -1 so we dont include label field
    num_fields=len(metadata["field_names"])-1
    thetas = np.zeros((num_fields,))
     
    for index in range(num_fields):
        if data_range[index] == 0:
            thetas[index] = 0
            continue
        type = metadata["dtypes"][index]
        # if continous theta is set to percent theta
        if type.startswith("float"):
            thetas[index] = percent_theta
        # if discrete theta is set to percent theta of the data range rounded evenly
        if type.startswith("int"):
            theta = np.around(
                percent_theta * data_range[index]) / data_range[index]
            # in range is too small, treat like symbolic data
            if theta == 0:
                theta = 1 / data_range[index]
            thetas[index] = theta
        
    return thetas
    
def generate_forward_derivative(input_sample, model, num_classes):
    """calculates the forward derivative of model with respect to input for each
    output classes

    Args:
        input_sample (tensors): the input data as tensor.
        model (nn model): the classification model.
        num_classes (int): total number of output classes. Defaults to 2.

    Returns:
        list: the forward derivative ordered by class index

    """
    predictions = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_sample)
        prediction = model(input_sample)[0]
        for i in range(num_classes):
            predictions.append(prediction[i])

    gradients = []
    for i in range(num_classes):
        # filter out non differentiable input
        for g in tape.gradient(predictions[i], input_sample):
            if g != None:
                gradient = g.numpy()

        gradients.append(gradient.flatten())
    return gradients


def generate_saliency_map(gradients, target, map_type="increase"):
    """calculates the adversarial saliency map given gradients and target class

    Args:
        gradients (array): gradients calcualted with generate_forward_derivative function.
        target (int): the target class index.
        map_type (string): the type of saliency map, increase or decrease

    Returns:
        array: the saliency map indicating each features' impact on target

    """
    # calculate saliency map
    saliency_map = []
    # iterate through all features
    for i in range(len(gradients[0])):
        other_sum = 0
        # iterate through all classes
        for j in range(len(gradients)):
            if j != target:
                other_sum += gradients[j][i]

        if map_type == "increase":
            # equation in the formula
            if gradients[target][i] < 0 or other_sum > 0:
                saliency_map.append(0)
            else:
                saliency_map.append(gradients[target][i] * abs(other_sum))
        elif map_type == "decrease":
            if gradients[target][i] > 0 or other_sum < 0:
                saliency_map.append(0)
            else:
                saliency_map.append(abs(gradients[target][i]) * other_sum)

    return saliency_map


def jsma_attack(input_sample, target, model, thetas, num_classes, max_iter=100, fixed=[], embed=False):
    """applies jacobian saliency map attack on input_sample

    Args:
        input_sample (tensor): the input to be modified.
        target (int): the target class index.
        model (nn network): the neural network to be attacked.
        thetas (array): how much perturbation to add in each step for each attribute.
        max_iter (int): maximum number of iterations to carry out.
        fixed (array): the indices of features that cannot be altered

    Returns:
        array, int, int: the modified adversarial sample, number of iterations taken, prediction outcome

    """
    input_shape=input_sample.shape[1]
    perturbations = np.zeros(input_shape)
    x_adv = input_sample
    # iterate through max_iteration
    for i in range(max_iter):

        # stop if we hit target
        if np.argmax(model.predict(x_adv, steps=1)) == target:
            # print("jsma found in:", i)
            break

        # find forward gradient for each output class
        gradients = generate_forward_derivative(x_adv, model, num_classes)

        # filter out features that cannot be increased based on current value
        if embed:
            field = x_adv[-1].numpy()
        else:
            field = x_adv.numpy()

        # make sure it does not increase more than 1
        inc_mask = (field <= 1 - thetas).astype(int)
        # make sure it would not be less than 0
        dec_mask = (field >= thetas).astype(int)

        # check if any values in thetas are 0
        theta_mask = (thetas != 0).astype(int)
        inc_mask *= theta_mask
        dec_mask *= theta_mask

        # index in fixed cannot be changed
        for i in fixed:
            inc_mask[0][i] = 0
            dec_mask[0][i] = 0

        # calculate saliency map
        inc_saliency_map = generate_saliency_map(
            gradients, target, map_type="increase")
        dec_saliency_map = generate_saliency_map(
            gradients, target, map_type="decrease")

        inc_saliency_map *= inc_mask
        dec_saliency_map *= dec_mask

        # find strongest feature
        if np.max(inc_saliency_map) > np.max(dec_saliency_map):
            feature_index = np.argmax(inc_saliency_map)
            theta = thetas[feature_index]
        else:
            feature_index = np.argmax(dec_saliency_map)
            theta = -thetas[feature_index]

        # below are print statements used for debugging
        # print(feature_index)
        # print(theta)

        # print(inc_saliency_map)
        # print(dec_saliency_map)
        # print(thetas)

        # alter by theta and convert back
        perturbations[feature_index] += theta
        if embed:
            non_cat = input_sample[-1].numpy() + perturbations
            non_cat = tf.convert_to_tensor(non_cat)

            x_adv = [*input_sample[:3], non_cat]

        else:
            x_adv = input_sample.numpy() + perturbations
            x_adv = tf.convert_to_tensor(x_adv)

    # print("input_sample:", input_sample)
    # print("perturbation:", perturbations)
    # print("x_adv:", x_adv)
    # print("prediction:", np.argmax(model.predict(x_adv, steps=1)))
    # print(i)
    prediction=np.argmax(model.predict(x_adv,steps=1))
    # draw_perturbation(input_sample.numpy()[0], x_adv.numpy()[0], "../visualizations/adversarial/adv_stack_jsma.png")

    return x_adv, i, prediction

def adversarial_generation(dataset_name, model_path, target_class, set_name, num_samples=100, theta=0.01):
    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)
    
    batch_size=1
    # remove label field
    field_names=metadata["field_names"][:-1]
    num_classes=metadata["num_classes"]
    data=load_dataset(dataset_name, sets=[set_name], label_name="Label", batch_size=batch_size)[0]
    data=data.unbatch().filter(lambda feature, label: label!=target_class).batch(batch_size)
    packed_data = data.map(PackNumericFeatures(field_names))
    # take 1
    # sample, label = next(iter(packed_val_data))
    
    model = tf.keras.models.load_model(model_path)

    thetas = find_theta(metadata, theta)

    maps = genfromtxt(
        "../data/{}/maps/attack label.csv".format(dataset_name), delimiter=',')
    adv_col_names=metadata["field_names"]+["Iterations", "Adv Label"]
    adv_df=pd.DataFrame(columns=adv_col_names)
    pert_df=pd.DataFrame(columns=field_names)
    scaler=model.get_layer("scaler")
    for sample, label in packed_data.take(num_samples):
        
        # fixes flags
        jsma_sample, num_iter, pred= jsma_attack(sample["numeric"],
                           target_class, model, thetas, num_classes, max_iter=200)
        
        row=jsma_sample.numpy()
        row=np.append(row, [label.numpy()[0], num_iter, pred])
        adv_df=adv_df.append(pd.Series(row, index=adv_col_names),ignore_index=True)
        
        pert=jsma_sample.numpy()-sample["numeric"].numpy()
        scaled_pert=scaler({'numeric':pert})
        pert_df=pert_df.append(pd.Series(scaled_pert.numpy()[0], index=field_names), ignore_index=True)
        # scaled_sample=scaler(sample)
        # jsma={'numeric':jsma_sample}
        # scaled_jsma=scaler(jsma)
        # draw_perturbation(scaled_sample, scaled_jsma, "../experiment/pert_vis/test.png", field_names)
    
    adv_df.to_csv("../experiment/adv_data/{}_{}.csv".format(dataset_name,set_name), index=False)
    pert_df.to_csv("../experiment/adv_data/{}_{}_pert.csv".format(dataset_name,set_name), index=False)
    # draw distributions of each attribute for all data
    axes = pert_df.hist(figsize=(50, 50))

    plt.savefig("../experiment/pert_vis/{}_{}_hist.png".format(dataset_name,set_name))


def tensor_to_numpy(x):
    return x.numpy()[0]
    
def draw_perturbation(ori, adv, output_file_name, field_names):
    """
    draws the perturbation for single sample

    Args:
        ori (eager tensor): original sample.
        adv (eager tensor): adversarial sample.
        output_file_name (string): path to output file.
        field_names (array): column names.

    Returns:
        None: draws bar plot at output_file_name.

    """
    
    
    ori=tensor_to_numpy(ori)
    adv=tensor_to_numpy(adv)

    diff = adv-ori

    min = np.minimum(ori, adv)
    y_pos = list(range(diff.shape[0]))
    colour_map = {-1: 'r', 1: 'g', 0: 'b'}
    
    f = plt.figure(figsize=(16, 9))
    plt.bar(y_pos, min)
    plt.bar(y_pos, np.abs(diff), color=[
            colour_map[i] for i in np.sign(diff)], bottom=min)
    plt.xticks(y_pos, field_names, rotation=30,
               horizontalalignment='right')
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.ylim(0, 1.1)
    f.savefig(output_file_name)