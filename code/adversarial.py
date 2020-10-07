import csv
import json
import sys
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as K
from numpy import genfromtxt
from tqdm import tqdm

import joblib
import matplotlib
import matplotlib.pyplot as plt
from input_utils import *
from input_utils import load_dataset
from sklearn import preprocessing

matplotlib.use('Agg')

def create_autoencoder_pattern(model, input_feature):
    loss_object = tf.keras.losses.MeanSquaredError()
    with tf.GradientTape() as tape:
        tape.watch(input_feature)
        prediction = model(input_feature)
        loss = loss_object(input_feature, prediction)

    # Get the gradients of the loss w.r.t to the input image.
    gradient = tape.gradient(loss, input_feature)
    # Get the sign of the gradients to create the perturbation
    signed_grad = tf.sign(gradient)
    return signed_grad, np.abs(gradient)

def adversarial_sample_ae(model, input_feature):

    loss_object = tf.keras.losses.MeanSquaredError()
    adv_x=input_feature
    count=0
    rmse=loss_object(adv_x, model(adv_x))

    while rmse>0.1 and count< 1000:
        sign, magnitude = create_autoencoder_pattern(model, tf.convert_to_tensor(input_feature))
        mask = adv_x == 0
        magnitude=np.ma.masked_array(magnitude, mask=mask)
        strongest_index=magnitude.argmax()
        adv_x[0][strongest_index] -= adv_x[0][strongest_index]*0.1*sign[0][strongest_index]
        count+=1
        rmse=loss_object(adv_x, model(adv_x))
        # if count>520:
        #     print(magnitude)
        # print("count: {}, rmse: {}".format(count,rmse))
    return adv_x

def find_theta(metadata, percent_theta, fixed=[]):
    """
    finding the theta used in adversarial perturbation, as different data types
    handle different different theta. For symbolic data theta is set to difference between each of
    its level (e.g. for boolean attribute theta is 1). For continuous data it is set to
    percent_theta. For discrete data it changes by percent_theta of the range rounded evenly to
    nearest integer value before scaling, if data_range is too small treat like symbolic data.
    If fixed is specified, the theta at that index is set to 0

    Args:
        metadata (dictionary): metadata for the dataset.
        percent_theta (float): how much continuous attribute change as percentage (0~1).
        fixed (int list): index of feature to be fixed

    Returns:
        array: the theta values for each feature.

    """

    data_range = np.array(metadata["col_max"]) - np.array(metadata["col_min"])

    # -1 so we dont include label field
    num_fields = len(metadata["field_names"]) - 1
    thetas = np.zeros((num_fields,))

    for index in range(num_fields):
        if data_range[index] == 0 or index in fixed:
            thetas[index] = 0
            continue
        type = metadata["dtypes"][index]
        # if continous theta is set to percent theta
        if type.startswith("float"):
            theta = percent_theta
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


def jsma_attack(input_sample, target, model, thetas, metadata, max_iter=100):
    """applies jacobian saliency map attack on input_sample

    Args:
        input_sample (tensor): the input to be modified.
        target (int): the target class index.
        model (nn network): the neural network to be attacked.
        thetas (array): how much perturbation to add in each step for each attribute.
        metadata (dict): metadata of dataset
        max_iter (int): maximum number of iterations to carry out.

    Returns:
        array, int, int: the modified adversarial sample, number of iterations taken, prediction outcome

    """
    input_shape = input_sample.shape[1]
    perturbations = np.zeros(input_shape)
    x_adv = input_sample
    # iterate through max_iteration
    for i in range(max_iter):

        # stop if we hit target
        if np.argmax(model.predict(x_adv, steps=1)) == target:
            # print("jsma found in:", i)
            break

        # find forward gradient for each output class
        gradients = generate_forward_derivative(
            x_adv, model, metadata["num_classes"])

        # filter out features that cannot be increased based on current value
        field = x_adv.numpy()

        # make sure it does not increase more than 1 or col_max

        inc_mask = (field <= 1 - thetas).astype(int)
        dec_mask = (field >= thetas).astype(int)

        # make sure it would not be less than 0 or col_min

        # check if any values in thetas are 0
        theta_mask = (thetas != 0).astype(int)
        inc_mask *= theta_mask
        dec_mask *= theta_mask

        # index in fixed cannot be changed
        # for i in fixed:
        #     inc_mask[0][i] = 0
        #     dec_mask[0][i] = 0

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

        x_adv = input_sample.numpy() + perturbations
        x_adv = tf.convert_to_tensor(x_adv)

    # print("input_sample:", input_sample)
    # print("perturbation:", perturbations)
    # print("x_adv:", x_adv)
    # print("prediction:", np.argmax(model.predict(x_adv, steps=1)))
    # print(i)
    prediction = np.argmax(model.predict(x_adv, steps=1))
    # draw_perturbation(input_sample.numpy()[0], x_adv.numpy()[0], "../visualizations/adversarial/adv_stack_jsma.png")

    return x_adv, i, prediction


def adversarial_generation(dataset_name, model_path, target_class, set_name, num_samples=100, theta=0.01, fixed=[], label_name="Label", alter=None):
    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)

    batch_size = 1
    # remove label field
    field_names = metadata["field_names"][:-1]
    num_classes = metadata["num_classes"]
    data = load_dataset(dataset_name, sets=[
                        set_name], label_name=label_name, batch_size=batch_size)[0]

    if alter is not None:
        data = data.unbatch().filter(lambda feature, label: label==alter).batch(batch_size)
    else:
        data = data.unbatch().filter(lambda feature, label: label !=
                                     target_class).batch(batch_size)
    packed_data = data.map(PackNumericFeatures(field_names))
    # take 1
    # sample, label = next(iter(packed_val_data))

    print("loading model")
    model = tf.keras.models.load_model(model_path)

    fixed_index = field_name_search(metadata["field_names"], fixed)
    print("finding theta")
    thetas = find_theta(metadata, theta, fixed_index)
    print("genrating attack label map")
    maps = genfromtxt(
        "../data/{}/maps/{}.csv".format(dataset_name, label_name), delimiter=',')

    print("creating dataframes")
    adv_col_names = metadata["field_names"] + ["Iterations", "Adv Label"]
    adv_df = pd.DataFrame(columns=adv_col_names)
    pert_df = pd.DataFrame(columns=field_names)
    min = np.array(metadata["col_min"][:-1])
    max = np.array(metadata["col_max"][:-1])

    scaler, unscaler = min_max_scaler_gen(min, max)

    print("starting generation")
    with tqdm(total=num_samples) as pbar:
        for sample, label in packed_data.take(num_samples):

            scaled_sample=scaler(sample)
            jsma_sample, num_iter, pred = jsma_attack(scaled_sample,
                                                      target_class, model, thetas, metadata, max_iter=200)
            unscaled_sample=unscaler(jsma_sample)
            row = unscaled_sample.numpy()
            row = np.append(row, [label.numpy()[0], num_iter, pred])
            adv_df = adv_df.append(
                pd.Series(row, index=adv_col_names), ignore_index=True)

            pert = unscaled_sample.numpy() - sample.numpy()
            pert_df = pert_df.append(
                pd.Series(pert[0], index=field_names), ignore_index=True)
            # scaled_sample=scaler(sample)
            # jsma={'numeric':jsma_sample}
            # scaled_jsma=scaler(jsma)
            # draw_perturbation(scaled_sample, scaled_jsma, "../experiment/pert_vis/test.png", field_names)
            pbar.update(1)
    print("finished all JSMA samples")
    adv_df.to_csv(
        "../experiment/adv_data/{}_{}.csv".format(dataset_name, set_name), index=False)
    pert_df.to_csv(
        "../experiment/adv_data/{}_{}_pert.csv".format(dataset_name, set_name), index=False)
    # draw distributions of each attribute for all data
    axes = pert_df.hist(figsize=(50, 50))

    plt.savefig(
        "../experiment/pert_vis/{}_{}_hist.png".format(dataset_name, set_name))


def tensor_to_numpy(x):
    return x.numpy()[0]


def field_name_search(field_names, search_strings):
    """
    helper function to quickly find feature index to fix. The function returns
    index of features that has elements of search_strings contained in feature name.
    e.g. if search_strings=["Flags"] it will return index of all features names
    containing flags. Multiple elements in search_strings returns the union of
    indexes.

    Args:
        field_names (string array): list of field names from metadata.
        search_strings (string array): list of string to search.

    Returns:
        list: a unique list of all feature indexes matched by search_strings.

    """
    results = []
    num_fields = len(field_names)
    for string in search_strings:
        matching = [s for s in range(num_fields) if string in field_names[s]]
        results += matching

    # for i in range(num_fields):
    #     print(field_names[i], i in results)

    return list(set(results))


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

    ori = tensor_to_numpy(ori)
    adv = tensor_to_numpy(adv)

    diff = adv - ori

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


if __name__ == '__main__':
    features=[[1.915256162464416079e+01,2.000000000000000568e+02,0.000000000000000000e+00,1.984792566408835057e+01,1.999999999998176179e+02,2.063461579382419586e-08,2.080399595160168147e+01,1.999723160279728518e+02,3.103383093635784462e+00,2.415809596558893645e+03,1.977538001844734197e+02,2.546373985730388085e+02,6.511922154797476651e+03,1.977945174402764223e+02,2.514785067714692559e+02,1.915256162464416079e+01,2.000000000000000568e+02,0.000000000000000000e+00,3.487542780011128343e+03,2.575064004198315502e+02,3.583560144424815380e-02,0.000000000000000000e+00,1.984792566408835057e+01,1.999999999998176179e+02,2.063461579382419586e-08,3.352545647791808733e+03,2.593038276318127373e+02,8.483370583813504284e-01,1.019959425246314879e+02,2.080399595160168147e+01,1.999723160279728518e+02,3.103383093635784462e+00,3.760847756728969671e+03,2.558071571633972781e+02,1.301849239806803205e+01,1.205036484995891666e-01,2.415809596558893645e+03,1.977538001844733913e+02,2.546373985730388085e+02,1.605154586397471348e+03,2.111798061313541268e+02,1.599875635644368410e+00,2.518453916215805406e-03,6.511922154797478470e+03,1.977945174402763655e+02,2.514785067714692559e+02,8.771465931251524353e+02,2.089524805195362660e+02,6.403647190349308183e+00,1.393007751134688055e-02,1.915256162464416079e+01,5.567468506146893148e-01,6.886576634559165910e+00,1.984792566408835057e+01,5.825298535871317185e-01,7.193484126539214429e+00,2.080399595160168147e+01,6.026334415900700403e-01,7.432413933648584603e+00,2.415809596558893645e+03,6.139165629257020869e-03,6.961251988960387438e-02,6.511922154797476651e+03,2.768781697154371665e-03,2.593594755024272377e-02,1.097351794748506038e+01,2.000000000000000000e+02,0.000000000000000000e+00,0.000000000000000000e+00,2.000000000000000000e+02,0.000000000000000000e+00,0.000000000000000000e+00,1.098409573897681213e+01,1.999999999999466809e+02,6.039044819772243500e-09,6.039044819772243500e-09,1.999999999999466809e+02,0.000000000000000000e+00,0.000000000000000000e+00,1.102911490972858388e+01,1.999915042540189631e+02,9.538266239105723798e-01,9.538266239105723798e-01,1.999915042540189631e+02,0.000000000000000000e+00,0.000000000000000000e+00,4.136041972997104494e+02,1.978534278620882674e+02,2.451308279904333176e+02,2.451308279904333176e+02,1.978534278620882674e+02,0.000000000000000000e+00,0.000000000000000000e+00,1.102199491582856353e+03,1.978611815970103578e+02,2.458266481364262290e+02,2.458266481364262290e+02,1.978611815970103578e+02,0.000000000000000000e+00,0.000000000000000000e+00]
]

    model=tf.keras.models.load_model("../models/surrogate_ae_video.h5")
    max_val=np.genfromtxt("max.csv",delimiter=",")
    min_val=np.genfromtxt("min.csv",delimiter=",")
    features=(features - min_val) / (max_val - min_val)
    features=features.astype(np.float32)

    print(features)
    loss_object = tf.keras.losses.MeanSquaredError()
    recon=model(features)
    print(recon)
    rmse=mean_squared_error(features, recon)

    print("original RMSE:", rmse)
    adv_x=adversarial_sample_ae(model, features)
    print(adv_x)
