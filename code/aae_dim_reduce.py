import json
from datetime import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.constraints import unit_norm
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Dense, Dropout, Embedding,
                                     Flatten, GaussianNoise, Input, Lambda,
                                     Layer, LeakyReLU, MaxPooling2D, Reshape,
                                     ZeroPadding2D, multiply)
from tensorflow.keras.metrics import Accuracy, BinaryAccuracy, MeanSquaredError
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model, to_categorical
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow_addons as tfa
from input_utils import (PackNumericFeatures, load_dataset, min_max_scaler_gen,
                         read_maps)
from sklearn import svm
from sklearn.metrics import accuracy_score
from vae import forward_derviative
from vis_utils import *

@tf.function
def tracer(model, inputs):
    output = model(inputs)
    return output


def sample_prior(batch_size, distro, num_classes=0, latent_dim=0):
    if distro == "normal":
        return np.random.normal(size=(batch_size, latent_dim))
    if distro == "uniform":
        return np.random.uniform(-3, 3, size=(batch_size, latent_dim))
    if distro == "categorical":
        choices = np.random.choice(num_classes, batch_size)
        return np.eye(num_classes)[choices]


class WcLayer(Layer):
    def __init__(self, dimensions=3, distance_thresh=0.5, **kwargs):
        super(WcLayer, self).__init__(**kwargs)
        self.dimensions = dimensions
        self.distance_thresh = distance_thresh

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.dimensions),
                                 initializer="random_uniform",
                                 trainable=True)

    def get_config(self):
        config = super(WcLayer, self).get_config()
        config.update({"dimensions": self.dimensions,
                       "distance_thresh": self.distance_thresh})
        return config

    def call(self, inputs):

        cluster_head = tf.matmul(inputs, self.w)
        num_cluster_heads = self.w.shape[0]

        # calculate total number of pairs
        n = num_cluster_heads - 1
        num_pairs = n * (n + 1) / 2
        # calculate pairwise distance between cluster heads
        distances = tfa.losses.metric_learning.pairwise_distance(self.w)
        # map threshold
        less_than_thresh = tf.boolean_mask(distances, tf.math.less(
            distances, tf.constant(self.distance_thresh)))
        # add all distance less than threshold. it is inversed so that closer distance is penalized more than further distances
        distance_loss = self.distance_thresh * num_pairs - \
            (tf.math.reduce_sum(less_than_thresh) / 2)
        self.add_loss(distance_loss)

        return cluster_head


def build_encoder(original_dim, latent_dim, intermediate_dim, num_classes, kernel_constraint=None, activity_regularizer=None, name="encoder"):
    input_layer = Input(shape=(original_dim,), name="encoder_input")

    x = Dense(intermediate_dim, name="encoder_dense1",
              kernel_constraint=kernel_constraint, activity_regularizer=activity_regularizer)(input_layer)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2, name="encoder_act1")(x)
    x = Dense(intermediate_dim, name="encoder_dense2",
              kernel_constraint=kernel_constraint, activity_regularizer=activity_regularizer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2, name="encoder_act2")(x)

    latent_repr = Dense(latent_dim, activation="linear",
                        name="encoder_latent_out")(x)
    cat_out = Dense(num_classes, activation="sigmoid",
                    name="encoder_cat_out")(x)

    return Model(input_layer, [latent_repr, cat_out], name=name)


def build_discriminator(input_shape, name="discriminator", int_activation="relu", output_activation="sigmoid"):
    disc_input = Input(shape=(input_shape, ), name="disc_input")
    x = Dense(24, activation=int_activation,name="disc_dense1")(disc_input)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.2, name="disc_act1")(x)
    x = Dense(12, activation=int_activation, name="disc_dense2")(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=0.2, name="disc_act2")(x)
    validity = Dense(1, activation=output_activation, name="disc_out")(x)
    return Model(disc_input, validity, name=name)


def build_decoder(original_dim, latent_dim, intermediate_dim, num_classes, kernel_constraint=None, activity_regularizer=None, name="decoder"):
    repr_input = Input(shape=(latent_dim,), name="latent_input")
    x = Dense(intermediate_dim, name="decoder_dense1",
              kernel_constraint=kernel_constraint, activity_regularizer=activity_regularizer)(repr_input)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2, name="decoder_act1")(x)
    x = Dense(intermediate_dim, name="decoder_dense2",
              kernel_constraint=kernel_constraint, activity_regularizer=activity_regularizer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2, name="decoder_act2")(x)
    output = Dense(original_dim, activation="linear", name="decoder_out")(x)

    return Model(repr_input, output, name=name)


def build_aae_dim_reduce(original_dim, intermediate_dim, latent_dim, num_classes, distance_thresh, kernel_constraint, activity_regularizer, loss_weights):
    encoder = build_encoder(original_dim=original_dim, latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim, num_classes=num_classes, kernel_constraint=kernel_constraint, activity_regularizer=activity_regularizer)

    decoder = build_decoder(original_dim=original_dim, latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim, num_classes=num_classes, kernel_constraint=kernel_constraint, activity_regularizer=activity_regularizer)

    latent_discriminator = build_discriminator(
        latent_dim, name="latent_disc", int_activation="sigmoid",output_activation="sigmoid")
    latent_discriminator.compile(loss='binary_crossentropy',
                                 optimizer='adam', metrics=[BinaryAccuracy()])
    latent_discriminator.trainable = False

    cat_discriminator = build_discriminator(
        num_classes, name="cat_disc", int_activation="sigmoid", output_activation="sigmoid")
    cat_discriminator.compile(loss='binary_crossentropy',
                              optimizer='adam', metrics=[BinaryAccuracy()])
    cat_discriminator.trainable = False

    inputs = Input(shape=(original_dim,), name="aae_input")
    latent_repr, cat_out = encoder(inputs)

    wc_layer = WcLayer(dimensions=latent_dim, distance_thresh=distance_thresh)
    cluster_head = wc_layer(cat_out)

    representation = tf.keras.layers.Add()([cluster_head, latent_repr])
    recon = decoder(representation)

    latent_validity = latent_discriminator(latent_repr)
    cat_validity = cat_discriminator(cat_out)

    aae = Model(inputs, [recon, latent_validity, cat_validity, cat_out])

    return aae, encoder, decoder, latent_discriminator, cat_discriminator


def weighted_mse(datarange):

    datarange = tf.cast(tf.math.sqrt(datarange), dtype="float32")

    def mse(y_true, y_pred):
        loss = tf.math.square(y_true - y_pred)
        weighted_loss = datarange * loss
        return tf.reduce_mean(weighted_loss, axis=-1)
    return mse


def train_aae(configs):
    batch_size = configs["batch_size"]
    dataset_name = configs["dataset_name"]
    filter = configs["filter"]

    train, val = load_dataset(
        dataset_name, sets=["train", "val"], shuffle=False,
        label_name="category", batch_size=batch_size, filter=filter)

    epochs = configs["epochs"]
    latent_dim = configs["latent_dim"]
    intermediate_dim = configs["intermediate_dim"]
    use_clf_label = configs["use_clf_label"]
    weight = configs["reconstruction_weight"]
    loss_weights = [weight, (1 - weight) / 3,
                    (1 - weight) / 3, (1 - weight) / 3]
    distance_thresh = configs["distance_thresh"]
    classification_model = tf.keras.models.load_model(
        "../models/{}_{}".format(configs["clf_type"], dataset_name))
    # if not hyperparameter tuning, set up tensorboard
    logdir = "tensorboard_logs/aae/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    train_summary_writer = tf.summary.create_file_writer(logdir)

    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)

    num_classes = metadata["num_classes"]
    field_names = metadata["field_names"][:-1]
    input_dim = len(field_names)

    datamin = np.array(metadata["col_min"][:-1])
    datamax = np.array(metadata["col_max"][:-1])

    scaler, unscaler = min_max_scaler_gen(datamin, datamax)
    data_range = datamax - datamin
    #
    packed_train_data = train.map(
        PackNumericFeatures(field_names, num_classes, scaler=scaler))
    packed_val_data = val.map(PackNumericFeatures(
        field_names,  num_classes, scaler=scaler))

    # oop version does not provide a good graph trace so using functional instead
    aae, encoder, decoder, latent_discriminator, cat_discriminator = build_aae_dim_reduce(
        input_dim, intermediate_dim, latent_dim, num_classes, distance_thresh, None, None, loss_weights)

    # wmse=weighted_mse(data_range)
    aae.compile(loss=[tf.keras.losses.MeanSquaredError(), tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.BinaryCrossentropy(), tf.keras.losses.MeanSquaredError()],
                loss_weights=loss_weights, optimizer='adam',
                metrics=[[MeanSquaredError(name="latent_mse")], [BinaryAccuracy(name="latent_acc")], [BinaryAccuracy(name="label_acc")], [MeanSquaredError(name="label_mse")]])

    valid = np.ones((batch_size, 1))
    invalid = np.zeros((batch_size, 1))
    step = 0

    pbar = tqdm(range(epochs), desc="epoch")

    for epoch in pbar:
        steps = metadata["num_train"] // batch_size
        step_pbar = tqdm(total=steps, desc="steps", leave=False, position=1)
        for feature, label in packed_train_data.take(steps):
            fake_latent, fake_cat = encoder(feature)
            # train latent discriminator
            latent_discriminator.trainable = True
            real_latent = sample_prior(
                batch_size, distro="normal", latent_dim=latent_dim)

            noise_real = np.random.normal(
                0, 0.5, size=(batch_size, latent_dim))

            d_loss_real = latent_discriminator.train_on_batch(
                real_latent + noise_real, valid)
            d_loss_fake = latent_discriminator.train_on_batch(
                fake_latent, invalid)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            latent_discriminator.trainable = False

            # train cat discriminator
            cat_discriminator.trainable = True
            if use_clf_label:
                real_cat = classification_model(feature)
                real_cat = np.argmax(real_cat, axis=-1)
                real_cat = np.eye(num_classes)[real_cat]
            else:
                real_cat = label

            cat_loss_real = cat_discriminator.train_on_batch(real_cat, valid)

            cat_loss_fake = cat_discriminator.train_on_batch(fake_cat, invalid)
            cat_loss = 0.5 * np.add(cat_loss_real, cat_loss_fake)

            cat_discriminator.trainable = False

            # train generator
            g_loss = aae.train_on_batch(
                feature, [feature, valid, valid, real_cat])

            # record losses if not tuning
            with train_summary_writer.as_default():
                tf.summary.scalar('latent loss', d_loss[0], step=step)
                tf.summary.scalar('latent acc', d_loss[1], step=step)
                tf.summary.scalar('cat loss', cat_loss[0], step=step)
                tf.summary.scalar('cat acc', cat_loss[1], step=step)
                for i in range(len(aae.metrics_names)):
                    tf.summary.scalar(
                        aae.metrics_names[i], g_loss[i], step=step)

            step += 1
            step_pbar.update(1)

        # record distribution after epoch
        style, label = encoder(feature)

        with train_summary_writer.as_default():
            tf.summary.histogram(
                'cat_disc_out', cat_discriminator(label), step=step)
            tf.summary.histogram(
                'lat_disc_out', latent_discriminator(style), step=step)
            tf.summary.histogram('style', style, step=step)
            tf.summary.histogram('label', label, step=step)
            tf.summary.histogram(
                'prior style', real_latent + noise_real, step=step)
            tf.summary.histogram('prior label', real_cat, step=step)

        step_pbar.reset()
        postfix = {"latent_acc": 100 *
                   d_loss[1], "cat_acc": 100 * cat_loss[1], "mse": g_loss[5]}
        pbar.set_postfix(postfix)

    # trace aae with dummy value and save model
    feature, label = list(packed_val_data.take(1).as_numpy_iterator())[0]
    feature = np.zeros((1, input_dim))
    label = np.zeros((num_classes))
    latent = np.zeros((1, latent_dim))

    # trace aae
    with train_summary_writer.as_default():
        tf.summary.trace_on(graph=True, profiler=True)

        tracer(aae, feature)

        tf.summary.trace_export(
            name="aae",
            step=0,
            profiler_outdir=logdir)

    # tf.keras.models.save_model(aae, "../models/aae/test")
    prefix = "{}_{}_{}".format(dataset_name, latent_dim, use_clf_label)
    aae.save("../models/aae/{}_aae.h5".format(prefix))
    encoder.save("../models/aae/{}_encoder.h5".format(prefix))
    decoder.save("../models/aae/{}_decoder.h5".format(prefix))
    latent_discriminator.save("../models/aae/{}_lat_disc.h5".format(prefix))
    cat_discriminator.save("../models/aae/{}_cat_disc.h5".format(prefix))


def eval(configs):
    batch_size = configs["batch_size"]
    dataset_name = configs["dataset_name"]
    filter = configs["filter"]
    subset = "test"
    # dont shuffle since its already shuffled in datareader
    test, meta = load_dataset(
        dataset_name, sets=[subset], include_meta=True,
        label_name="category", batch_size=batch_size, filter=filter, shuffle=False)[0]

    draw_scatter = configs["draw_scatter"]
    latent_dim = configs["latent_dim"]
    use_clf_label = configs["use_clf_label"]
    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)

    num_classes = metadata["num_classes"]
    field_names = metadata["field_names"][:-1]
    input_dim = len(field_names)

    datamin = np.array(metadata["col_min"][:-1])
    datamax = np.array(metadata["col_max"][:-1])

    scaler, unscaler = min_max_scaler_gen(datamin, datamax)

    packed_test_data = test.map(PackNumericFeatures(
        field_names, num_classes, scaler=None))

    custom_objects = {'WcLayer': WcLayer}
    prefix = "{}_{}_{}".format(dataset_name, latent_dim, use_clf_label)
    aae = tf.keras.models.load_model(
        "../models/aae/{}_aae.h5".format(prefix), custom_objects=custom_objects)
    encoder = tf.keras.models.load_model(
        "../models/aae/{}_encoder.h5".format(prefix), custom_objects=custom_objects)
    decoder = tf.keras.models.load_model(
        "../models/aae/{}_decoder.h5".format(prefix), custom_objects=custom_objects)

    latent_discriminator = tf.keras.models.load_model(
        "../models/aae/{}_lat_disc.h5".format(prefix))
    cat_discriminator = tf.keras.models.load_model(
        "../models/aae/{}_cat_disc.h5".format(prefix))



    steps = metadata["num_{}".format(subset)] // batch_size


    correct_label = 0

    #
    # #get wc layer
    wc_layer = aae.get_layer("wc_layer")
    wc_weights = wc_layer.weights[0].numpy()
    print("cluster_heads:", wc_weights)
    #

    if draw_scatter:
        fig, ax = plt.subplots(2, 1, figsize=(10, 20))
        ax[0].set_title("style")
        ax[1].set_title("representations")
    else:
        # visualize stuff
        latent_file = open(
            "../experiment/aae_vis/{}_style.tsv".format(prefix), "w")
        latent_label = open(
            "../experiment/aae_vis/{}_labels.tsv".format(prefix), "w")
        representation_file = open(
            "../experiment/aae_vis/{}_representation.tsv".format(prefix), "w")

        # read attack map for labelling
        attack_map = read_maps(
            "../data/{}/maps/category.csv".format(dataset_name))
        attack_mapper = np.vectorize(lambda x: attack_map[x])

        # write clusterheads to style and representation file
        np.savetxt(latent_file, wc_weights, delimiter="\t")
        np.savetxt(representation_file, wc_weights, delimiter="\t")
        # write header for meta file
        meta_col = ["src_ip", "dst_ip", "timestamp","idx","fin_flag", "syn_flag",
                    "rst_flag", "psh_flag", "ack_flag", "urg_flag", "ece_flag", "cwr_flag"]
        header = [['label', 'pred_label', "clf_label"] +
                  meta_col + ["dim1", "dim2", "dim3"]]
        np.savetxt(latent_label, header, delimiter="\t", fmt="%s")

        ch_labels = np.core.defchararray.add("ch_", attack_map)
        ch_meta = np.stack((ch_labels, ch_labels, ch_labels, *list(np.zeros(ch_labels.shape) for i in range(
            len(meta_col))), wc_weights[:, 0], wc_weights[:, 1], wc_weights[:, 2]), axis=1)
        np.savetxt(latent_label, ch_meta, delimiter="\t", fmt="%s")


    classification_model = tf.keras.models.load_model(
        "../models/{}_{}".format(configs["clf_type"], dataset_name))
    correct_clf_label = 0
    clf_real = 0

    recon_mse = tf.keras.metrics.MeanSquaredError()
    label_mse = tf.keras.metrics.MeanSquaredError()
    normalized_mse=tf.keras.metrics.MeanSquaredError()
    for a, b in tf.data.Dataset.zip((packed_test_data, meta)).take(steps):
        features = a[0]
        labels = a[1]

        input_feature = scaler(features.numpy())

        style, pred_label = encoder(input_feature)
        cluster_head = wc_layer(pred_label)
        representation = cluster_head + style
        decoded = decoder(representation)
        reconstruction = unscaler(decoded.numpy())

        label_mse.update_state(labels, pred_label)
        labels = np.argmax(labels, axis=1)

        pred_label = np.argmax(pred_label.numpy(), axis=1)
        correct_label += sum(labels == pred_label)

        clf_label = classification_model(input_feature)
        clf_label = np.argmax(clf_label.numpy(), axis=1)
        correct_clf_label += sum(clf_label == pred_label)

        clf_real += sum(clf_label == labels)

        recon_mse.update_state(features.numpy(),reconstruction)
        normalized_mse.update_state(decoded, input_feature)
        # encoded = aae.encoder(features['numeric'].numpy())
        # output_wrt_dim(vae., features['numeric'].numpy()[0], field_names)
        # forward_derviative(vae.decoder, encoded[0], ["dim{}".format(i) for i in range(latent_dim)])
        # eval_with_different_label(aae, features["numeric"].numpy(), labels)
        if draw_scatter:
            ax[0].scatter(style.numpy()[:, 0], style.numpy()
                          [:, 1], c=labels, s=1)
            ax[1].scatter(representation.numpy()[:, 0],
                          representation.numpy()[:, 1], c=labels, s=1)
        else:
            np.savetxt(latent_file, style, delimiter="\t")
            np.savetxt(representation_file, representation, delimiter="\t")
            label_arr = np.stack((attack_mapper(labels), attack_mapper(pred_label), attack_mapper(clf_label), *list(
                b[x].numpy() for x in b.keys()), representation[:, 0], representation[:, 1], representation[:, 2]), axis=1)

            np.savetxt(latent_label, label_arr, delimiter="\t", fmt="%s")

    print("average mse:", recon_mse.result().numpy())
    print("average scaled mse:", normalized_mse.result().numpy())
    print("average label mse: ", label_mse.result().numpy())
    print("average real vs pred acc:", correct_label / batch_size / steps)
    print("average clf vs pred acc:", correct_clf_label / batch_size / steps)
    print("average clf vs real acc:", clf_real / batch_size / steps)

    encode_adv(
        "../experiment/adv_data/{}_{}.csv".format(dataset_name, "train"),encoder, scaler, wc_layer, field_names, num_classes, representation_file, latent_label,attack_mapper)

    if draw_scatter:
        # add cluster_head
        ax[0].scatter(wc_weights[:, 0], wc_weights[:, 1],
                      c=[x for x in range(num_classes)])
        ax[1].scatter(wc_weights[:, 0], wc_weights[:, 1],
                      c=[x for x in range(num_classes)])

        legend1 = ax[0].legend(
            *ax[0].collections[0].legend_elements(), loc="lower left", title="Classes")
        legend12 = ax[0].legend(
            *ax[0].collections[-1].legend_elements(), loc="upper left", title="Cluster Heads")

        legend2 = ax[1].legend(
            *ax[1].collections[0].legend_elements(), loc="lower left", title="Classes")
        legend22 = ax[1].legend(
            *ax[1].collections[-1].legend_elements(), loc="upper left", title="Cluster Heads")
        ax[0].add_artist(legend1)
        ax[0].add_artist(legend12)

        ax[1].add_artist(legend2)
        ax[1].add_artist(legend22)
        plt.tight_layout()
        plt.savefig('../experiment/aae_vis/{}_scatter.png'.format(prefix))
    else:
        latent_file.close()
        latent_label.close()
        representation_file.close()
    # keract_stuff(encoder, features['numeric'].numpy())
    # encoded = encoder(features['numeric'].numpy())
    # forward_derviative(decoder, encoded.numpy()[0], ["dim{}".format(i) for i in range(configs["latent_dim"])])
    # decoder_impact(decoder, [0.40617728,0.46284026,0.66842294])

    # decode_representation(decoder, [[0.850888729095459  ,	-0.7151963710784912,0.25660669803619385]
    # ,[0.962908148765564,-0.7504543662071228,	0.20932671427726746]], field_names,unscaler, "clusters")
    # decode_representation_idx(418180,52720,field_names, "../ku_httpflooding/[HTTP_Flooding]GoogleHome_thread_800.csv", "same_cluster")
    # vis_clusters("../metadata-edited.tsv",["cluster1","cluster2"],"../ku_httpflooding/[HTTP_Flooding]GoogleHome_thread_800.csv",field_names)

def encode_adv(adv_path, encoder, scaler, wc_layer, field_names, num_classes,representation_file, latent_label, attack_mapper):
    data = tf.data.experimental.make_csv_dataset(adv_path, batch_size=1000, select_columns=field_names+["Adv Label"], label_name="Adv Label")
    packed_data = data.map(PackNumericFeatures(field_names, num_classes ))

    for features,labels in packed_data.take(1):
        input_feature = scaler(features.numpy())

        style, pred_label = encoder(input_feature)
        cluster_head = wc_layer(pred_label)
        representation = cluster_head + style
        pred_label = np.argmax(pred_label.numpy(), axis=1)

        labels = np.argmax(labels.numpy(), axis=1)
        np.savetxt(representation_file, representation, delimiter="\t")
        np.savetxt(latent_label, np.stack((attack_mapper(labels), attack_mapper(pred_label)), axis=1), delimiter="\t", fmt="%s")



def decode_representation_idx(idx1, idx2, field_names, filename, suffix):
    df = pd.read_csv(filename, usecols=field_names+["idx"])
    df['protocol_type'] = df['protocol_type'].astype("category")
    df['protocol_type'] = df['protocol_type'].cat.codes

    pt1 = df.loc[df['idx'] == idx1].drop(columns=["idx"]).to_numpy()
    pt2 = df.loc[df['idx'] == idx2].drop(columns=["idx"]).to_numpy()

    diffs = pt1-pt2

    f = plt.figure(figsize=(20, 14))
    y_pos = list(range(len(field_names)))

    plt.barh(y_pos, diffs[0])
    plt.yticks(y_pos, field_names,
               horizontalalignment='right')

    for index, value in enumerate(diffs[0]):
        plt.text(value, index, "{:.3f}".format(value))
    plt.savefig("../experiment/aae_vis/point_diff_{}.png".format(suffix))


def decode_representation(decoder, representation, feature_names, unscaler,filename):
    decoded = decoder(np.array(representation))
    decoded = unscaler(decoded)
    diffs = decoded[0] - decoded[1]
    f = plt.figure(figsize=(20, 14))
    y_pos = list(range(len(feature_names)))


    plt.barh(y_pos, diffs.numpy())
    plt.yticks(y_pos, feature_names,
               horizontalalignment='right')

    for index, value in enumerate(diffs.numpy()):
        plt.text(value, index, "{:.3f}".format(value))
    plt.savefig("../experiment/aae_vis/point_diff_{}.png".format(filename))


if __name__ == '__main__':
    training_configs = {
        "batch_size": 2048,
        "dataset_name": "ku_flooding_800",
        "epochs": 100,
        "latent_dim": 3,
        "reconstruction_weight": 0.7,
        "intermediate_dim": 24,
        "use_clf_label": False,
        "distance_thresh": 0.5,
        "filter": None,
        "clf_type": "3layer"
    }
    eval_configs = {
        "batch_size": 4096,
        "dataset_name": "ku_flooding_800",
        "latent_dim": 3,
        "draw_scatter": False,
        "use_clf_label": False,
        "filter": None,
        "clf_type": "3layer"
    }

    # train_aae(training_configs)
    eval(eval_configs)
