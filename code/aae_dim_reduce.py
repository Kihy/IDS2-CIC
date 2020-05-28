import json
from datetime import datetime

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from input_utils import PackNumericFeatures, load_dataset, min_max_scaler_gen, read_maps
from sklearn import svm
from sklearn.metrics import accuracy_score
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Dense, Dropout, Embedding,
                                     Flatten, GaussianNoise, Input, Lambda,
                                     Layer, LeakyReLU, MaxPooling2D, Reshape,
                                     ZeroPadding2D, multiply)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model, to_categorical


@tf.function
def tracer(model, inputs):
    output = model(inputs)
    return output

def sample_prior(batch_size, distro, num_classes=0, latent_dim=0):
    if distro == "normal":
        return np.random.normal(size=(batch_size, latent_dim))
    if distro == "uniform":
        return np.random.uniform(size=(batch_size, latent_dim))
    if distro == "categorical":
        choices = np.random.choice(num_classes, batch_size)
        return np.eye(num_classes)[choices]


class WcLayer(Layer):
    def __init__(self, dimensions=3, distance_thresh=0.5, **kwargs):
        super(WcLayer, self).__init__(**kwargs)
        self.dimensions = dimensions
        self.distance_thresh=distance_thresh

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.dimensions),
                                 initializer="random_normal",
                                 trainable=True)

    def get_config(self):
        config = super(WcLayer, self).get_config()
        config.update({"dimensions": self.dimensions, "distance_thresh":self.distance_thresh})
        return config

    def call(self, inputs):

        cluster_head = tf.matmul(inputs, self.w)
        num_cluster_heads=self.w.shape[0]

        #calculate total number of pairs
        n=num_cluster_heads-1
        num_pairs=n*(n+1)/2
        #calculate pairwise distance between cluster heads
        distances = tfa.losses.metric_learning.pairwise_distance(self.w)
        #map threshold
        less_than_thresh= tf.boolean_mask(distances, tf.math.less(distances, tf.constant(self.distance_thresh)))
        #add all distance less than threshold. it is inversed so that closer distance is penalized more than further distances
        distance_loss=self.distance_thresh*num_pairs-(tf.math.reduce_sum(less_than_thresh)/2)
        self.add_loss(distance_loss)

        return cluster_head



def build_encoder(original_dim, latent_dim, intermediate_dim, num_classes, name="encoder"):
    input_layer = Input(shape=(original_dim,), name="encoder_input")

    x = Dense(intermediate_dim, name="encoder_dense1")(input_layer)
    x = LeakyReLU(alpha=0.2, name="encoder_act1")(x)
    x = Dense(intermediate_dim, name="encoder_dense2")(x)
    x = LeakyReLU(alpha=0.2, name="encoder_act2")(x)

    latent_repr = Dense(latent_dim, name="encoder_latent_out")(x)
    cat_out = Dense(num_classes, name="encoder_cat_out")(x)

    return Model(input_layer, [latent_repr, cat_out], name=name)


def build_discriminator(input_shape, name="discriminator"):
    disc_input = Input(shape=(input_shape, ), name="disc_input")
    x = Dense(24, name="disc_dense1")(disc_input)
    x = LeakyReLU(alpha=0.2, name="disc_act1")(x)
    x = Dense(12, name="disc_dense2")(x)
    x = LeakyReLU(alpha=0.2, name="disc_act2")(x)
    validity = Dense(1, activation="sigmoid", name="disc_out")(x)
    return Model(disc_input, validity, name=name)


def build_decoder(original_dim, latent_dim, intermediate_dim, num_classes, name="decoder"):
    repr_input = Input(shape=(latent_dim,), name="latent_input")
    x = Dense(intermediate_dim, name="decoder_dense1")(repr_input)
    x = LeakyReLU(alpha=0.2, name="decoder_act1")(x)
    x = Dense(intermediate_dim, name="decoder_dense2")(x)
    x = LeakyReLU(alpha=0.2, name="decoder_act2")(x)
    output = Dense(original_dim, activation="tanh", name="decoder_out")(x)

    return Model(repr_input, output, name=name)

def build_aae_dim_reduce(original_dim, intermediate_dim, latent_dim, num_classes,distance_thresh):
    encoder = build_encoder(original_dim=original_dim, latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim, num_classes=num_classes)

    decoder = build_decoder(original_dim=original_dim, latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim, num_classes=num_classes)

    latent_discriminator = build_discriminator(latent_dim, name="latent_disc")
    latent_discriminator.compile(loss='binary_crossentropy',
                                 optimizer='adam', metrics=['accuracy'])
    latent_discriminator.trainable = False

    cat_discriminator = build_discriminator(num_classes, name="cat_disc")
    cat_discriminator.compile(loss='binary_crossentropy',
                              optimizer='adam', metrics=['accuracy'])
    cat_discriminator.trainable = False

    inputs = Input(shape=(original_dim,), name="aae_input")
    latent_repr, cat_out = encoder(inputs)

    wc_layer=WcLayer(dimensions=latent_dim, distance_thresh=distance_thresh)
    cluster_head = wc_layer(cat_out)

    representation = tf.keras.layers.Add()([cluster_head, latent_repr])
    recon = decoder(representation)

    latent_validity = latent_discriminator(latent_repr)
    cat_validity = cat_discriminator(cat_out)

    aae = Model(inputs, [recon, latent_validity, cat_validity])


    return aae, encoder, decoder, latent_discriminator, cat_discriminator

def train_aae(configs):
    batch_size = configs["batch_size"]
    dataset_name = configs["dataset_name"]
    train, val = load_dataset(
        dataset_name, sets=["train","val"],
        label_name="category", batch_size=batch_size)
    epochs = configs["epochs"]
    latent_dim = configs["latent_dim"]
    intermediate_dim = configs["intermediate_dim"]
    use_cat_dist=configs["use_cat_dist"]
    weight = configs["reconstruction_weight"]
    loss_weights = [weight, (1 - weight) / 2, (1 - weight) / 2]
    distance_thresh=configs["distance_thresh"]

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

    scaler, _ = min_max_scaler_gen(datamin, datamax)
    #
    packed_train_data = train.map(
        PackNumericFeatures(field_names, num_classes, scaler=scaler))
    packed_val_data = val.map(PackNumericFeatures(
        field_names,  num_classes, scaler=scaler))

    # oop version does not provide a good graph trace so using functional instead
    aae, encoder, decoder, latent_discriminator, cat_discriminator = build_aae_dim_reduce(
        input_dim, intermediate_dim, latent_dim, num_classes,distance_thresh)

    aae.compile(loss=['mse', 'binary_crossentropy', 'binary_crossentropy'],
                loss_weights=loss_weights, optimizer='adam',
                metrics=[["mse"], ['acc'], ['acc']])

    valid = np.ones((batch_size, 1))
    invalid = np.zeros((batch_size, 1))
    step = 0

    pbar = tqdm(range(epochs), desc="epoch")

    for epoch in pbar:
        steps = metadata["num_train"] // batch_size
        step_pbar = tqdm(total=steps, desc="steps",leave=False, position=1)
        for feature, label in packed_train_data.take(steps):

            input_feature = feature['numeric']

            fake_latent, fake_cat = encoder(input_feature)
            # train latent discriminator
            latent_discriminator.trainable = True
            real_latent = sample_prior(
                batch_size, distro="uniform", latent_dim=latent_dim)

            d_loss_real = latent_discriminator.train_on_batch(
                real_latent, valid)
            d_loss_fake = latent_discriminator.train_on_batch(
                fake_latent, invalid)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            latent_discriminator.trainable = False

            # train cat discriminator
            cat_discriminator.trainable = True
            if use_cat_dist:
                real_cat = sample_prior(
                    batch_size, distro="categorical", num_classes=num_classes)
            else:
                real_cat = label
            cat_loss_real = cat_discriminator.train_on_batch(real_cat, valid)
            cat_loss_fake = cat_discriminator.train_on_batch(fake_cat, invalid)
            cat_loss = 0.5 * np.add(cat_loss_real, cat_loss_fake)

            cat_discriminator.trainable = False

            # train generator
            g_loss = aae.train_on_batch(
                input_feature, [input_feature, valid, valid])

            # record losses if not tuning
            with train_summary_writer.as_default():
                tf.summary.scalar('latent loss', d_loss[0], step=step)
                tf.summary.scalar('latent acc', d_loss[1], step=step)
                tf.summary.scalar('cat loss', cat_loss[0], step=step)
                tf.summary.scalar('cat acc', cat_loss[1], step=step)
                tf.summary.scalar('mse', g_loss[1], step=step)

            step += 1
            step_pbar.update(1)
        step_pbar.reset()
        postfix={"latent_acc": 100 * d_loss[1], "cat_acc":100 * cat_loss[1], "mse": g_loss[1]}
        pbar.set_postfix(postfix)


    # other wise trace aae with dummy value and save model
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
    prefix="{}_{}_{}".format(dataset_name, latent_dim, use_cat_dist)
    aae.save("../models/aae/{}_aae.h5".format(prefix))
    encoder.save("../models/aae/{}_encoder.h5".format(prefix))
    decoder.save("../models/aae/{}_decoder.h5".format(prefix))

def eval(configs):
    batch_size = configs["batch_size"]
    dataset_name = configs["dataset_name"]
    subset="test"
    # dont shuffle since its already shuffled in datareader
    test,meta = load_dataset(
        dataset_name, sets=[subset], include_meta=True,
        label_name="category", batch_size=batch_size, shuffle=False)[0]

    draw_scatter=configs["draw_scatter"]
    latent_dim=configs["latent_dim"]
    use_cat_dist=configs["use_cat_dist"]
    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)

    num_classes = metadata["num_classes"]
    field_names = metadata["field_names"][:-1]
    input_dim = len(field_names)

    datamin = np.array(metadata["col_min"][:-1])
    datamax = np.array(metadata["col_max"][:-1])

    scaler, unscaler = min_max_scaler_gen(datamin, datamax)

    # original, jsma, fgsm=main()
    packed_test_data = test.map(PackNumericFeatures(
        field_names, num_classes, scaler=scaler))

    custom_objects = {'WcLayer': WcLayer}
    prefix="{}_{}_{}".format(dataset_name, latent_dim, use_cat_dist)
    aae = tf.keras.models.load_model("../models/aae/{}_aae.h5".format(prefix), custom_objects=custom_objects)
    encoder = tf.keras.models.load_model("../models/aae/{}_encoder.h5".format(prefix))
    decoder = tf.keras.models.load_model("../models/aae/{}_decoder.h5".format(prefix))

    loss_func = tf.keras.losses.MeanSquaredError()

    steps = metadata["num_{}".format(subset)] // batch_size

    # visualize stuff
    latent_file = open(
        "../experiment/aae_vis/{}_style.tsv".format(prefix), "w")
    latent_label = open(
        "../experiment/aae_vis/{}_labels.tsv".format(prefix), "w")
    representation_file = open(
        "../experiment/aae_vis/{}_representation.tsv".format(prefix), "w")
    total_mse = 0
    correct_label = 0

    #read attack map for labelling
    attack_map=read_maps("../data/{}/maps/category.csv".format(dataset_name))
    attack_mapper=np.vectorize(lambda x: attack_map[x])

    #get wc layer
    wc_layer=aae.get_layer("wc_layer")
    wc_weights=wc_layer.weights[0].numpy()
    print("cluster_heads:", wc_weights)

    #write clusterheads to style and representation file
    np.savetxt(latent_file, wc_weights, delimiter="\t")
    np.savetxt(representation_file, wc_weights, delimiter="\t")
    #write header for meta file
    meta_col=["src_ip","dst_ip","idx", "fin_flag", "syn_flag" ,"rst_flag", "psh_flag","ack_flag","urg_flag","ece_flag","cwr_flag"]
    header=[['label','pred_label']+meta_col+["dim1","dim2","dim3"]]
    np.savetxt(latent_label, header,delimiter="\t",fmt="%s")


    ch_labels=np.core.defchararray.add("ch_", attack_map)
    ch_meta=np.stack((ch_labels,ch_labels,*list(np.zeros(ch_labels.shape) for i in range(len(meta_col))), wc_weights[:,0], wc_weights[:,1], wc_weights[:,2]),axis=1)
    np.savetxt(latent_label, ch_meta, delimiter="\t", fmt="%s")

    if draw_scatter:
        fig,ax=plt.subplots(2,1)
    p1=None
    p2=None
    for a, b in tf.data.Dataset.zip((packed_test_data, meta)).take(steps):
        features=a[0]
        labels=a[1]

        encoded, pred_label = encoder(features['numeric'].numpy())

        cluster_head = wc_layer(pred_label)
        representation = cluster_head+encoded
        decoded = decoder(representation)
        labels = np.argmax(labels, axis=1)
        pred_label = np.argmax(pred_label.numpy(), axis=1)
        correct_label += sum(labels == pred_label)


        if 6727 in b["idx"].numpy():
            index=np.where(b["idx"]==6727)
            if labels[index]==1:
                p1=features['numeric'].numpy()[index]
        if 2336 in b["idx"].numpy():
            index=np.where(b["idx"]==2336)
            if labels[index]==1:
                p2=features['numeric'].numpy()[index]

        # if p1 is not None and p2 is not None:
        #     print(p1-p2)

        total_mse += loss_func(decoded, features['numeric'].numpy())
        # encoded = aae.encoder(features['numeric'].numpy())
        # output_wrt_dim(vae., features['numeric'].numpy()[0], field_names)
        # forward_derviative(vae.decoder, encoded[0], ["dim{}".format(i) for i in range(latent_dim)])
        # eval_with_different_label(aae, features["numeric"].numpy(), labels)
        np.savetxt(latent_file, encoded, delimiter="\t")
        np.savetxt(representation_file, representation, delimiter="\t")

        label_arr=np.stack((attack_mapper(labels), attack_mapper(pred_label),*list(b[x].numpy() for x in b.keys()), encoded[:,0], encoded[:,1], encoded[:,2]), axis=1)

        np.savetxt(latent_label, label_arr, delimiter="\t", fmt="%s")
        if draw_scatter:
            scatter1=ax[0].scatter(encoded.numpy()[:, 0], encoded.numpy()[:, 1], c=labels, label=labels)
            scatter2=ax[1].scatter(representation.numpy()[:,0],representation.numpy()[:,1],c=labels, label=labels)



    print("average mse:", total_mse / steps)
    print("average label acc:", correct_label / 1024 / steps)
    if draw_scatter:
        #add cluster_head
        scatter1=ax[0].scatter(wc_weights[:,0],wc_weights[:,1], )
        scatter2=ax[1].scatter(wc_weights[:,0],wc_weights[:,1], )

        legend1 = ax[0].legend(*scatter1.legend_elements(),
                             loc="lower left", title="Classes")

        legend2 = ax[1].legend(*scatter2.legend_elements(),
                          loc="lower left", title="Classes")
        ax[0].add_artist(legend1)
        ax[1].add_artist(legend2)
        plt.tight_layout()
        plt.savefig('../experiment/aae_vis/{}_scatter.png'.format(prefix))
    latent_file.close()
    latent_label.close()
    representation_file.close()
    # keract_stuff(encoder, features['numeric'].numpy())
    # encoded = encoder(features['numeric'].numpy())
    # forward_derviative(decoder, encoded.numpy()[0], ["dim{}".format(i) for i in range(configs["latent_dim"])])
    # decoder_impact(decoder, [0.40617728,0.46284026,0.66842294])


if __name__ == '__main__':
    training_configs = {
        "batch_size": 1024,
        "dataset_name": "ku_httpflooding_both",
        "epochs": 30,
        "latent_dim": 3,
        "reconstruction_weight": 0.8,
        "intermediate_dim": 24,
        "use_cat_dist":False,
        "distance_thresh":0.1
    }
    eval_configs = {
        "batch_size": 1024,
        "dataset_name": "ku_httpflooding_both",
        "latent_dim": 3,
        "draw_scatter":False,
        "use_cat_dist":False
    }

    # train_aae(training_configs)
    eval(eval_configs)
