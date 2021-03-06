import json
from datetime import datetime

import numpy as np

import keract
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from input_utils import PackNumericFeatures, load_dataset, min_max_scaler_gen
from sklearn import svm
from sklearn.metrics import accuracy_score, pairwise_distances
from tensorboard.plugins.hparams import api as hp
from tensorflow.keras.layers import (Activation, BatchNormalization,
                                     Concatenate, Dense, Dropout, Embedding,
                                     Flatten, GaussianNoise, Input, Lambda,
                                     Layer, LeakyReLU, MaxPooling2D, Reshape,
                                     ZeroPadding2D, multiply)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model, to_categorical
from vae import forward_derviative, generate_latent_tsva

# class Encoder(Model):
#
#     def __init__(self, latent_dim=2, intermediate_dim=20, deterministic=True, name="encoder", **kwargs):
#         super(Encoder, self).__init__(name=name, **kwargs)
#
#         # regularizer=None
#
#         # projection layers, using leaky relu as activation
#         self.dense_1 = Dense(intermediate_dim)
#         self.dense_2 = Dense(intermediate_dim)
#         self.leaky_relu_1 = LeakyReLU(0.2)
#         self.leaky_relu_2 = LeakyReLU(0.2)
#
#         # latent projections, with no linear activation
#         self.dense_latent = Dense(latent_dim)
#         self.dense_log_var = Dense(latent_dim)
#         self.sampling = Sampling()
#         self.deterministic = deterministic
#
#     @tf.function
#     def call(self, inputs):
#         """runs the encoder model.
#
#         Args:
#             inputs (tensor): inputs to the network.
#
#         Returns:
#             type: latent tensor z.
#
#         """
#         x = self.dense_1(inputs)
#         x = self.leaky_relu_1(x)
#         x = self.dense_2(x)
#         x = self.leaky_relu_2(x)
#
#         if self.deterministic:
#             z = self.dense_latent(x)
#         else:
#             z_mean = self.dense_latent(x)
#             z_log_var = self.dense_log_var(x)
#             z = self.sampling((z_mean, z_log_var))
#         return z
#
#
# class Decoder(Model):
#     def __init__(self, original_dim, intermediate_dim=20, supervised=False, name="decoder", **kwargs):
#         super(Decoder, self).__init__(name=name, **kwargs)
#         # projection layers, using leaky relu as activation
#         self.dense_1 = Dense(intermediate_dim)
#         self.dense_2 = Dense(intermediate_dim)
#         self.leaky_relu_1 = LeakyReLU(0.2)
#         self.leaky_relu_2 = LeakyReLU(0.2)
#
#         self.dense_output = Dense(original_dim, activation='tanh')
#         self.supervised = supervised
#         if supervised:
#             self.concat = Concatenate()
#
#     @tf.function
#     def call(self, inputs):
#         """
#         runs the decoder model
#
#         Args:
#             inputs (tensor): inputs of the decoder network.
#
#         Returns:
#             type: the output of this model which is the reformed data.
#
#         """
#         if self.supervised:
#             z, label = inputs
#             inputs = self.concat([z, label])
#
#         x = self.dense_1(inputs)
#         x = self.leaky_relu_1(x)
#         x = self.dense_2(x)
#         x = self.leaky_relu_2(x)
#
#         decoded = self.dense_output(x)
#         return decoded
#
#
# class Discriminator(Model):
#     def __init__(self, name="discriminator",  **kwargs):
#         super(Discriminator, self).__init__(name=name, **kwargs)
#         self.dense_1 = Dense(24)
#         self.leaky_relu_1 = LeakyReLU(0.2)
#         self.dense_2 = Dense(12)
#         self.leaky_relu_2 = LeakyReLU(0.2)
#         self.output_layer = Dense(1, activation='sigmoid')
#
#     @tf.function
#     def call(self, inputs):
#         # intermediate layers
#         x = self.dense_1(inputs)
#         x = self.leaky_relu_1(x)
#         x = self.dense_2(x)
#         x = self.leaky_relu_2(x)
#
#         # output
#         validity = self.output_layer(x)
#         return validity
#
#
# class AdversarialAutoEncoder(Model):
#
#     def __init__(self, original_dim, latent_dim=2, intermediate_dim=20, name="aae", deterministic=True, supervised=False, ** kwargs):
#         super(AdversarialAutoEncoder, self).__init__(name=name, **kwargs)
#         # build encoder and decoder
#         self.encoder = Encoder(
#             latent_dim=latent_dim, intermediate_dim=intermediate_dim, deterministic=deterministic, name="encoder")
#         self.decoder = Decoder(
#             original_dim, intermediate_dim=intermediate_dim, supervised=supervised, name="decoder")
#
#         # build and compile discriminator
#         self.discriminator = Discriminator()
#         self.discriminator.compile(
#             loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#         self.discriminator.trainable = False
#         self.supervised = supervised
#
#     @tf.function
#     def call(self, inputs):
#         """connects encoder and decoder together and adds kl loss.
#
#         Args:
#             inputs (tensor): original input.
#
#         Returns:
#             type: reconstructed input.
#
#         """
#         sample, label = inputs
#         z = self.encoder(sample)
#         if self.supervised:
#             reconstructed = self.decoder([z, label])
#         else:
#             reconstructed = self.decoder(z)
#         validity = self.discriminator(z)
#         return reconstructed, validity


class WcLayer(Layer):
    def __init__(self, dimensions=3, distance_thresh=0.5):
        super(WcLayer, self).__init__()
        self.dimensions = dimensions
        self.distance_thresh=distance_thresh

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.dimensions),
                                 initializer=tf.keras.initializers.Constant(3.),
                                 trainable=True)

    def call(self, inputs):

        cluster_head = tf.matmul(inputs, self.w)
        # distances = tfa.losses.metric_learning.pairwise_distance(cluster_head)
        # distance_loss=0
        # for i in range(distances.shape[0]):
        #     for j in range(i):
        #         if distances[i,j]>self.distance_thresh:
        #             distance_loss+=distances[i,j]
        # self.add_loss(distance_loss)

        return cluster_head


def build_encoder(input_shape=(21,), latent_dim=2, intermediate_dim=20, name="encoder", output_cat=False, num_classes=None):
    input_layer = Input(shape=input_shape, name="encoder_input")

    x = Dense(intermediate_dim, name="encoder_dense1")(input_layer)
    x = LeakyReLU(alpha=0.2, name="encoder_act1")(x)
    x = Dense(intermediate_dim, name="encoder_dense2")(x)
    x = LeakyReLU(alpha=0.2, name="encoder_act2")(x)

    latent_repr = Dense(latent_dim, name="encoder_latent_out")(x)

    if output_cat and num_classes is not None:
        cat_out = Dense(num_classes, name="encoder_cat_out")(x)
        model_output = [latent_repr, cat_out]
    else:
        model_output = latent_repr

    return Model(input_layer, model_output, name=name)


def build_discriminator(input_shape, name="discriminator"):
    disc_input = Input(shape=(input_shape, ), name="disc_input")
    x = Dense(24, name="disc_dense1")(disc_input)
    x = LeakyReLU(alpha=0.2, name="disc_act1")(x)
    x = Dense(12, name="disc_dense2")(x)
    x = LeakyReLU(alpha=0.2, name="disc_act2")(x)
    validity = Dense(1, activation="sigmoid", name="disc_out")(x)

    return Model(disc_input, validity, name=name)


def build_decoder(original_dim, latent_dim=2, intermediate_dim=20, labelled=False, num_classes=None, name="decoder"):
    if labelled and num_classes is not None:
        latent_input = Input(shape=(latent_dim,), name="latent_input")
        label_input = Input(shape=(num_classes,), name="label_input")
        inputs = Concatenate()([latent_input, label_input])
    else:
        inputs = Input(shape=(latent_dim,), name="latent_input")

    x = Dense(intermediate_dim, name="decoder_dense1")(inputs)
    x = LeakyReLU(alpha=0.2, name="decoder_act1")(x)
    x = Dense(intermediate_dim, name="decoder_dense2")(x)
    x = LeakyReLU(alpha=0.2, name="decoder_act2")(x)
    output = Dense(original_dim, activation="tanh", name="decoder_out")(x)

    if labelled:
        model_input = [latent_input, label_input]
    else:
        model_input = inputs
    return Model(model_input, output, name=name)


def build_aae(original_dim, intermediate_dim, latent_dim, supervised=False, num_classes=0):
    encoder = build_encoder(input_shape=(original_dim,), latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim)
    decoder = build_decoder(original_dim=original_dim, latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim, supervised=supervised, num_classes=num_classes)

    discriminator = build_discriminator(latent_dim)
    discriminator.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=['accuracy'])
    discriminator.trainable = False

    inputs = Input(shape=(original_dim,), name="aae_input")
    encoded = encoder(inputs)
    if supervised:
        label_input = Input(shape=(num_classes,), name="label_input")
        decoded = decoder([encoded, label_input])
    else:
        decoded = decoder(encoded)
    validity = discriminator(encoded)

    aae = Model(inputs, [decoded, validity])
    return aae, encoder, decoder, discriminator


def build_label_aae(original_dim, intermediate_dim, latent_dim, num_classes, representation=False):
    encoder = build_encoder(input_shape=(original_dim,), latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim, output_cat=True, num_classes=num_classes)

    decoder = build_decoder(original_dim=original_dim, latent_dim=latent_dim,
                            intermediate_dim=intermediate_dim, labelled=False, num_classes=num_classes)

    latent_discriminator = build_discriminator(latent_dim, name="latent_disc")
    latent_discriminator.compile(loss='binary_crossentropy',
                                 optimizer='adam', metrics=['accuracy'])
    latent_discriminator.trainable = False

    cat_discriminator = build_discriminator(num_classes, name="cat_disc")
    cat_discriminator.compile(loss='binary_crossentropy',
                              optimizer='adam', metrics=['accuracy'])
    cat_discriminator.trainable = False

    inputs = Input(shape=(original_dim,), name="aae_input")
    encoded, cat_out = encoder(inputs)

    if representation:

        wc_layer=WcLayer(dimensions=latent_dim, distance_thresh=0.4)
        cluster_head = wc_layer(cat_out)
        representation = tf.keras.layers.Add()([cluster_head, encoded])
    else:
        representation = [encoded, cat_out]

    decoded = decoder(representation)
    latent_validity = latent_discriminator(encoded)

    cat_validity = cat_discriminator(cat_out)

    aae = Model(inputs, [decoded, latent_validity, cat_validity])
    return aae, encoder, decoder, latent_discriminator, cat_discriminator


def sample_prior(batch_size, distro="normal", num_classes=0, latent_dim=0):
    if distro == "normal":
        return np.random.normal(size=(batch_size, latent_dim))
    if distro == "uniform":
        return np.random.uniform(size=(batch_size, latent_dim))
    if distro == "categorical":
        choices = np.random.choice(num_classes, batch_size)
        return np.eye(num_classes)[choices]


def hparam_tuning(configs):
    """used to tune hyperparameters with tensorboard's hparam module
    currently uses the following:

    LATENT_DIMS = hp.HParam('latent_dim', hp.Discrete([2, 3]))
    RECON_WEIGHTS = hp.HParam('reconstruction_weight',
                              hp.RealInterval(0.9, 0.99))
    INTERMEDIATE_DIM = hp.HParam('intermediate_dim', hp.Discrete([12, 24, 48]))

    metrics displayed are:
        - average reconstruction loss over sample
        - average accuracy of latent space clustering with svm


    Args:
        configs (dict): training configs.

    Returns:
        None.

    """
    # hyperparameters to tune
    logdir = "tensorboard_logs/aae/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    with tf.summary.create_file_writer(logdir + "/hparam_tuning").as_default():
        hp.hparams_config(
            hparams=[LATENT_DIMS, RECON_WEIGHTS, INTERMEDIATE_DIM],
            metrics=[hp.Metric("acc", display_name='accuracy score'), hp.Metric(
                "mse", display_name='reconstruction error')],
        )

    session_num = 0

    for intermediate_dim in INTERMEDIATE_DIM.domain.values:
        for recon_weights in np.arange(RECON_WEIGHTS.domain.min_value, RECON_WEIGHTS.domain.max_value, 0.05):
            for latent_dims in LATENT_DIMS.domain.values:
                hparams = {
                    INTERMEDIATE_DIM: intermediate_dim,
                    RECON_WEIGHTS: recon_weights,
                    LATENT_DIMS: latent_dims,
                }
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run(logdir + '/hparam_tuning/' + run_name, configs, hparams)
                session_num += 1


def run(run_dir, configs, hparams):
    """runs a session of hparam tuning

    Args:
        run_dir (string): directory of logs to be stored.
        configs (dict): normal training config.
        hparams (dict): dict containing hparams object.

    Returns:
        output saved to tensorboard

    """
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        mse, acc = train(configs, tuning=True, hparams=hparams)
        tf.summary.scalar("mse", mse, step=1)
        tf.summary.scalar("acc", acc, step=1)


def train_label_aae(configs, tuning=False, hparams=None):
    batch_size = configs["batch_size"]
    dataset_name = configs["dataset_name"]
    train, val = load_dataset(
        dataset_name, sets=configs["data_sets"],
        label_name="category", batch_size=batch_size)
    supervised = configs["supervised"]
    epochs = configs["epochs"]
    representation=configs["representation"]

    # if not hyperparameter tuning, set up tensorboard
    if not tuning:
        logdir = "tensorboard_logs/aae/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        train_summary_writer = tf.summary.create_file_writer(logdir)
        latent_dim = configs["latent_dim"]
        intermediate_dim = configs["intermediate_dim"]
        weight = configs["reconstruction_weight"]
    else:
        latent_dim = hparams[LATENT_DIMS]
        intermediate_dim = hparams[INTERMEDIATE_DIM]
        weight = hparams[RECON_WEIGHTS]
    loss_weights = [weight, (1 - weight) / 2, (1 - weight) / 2]

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
    aae, encoder, decoder, latent_discriminator, cat_discriminator = build_label_aae(
        input_dim, intermediate_dim, latent_dim, num_classes, representation=representation)

    aae.compile(loss=['mse', 'binary_crossentropy', 'binary_crossentropy'],
                loss_weights=loss_weights, optimizer='adam',
                metrics=["mse", 'acc', 'acc'])

    valid = np.ones((batch_size, 1))
    invalid = np.zeros((batch_size, 1))
    step = 0

    for epoch in range(epochs):
        steps = metadata["num_train"] // batch_size

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
            # real_cat = sample_prior(
            #     batch_size, distro="categorical", num_classes=num_classes)
            real_cat = label
            cat_loss_real = cat_discriminator.train_on_batch(real_cat, valid)
            cat_loss_fake = cat_discriminator.train_on_batch(fake_cat, invalid)
            cat_loss = 0.5 * np.add(cat_loss_real, cat_loss_fake)

            cat_discriminator.trainable = False

            # train generator
            g_loss = aae.train_on_batch(
                input_feature, [input_feature, valid, valid])

            # record losses if not tuning
            if not tuning:
                with train_summary_writer.as_default():
                    tf.summary.scalar('d loss', d_loss[0], step=step)
                    tf.summary.scalar('d acc', d_loss[1], step=step)
                    tf.summary.scalar('g loss', g_loss[0], step=step)
                    tf.summary.scalar('mse', g_loss[1], step=step)

            step += 1

        if epoch % 2 == 0:
            print("epoch:{} [D lat loss: {:.3f}, acc: {:.3f}%] [D cat loss: {:.3f}, acc: {:.3f}%] [G loss: {:.3f}, mse: {:.3f}, disc_loss: {:.3f}]" .format(
                epoch, d_loss[0], 100 * d_loss[1], cat_loss[0], 100 * cat_loss[1], g_loss[0], g_loss[1], g_loss[2]))

    # when tuning return the tuning metric
    if tuning:
        train_step = metadata["num_train"] // batch_size
        val_step = metadata["num_val"] // batch_size
        return h_measure(packed_train_data, packed_val_data, train_step, val_step, encoder, decoder)
    else:
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
        aae.save("../models/aae/aae.h5")
        encoder.save("../models/aae/encoder.h5")
        decoder.save("../models/aae/decoder.h5")


def train(configs, tuning=False, hparams=None):
    """
    training adversarial autoencoders

    Args:
        configs (dict): training configurations.
        tuning (boolean): whether the function should be ran under hyperparameter tuning mode. Defaults to False.
        hparams (dict): hparams object, has to be supplied if tuning is True. Defaults to None.

    Returns:
        None

    """
    batch_size = configs["batch_size"]
    dataset_name = configs["dataset_name"]
    train, val = load_dataset(
        dataset_name, sets=configs["data_sets"],
        label_name="category", batch_size=batch_size)
    supervised = configs["supervised"]
    epochs = configs["epochs"]

    # if not hyperparameter tuning, set up tensorboard
    if not tuning:
        logdir = "tensorboard_logs/aae/" + datetime.now().strftime("%Y%m%d-%H%M%S")
        train_summary_writer = tf.summary.create_file_writer(logdir)
        latent_dim = configs["latent_dim"]
        intermediate_dim = configs["intermediate_dim"]
        weight = configs["reconstruction_weight"]
    else:
        latent_dim = hparams[LATENT_DIMS]
        intermediate_dim = hparams[INTERMEDIATE_DIM]
        weight = hparams[RECON_WEIGHTS]
    loss_weights = [weight, 1 - weight]

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
    aae, encoder, decoder, discriminator = build_aae(
        input_dim, intermediate_dim, latent_dim, supervised=supervised)

    aae.compile(loss=['mse', 'binary_crossentropy'],
                loss_weights=loss_weights, optimizer='adam',
                metrics=["mse", 'acc'])

    valid = np.ones((batch_size, 1))
    invalid = np.zeros((batch_size, 1))
    step = 0

    for epoch in range(epochs):
        steps = metadata["num_train"] // batch_size

        for feature, label in packed_train_data.take(steps):

            input_feature = feature['numeric']

            # train discriminator
            discriminator.trainable = True

            fake = encoder(input_feature)
            real = sample_prior(latent_dim, batch_size, distro="uniform")

            d_loss_real = discriminator.train_on_batch(real, valid)
            d_loss_fake = discriminator.train_on_batch(fake, invalid)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            discriminator.trainable = False

            # train generator
            if supervised:
                g_loss = aae.train_on_batch(
                    [input_feature, label], [input_feature, valid])
            else:
                g_loss = aae.train_on_batch(
                    input_feature, [input_feature, valid])

            # record losses if not tuning
            if not tuning:
                with train_summary_writer.as_default():
                    tf.summary.scalar('d loss', d_loss[0], step=step)
                    tf.summary.scalar('d acc', d_loss[1], step=step)
                    tf.summary.scalar('g loss', g_loss[0], step=step)
                    tf.summary.scalar('mse', g_loss[1], step=step)

            step += 1

        if epoch % 2 == 0:
            print("epoch:{} [D loss: {:.3f}, acc: {:.3f}%] [G loss: {:.3f}, mse: {:.3f}, disc_loss: {:.3f}]" .format(
                epoch, d_loss[0], 100 * d_loss[1], g_loss[0], g_loss[1], g_loss[2]))

    # when tuning return the tuning metric
    if tuning:
        train_step = metadata["num_train"] // batch_size
        val_step = metadata["num_val"] // batch_size
        return h_measure(packed_train_data, packed_val_data, train_step, val_step, encoder, decoder)
    else:
        # other wise trace aae with dummy value and save model
        feature, label = list(packed_val_data.take(1).as_numpy_iterator())[0]
        feature = np.zeros((1, input_dim))
        label = np.zeros((num_classes))
        latent = np.zeros((1, latent_dim))

        # trace aae
        with train_summary_writer.as_default():
            tf.summary.trace_on(graph=True, profiler=True)
            if supervised:
                tracer(aae, [feature, label])
            else:
                tracer(aae, feature)

            tf.summary.trace_export(
                name="aae",
                step=0,
                profiler_outdir=logdir)

            # trace each component
            tf.summary.trace_on(graph=True, profiler=True)
            embeddings = tracer(encoder, feature)
            tf.summary.trace_export(
                name="encoder",
                step=0,
                profiler_outdir=logdir)

            tf.summary.trace_on(graph=True, profiler=True)
            if supervised:
                tracer(decoder, [latent, label])
            else:
                tracer(decoder, latent)
            tf.summary.trace_export(
                name="decoder",
                step=0,
                profiler_outdir=logdir)

            tf.summary.trace_on(graph=True, profiler=True)
            tracer(discriminator, latent)
            tf.summary.trace_export(
                name="discriminator",
                step=0,
                profiler_outdir=logdir)

        # tf.keras.models.save_model(aae, "../models/aae/test")
        aae.save("../models/aae/aae.h5")
        encoder.save("../models/aae/encoder.h5")
        decoder.save("../models/aae/decoder.h5")


@tf.function
def tracer(model, inputs):
    output = model(inputs)
    return output


def h_measure(train_data, val_data, train_step, val_step, encoder, decoder):
    """
    measure of quality of model used in hyperparameter tuning

    Args:
        train_data (csvData): training data.
        val_data (csvData): validation data.
        train_step (int): training data steps.
        val_step (int): validation data steps.
        encoder (model): encoder model.
        decoder (model): decoder model.

    Returns:
        mse: reconstruction error
        acc: accuracy of latent space clusters with svm

    """
    clf = svm.SVC()
    loss_func = tf.keras.losses.MeanSquaredError()
    total_loss = 0
    for features, labels in train_data.take(train_step):
        input_feature = features["numeric"]
        input_labels = np.argmax(labels, axis=1)
        encoded = encoder(input_feature)
        decoded = decoder(encoded)
        total_loss += loss_func(decoded, input_feature)
        clf.fit(encoded, input_labels)

    acc = 0
    for features, labels in val_data.take(val_step):
        input_feature = features["numeric"]
        input_labels = np.argmax(labels, axis=1)
        encoded = encoder(input_feature)
        y_pred = clf.predict(encoded)
        acc += accuracy_score(input_labels, y_pred)
    return total_loss / train_step, acc / val_step


def eval(configs):
    batch_size = configs["batch_size"]
    dataset_name = configs["dataset_name"]
    test = load_dataset(
        dataset_name, sets=configs["data_sets"],
        label_name="category", batch_size=batch_size)[0]
    supervised = configs["supervised"]

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

    aae = tf.keras.models.load_model("../models/aae/aae.h5")
    encoder = tf.keras.models.load_model("../models/aae/encoder.h5")
    decoder = tf.keras.models.load_model("../models/aae/decoder.h5")

    loss_func = tf.keras.losses.MeanSquaredError()

    steps = metadata["num_test"] // batch_size
    print("evaluating reconstruction error with mean squared error")

    f = plt.figure(figsize=(15, 10))

    # visualize stuff
    latent_file = open(
        "../experiment/aae_vis/{}_points.tsv".format(aae.name), "w")
    latent_label = open(
        "../experiment/aae_vis/{}_labels.tsv".format(aae.name), "w")
    pred_label_file = open(
        "../experiment/aae_vis/{}_pred_labels.tsv".format(aae.name), "w")
    total_mse = 0
    correct_label = 0
    for features, labels in packed_test_data.take(steps):
        encoded, pred_label = encoder(features['numeric'].numpy())
        decoded = decoder([encoded, pred_label])
        labels = np.argmax(labels, axis=1)
        pred_label = np.argmax(pred_label.numpy(), axis=1)
        correct_label += sum(labels == pred_label)
        total_mse += loss_func(decoded, features['numeric'].numpy())
        # encoded = aae.encoder(features['numeric'].numpy())
        # output_wrt_dim(vae., features['numeric'].numpy()[0], field_names)
        # forward_derviative(vae.decoder, encoded[0], ["dim{}".format(i) for i in range(latent_dim)])
        # eval_with_different_label(aae, features["numeric"].numpy(), labels)
        np.savetxt(latent_file, encoded, delimiter="\t")
        np.savetxt(latent_label, labels, delimiter="\n")
        np.savetxt(pred_label_file, pred_label, delimiter="\n")
        scatter = plt.scatter(encoded.numpy()[:, 0], encoded.numpy()[
                              :, 1], c=labels, label=labels)

    print("average mse:", total_mse / steps)
    print("average label acc:", correct_label / 1024 / steps)
    legend1 = plt.legend(*scatter.legend_elements(),
                         loc="lower left", title="Classes")
    f.add_artist(legend1)
    f.tight_layout()
    f.savefig('../experiment/aae_vis/scatter.png')
    latent_file.close()
    latent_label.close()
    pred_label_file.close()
    # keract_stuff(encoder, features['numeric'].numpy())
    # encoded = encoder(features['numeric'].numpy())
    # forward_derviative(decoder, encoded.numpy()[0], ["dim{}".format(i) for i in range(configs["latent_dim"])])
    # decoder_impact(decoder, [0.40617728,0.46284026,0.66842294])


def decoder_impact(decoder, encoded_repr):
    input_sample = tf.convert_to_tensor([encoded_repr])
    print(encoded_repr)
    outputs = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_sample)

        output = decoder(input_sample)

        for i in range(output.shape[1]):
            outputs.append(output[:, i])
    for i in outputs:
        g = tape.gradient(i, input_sample)
        print(g)

    print(decoder(input_sample))
    encoded_repr[1] += 0.1
    print(encoded_repr)
    print(decoder(np.array([encoded_repr])))


def eval_with_different_label(aae, inputs, labels):
    scalar_label = np.argmax(labels.numpy(), axis=1)
    for i in range(labels.shape[0]):
        # 1 normal 0 http flooding
        if scalar_label[i] == 0:
            encoded = aae.encoder(tf.expand_dims(inputs[i], axis=0))
            decoded = aae.decoder([encoded, [tf.one_hot(1, 2)]])
            original_decode = aae.decoder([encoded, [labels[i]]])
            print("original decode\n", original_decode.numpy())
            print("decoded with different label\n", decoded.numpy())
            print("original sample\n", inputs[i])
            return decoded, original_decode


def keract_stuff(model, x):
    input_sample = np.expand_dims(x[0], axis=0)
    activations = keract.get_activations(model, input_sample)
    fig, ax = plt.subplots(len(activations))
    i = 0
    for layer_name, act in activations.items():
        if act.ndim == 1:
            act = np.expand_dims(act, axis=0)
        # print(act)
        ax[i].imshow(act, cmap="Greys")
        ax[i].set_title(layer_name)
        i += 1
    fig.savefig("../experiment/aae_vis/activations.png")
    # print(activations)


LATENT_DIMS = hp.HParam('latent_dim', hp.Discrete([1, 2, 3]))
RECON_WEIGHTS = hp.HParam('reconstruction_weight',
                          hp.RealInterval(0.7, 1.0))
INTERMEDIATE_DIM = hp.HParam('intermediate_dim', hp.Discrete([12, 24, 48]))

if __name__ == '__main__':
    tuning_configs = {
        "batch_size": 1024,
        "dataset_name": "ku_google_home",
        "data_sets": ["train", "val"],
        "supervised": False,
        "epochs": 30
    }
    training_configs = {
        "batch_size": 1024,
        "dataset_name": "ku_httpflooding",
        "data_sets": ["train", "val"],
        "supervised": False,
        "epochs": 30,
        "latent_dim": 3,
        "reconstruction_weight": 0.9,
        "intermediate_dim": 24,
        "representation":False
    }
    eval_configs = {
        "batch_size": 1024,
        "dataset_name": "ku_httpflooding",
        "data_sets": ["test"],
        "supervised": False,
        "latent_dim": 3
    }

    # hparam_tuning(configs)
    # train(training_configs)
    # eval(eval_configs)
    train_label_aae(training_configs)
