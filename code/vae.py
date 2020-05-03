import sys
from datetime import datetime

import numpy as np
import scipy
from numpy import genfromtxt

import joblib
import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
from input_utils import *
from input_utils import load_dataset
from matplotlib import cm
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Layer
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model

matplotlib.use('Agg')


class Sampling(Layer):
    """
    The reparameterization trick that samples from standard normal and reparameterize
    it into any normal distribution with mean and standard deviation.

    Note when training the variables are mean and log(var), thus to get std it is
    multiplied by exp(log(var)*0.5)
    """

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


def kullback_leibler_loss(z_mean, z_log_var, beta, out_stream=sys.stdout):
    """
    calculates kl divergence between standard normal and normal with z_mean
    and z_log_var as mean and log var.

    Note whether to use sum or mean is debatable, but both accepted. The difference is
    that mean calculates the average difference of the latent variables, and sum
    calculates total difference between latent variables.

    Args:
        z_mean (tensor): mean value of z distribution.
        z_log_var (tensor): log var of z distribution.
        beta (float): hyperparameter to weight kl divergence
    Returns:
        the kl divergence between the two distributions

    """
    kl_loss = - 0.5 * beta * \
        tf.math.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    # return xent_loss + kl_loss
    # tf.print("Average KL Div across Batch: ", tf.math.reduce_mean(kl_loss), output_stream=out_stream)
    # tf.summary.scalar('mean kl loss', data=tf.math.reduce_mean(kl_loss),step=1)

    return kl_loss


def weighted_reconstruction_loss(alpha, recon_metric="mean_squared_error", out_stream=sys.stdout):
    """
    returns a loss function with specified reconstruction metric and weight
    alpha. i.e. alpha * recon_metric(y_true, y_pred)

    Args:
        alpha (float): the weight to put on reconstruction
        recon_metric(string): name of reconstruction loss.

    Returns:
        weighted reconstruction loss function

    """

    def recon_loss(y_true, y_pred):

        loss = tf.keras.losses.get(recon_metric)
        reconstruction_loss = tf.math.multiply(alpha, loss(y_true, y_pred))
        # multiply by alpha to make reconstruction_loss more important meaning greater seperation of clusters in latent space
        # tf.print("Average Reconstruction Loss across Batch:", tf.math.reduce_mean(reconstruction_loss), output_stream=out_stream)
        # a=tf.summary.scalar('mean recon loss', data=tf.math.reduce_mean(reconstruction_loss),step=1)
        return reconstruction_loss
    return recon_loss


def reg_loss(model, batchsize, l1=0.01, l2=0.01, out_stream=sys.stdout):
    """
    Calculates the regularization loss for all dense layers in model
    the regularization parameter can be specified with l1 and l2

    Args:
        model (model): an nn model.
        batchsize (int): batchsize for training.
        l1 (float): weighting on l1. Defaults to 0.01.
        l2 (float): weighting on l2. Defaults to 0.01.

    Returns:
        total regularization loss for model, filled to match batchsize

    """
    reg = tf.keras.regularizers.l1_l2(l1=l1, l2=l2)
    sum = 0
    for layer in model.layers:
        if layer.name.startswith("dense"):
            sum += reg(layer.kernel)
    # tf.print("Total regularization Loss for {}:".format(model.name), sum, output_stream=out_stream)
    # tf.summary.scalar('regularization loss {}'.format(model.name), data=sum, step=1)
    return tf.fill((batch_size,), sum)


class Encoder(Model):
    """
    the encoder model used in variational autoencoder. this is purposely made
    to be a model rather than layer so we can use it directly for encoding. the
    encoder currently only has 1 intermediate layer and 1 output layer.

    Args:
        latent_dim (integer): latent dimension of the encoder output. Defaults to 2.
        intermediate_dim (integer): Description of parameter `intermediate_dim`. Defaults to 20.
        name (string): name of the model. Defaults to "encoder".
        **kwargs (type): any parameter that is used for model.

    Attributes:
        dense_proj (layer): projection layer from input to intermediate.
        dense_mean (layer): the mean values of the latent layer in the latent space.
        dense_log_var (layer): the log of variance of the latent layers in latent space.
        sampling (layer): the sampling layer used to sample data points.

    """

    def __init__(self, latent_dim=2, intermediate_dim=20, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        # regularizer=None
        self.dense_proj = Dense(intermediate_dim, activation='relu')
        self.dense_mean = Dense(latent_dim, activation='relu')
        self.dense_log_var = Dense(latent_dim, activation='relu')
        self.sampling = Sampling()

    def call(self, inputs):
        """runs the encoder model.

        Args:
            inputs (tensor): inputs to the network.

        Returns:
            type: the mean, log variance and z.

        """
        x = self.dense_proj(inputs)
        z_mean = self.dense_mean(x)
        z_log_var = self.dense_log_var(x)
        z = self.sampling((z_mean, z_log_var))
        return z_mean, z_log_var, z


class Decoder(Model):
    """
    the decoder model used in variational autoencoder. this is purposely made
    to be a model rather than layer so we can use it directly for encoding. the
    encoder currently only has 1 intermediate layer and 1 output layer.

    Args:
        original_dim (integer): original dimensions of the input data.
        intermediate_dim (integer): intermediate layer dimension. Defaults to 20.
        name (string): name of this model. Defaults to "decoder".
        **kwargs (type): any parameter that is used for model.

    Attributes:
        dense_proj (type): dense layer that projects latent dim to intermeidate.
        dense_output (type): dense layer that maps intermediate to output.

    """

    def __init__(self, original_dim, intermediate_dim=20, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.dense_proj = Dense(intermediate_dim, activation='relu')
        self.dense_output = Dense(original_dim, activation='sigmoid')

    def call(self, inputs):
        """
        runs the decoder model

        Args:
            inputs (tensor): inputs of the decoder network.

        Returns:
            type: the output of this model which is the reformed data.

        """
        x = self.dense_proj(inputs)
        return self.dense_output(x)


class VariationalAutoEncoder(Model):
    """
    the variational autoencoder model consisting of encoder and decoder

    Args:
        original_dim (integer): original dimension of input.
        latent_dim (integer): dimension of latent space. Defaults to 2.
        intermediate_dim (integer): dimension of intermediate layer. Defaults to 20.
        name (string): name of model. Defaults to "vae".
        **kwargs (type): other arguments.

    Attributes:
        encoder (model): encoder model.
        decoder (model): decoder model.
        original_dim

    """

    def __init__(self, original_dim, latent_dim=2, intermediate_dim=20, name="vae", beta=0.01, l1=0.1, l2=0.01, ** kwargs):
        super(VariationalAutoEncoder, self).__init__(name=name, **kwargs)
        self.original_dim = original_dim
        self.encoder = Encoder(
            latent_dim=latent_dim, intermediate_dim=intermediate_dim, name="encoder")
        self.decoder = Decoder(
            original_dim, intermediate_dim=intermediate_dim, name="decoder")
        self.beta = beta
        self.l1 = l1
        self.l2 = l2
        self.batch_size = batch_size
        self.step = 0

    def call(self, inputs):
        """connects encoder and decoder together and adds kl loss.

        Args:
            inputs (tensor): original input.

        Returns:
            type: reconstructed input.

        """
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        kl_loss = kullback_leibler_loss(z_mean, z_log_var, self.beta)
        # out_stream=open("../tensorboard_logs/loss_val")

        self.add_loss(kl_loss)

        encoder_reg = reg_loss(self.encoder, batch_size, self.l1, self.l2)
        decoder_reg = reg_loss(self.decoder, batch_size, self.l1, self.l2)
        self.add_loss(encoder_reg)
        self.add_loss(decoder_reg)

        # if tf.keras.backend.learning_phase() is True:
        tf.print(tf.math.reduce_mean(kl_loss), ",", tf.math.reduce_mean(encoder_reg), ",",
                 tf.math.reduce_mean(decoder_reg), output_stream="file://loss_logs/{}.txt".format(datetime.now().strftime("%Y%m%d-%H%M%S")))

        return reconstructed


def generate_latent_tsv(model, input_data, label, output_dir, batch_size=128):
    """
    generates tsv files that can be used in https://projector.tensorflow.org/.

    Args:
        model (model): the model used for prediction (i.e encoder)
        input_data (np.array): input data to be visualized.
        label (int array): label corresponding to each input data (not one hot encoded).
        output_dir (string): directory of the output file.
        batch_size (int): batchsize for prediction. Defaults to 128.

    Returns:
        None: saves files at output_dir
    """

    _, _, x_test_encoded = model(input_data)
    label = np.argmax(label.numpy(), axis=1)

    np.savetxt("{}/{}_points.tsv".format(output_dir, model.name),
               x_test_encoded, delimiter="\t")
    np.savetxt("{}/{}_labels.tsv".format(output_dir,
                                         model.name), label, delimiter="\n")


def visualize_latent_dimensions(model, input_data, index, output_dir):
    """
    Visualizes the latent dimension of sample with index. The dimensions are plotted
    on the same plot and over the same range [-3,3]. Useful when checking the
    weights of input.

    Args:
        model (encoder): the encoder part of vae.
        input_data (2d array): a batch of input data.
        index (int): the index to visualize latent dimension.
        output_dir (string): output directory.

    Returns:
        plot of dimensions at output_dir

    """
    x_mean, x_log_var, x_test_encoded = model.predict(input_data)
    sample_mean = x_mean[index]
    sample_log_var = x_log_var[index]
    f = plt.figure()
    x = np.linspace(-3, 3, 100)
    latent_dim = len(sample_mean)
    colour_map = cm.get_cmap('viridis', latent_dim)
    for i in range(latent_dim):
        std = np.exp(0.5 * sample_log_var[i])
        y = scipy.stats.norm.pdf(x, sample_mean[i], std)
        plt.plot(x, y, color=colour_map.colors[i])
    f.savefig("{}/{}_latent_dims_{}.png".format(output_dir, model.name, index))


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
    return min_max_scaler


def train_vae(dataset_name, batch_size, intermediate_dim, latent_dim, epochs, alpha, beta, l1, l2, recon_loss):
    train, val = load_dataset(
        dataset_name, sets=["train", "val"],
        label_name="Label", batch_size=batch_size)
    #
    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)

    logdir = "tensorboard_logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=logdir, histogram_freq=1)


    num_classes = metadata["num_classes"]
    field_names = metadata["field_names"][:-1]
    input_dim = len(field_names)

    datamin = np.array(metadata["col_min"][:-1])
    datamax = np.array(metadata["col_max"][:-1])

    scaler = min_max_scaler_gen(datamin, datamax)
    #
    packed_train_data = train.map(
        PackNumericFeatures(field_names, num_classes, vae=True, scaler=scaler))
    packed_val_data = train.map(PackNumericFeatures(
        field_names, num_classes, vae=True, scaler=scaler))


    vae = VariationalAutoEncoder(input_dim, intermediate_dim=intermediate_dim,
                                 latent_dim=latent_dim, beta=beta, l1=l1, l2=l2, name="vae_{}_{}_{}".format(dataset_name, alpha, recon_loss))

    recon_func = weighted_reconstruction_loss(alpha, recon_loss)
    vae.compile(optimizer='adam',
                loss=recon_func, metrics=[recon_func])

# limit the number of training and validation data validation data should be disabled if dataset is small
    vae.fit(packed_train_data,
            shuffle=True,
            epochs=epochs,
            steps_per_epoch=metadata["num_train"] // batch_size,
            # validation_data=packed_val_data,
            # validation_steps=metadata["num_val"]//batch_size,
            callbacks=[tensorboard_callback]
            )

    # # save network
    tf.keras.models.save_model(
        vae, "../models/vae/{}_{}_{}".format(dataset_name, alpha, recon_loss))
    print("saved model at models/vae/{}_{}_{}".format(dataset_name, alpha, recon_loss))


def forward_derviative(model, input, feature_names):
    """
    calculates the gradient of the input with respect to each output logits/neurons
    for any model. This is done via gradient tape and backpropagation.
    For models with multiple outputs such as encoder, only the first set of
    output(mean) is considered.

    Args:
        model (model): vae model.
        input (1d array): input sample array.
        feature_names (string list): feature names of each feature, should match
        input size

    Returns:
        None, the plot is saved somewhere.

    """
    input_sample = tf.convert_to_tensor([input])
    outputs = []
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(input_sample)
        output = model(input_sample)[0]

        for i in range(output.shape[1]):
            outputs.append(output[:, i])

    y_pos = list(range(len(feature_names)))
    plt.figure(figsize=(15, 10))
    for i in range(len(outputs)):
        g = tape.gradient(outputs[i], input_sample)

        plt.subplot(len(outputs), 1, i + 1)
        plt.bar(y_pos, g.numpy()[0])
    plt.xticks(y_pos, field_names, rotation=30,
               horizontalalignment='right')

    plt.savefig("../experiment/vae_vis/forward_derivative.png")


def output_wrt_dim(model, input, feature_names):
    """
    calculates the gradient of the output with respect to the input(latent dimensions) for
    the decoder. This is done via symmetric difference quotient, f(x+theta)-f(x-theta)/2theta.
    where theta is 1e-4. The gradient is draw for each latent dimension, and saved
    somewhere

    Args:
        model (model): the vae model.
        input (1d array): a single input array.
        feature_names (string list): feature names for each feature.

    Returns:
        None

    """
    theta = 1
    input_sample = [input]
    mean, log_var, x = model.encoder(input_sample)
    original = model.decoder(mean)
    print(mean)
    diffs = []
    decode_input = mean.numpy()
    for i in range(decode_input.shape[1]):
        x1 = np.array(decode_input)
        x1[:, i] += theta
        x2 = np.array(decode_input)
        x2[:, i] -= theta
        x1_out = model.decoder(x1)
        x2_out = model.decoder(x2)

        diff = (x1_out - x2_out) / (2 * theta)
        diffs.append(diff)

    y_pos = list(range(len(feature_names)))
    plt.figure(figsize=(15, 10))
    for i in range(len(diffs)):
        plt.subplot(3, 1, i + 1)
        plt.bar(y_pos, diffs[i].numpy()[0])
    plt.xticks(y_pos, field_names, rotation=30,
               horizontalalignment='right')

    plt.savefig("../experiment/vae_vis/output_grad_wrt_dim.png")


if __name__ == '__main__':
    dataset_name = "dos_pyflowmeter"
    batch_size = 128
    intermediate_dim = 24
    latent_dim = 3
    epochs = 500
    alpha = 1.
    beta = 0.1
    l1 = 0.01
    l2 = 0.01
    recon_loss = "binary_crossentropy"

    # train the model
    # train_vae(dataset_name, batch_size, intermediate_dim,
    #           latent_dim, epochs, alpha, beta, l1, l2, recon_loss)

    # loading the model
    val = load_dataset(dataset_name, sets=["val"],
                       label_name="Label", batch_size=batch_size)[0]

    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)

    num_classes = metadata["num_classes"]
    field_names = metadata["field_names"][:-1]
    input_dim = len(field_names)

    datamin = np.array(metadata["col_min"][:-1])
    datamax = np.array(metadata["col_max"][:-1])

    scaler = min_max_scaler_gen(datamin, datamax)

    # original, jsma, fgsm=main()
    packed_val_data = val.map(PackNumericFeatures(
        field_names, num_classes, scaler=scaler))
    vae = tf.keras.models.load_model(
        "../models/vae/{}_{}_{}.h5".format(dataset_name, alpha, recon_loss), compile=False)

    # visualize stuff
    for features, labels in packed_val_data.take(1):
        output_wrt_dim(vae, features['numeric'].numpy()[0], field_names)
        forward_derviative(vae.encoder, features['numeric'].numpy()[0], field_names)
        generate_latent_tsv(
            vae.encoder, features['numeric'].numpy(), labels, "../experiment/vae_vis/")
    #
    visualize_latent_dimensions(
        vae.encoder, features['numeric'].numpy(), 1, "../experiment/vae_vis/")
