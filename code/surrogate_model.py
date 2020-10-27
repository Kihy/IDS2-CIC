import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from scipy.stats import norm

class AnomalyDetector(Model):
    def __init__(self):
        super(AnomalyDetector, self).__init__()
        self.encoder = tf.keras.Sequential([
          layers.Dense(32, activation="relu"),
          layers.Dense(8, activation="relu"),
          layers.Dense(2, activation="relu")])

        self.decoder = tf.keras.Sequential([
          layers.Dense(4, activation="relu"),
          layers.Dense(32, activation="relu"),
          layers.Dense(100, activation="sigmoid")])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def train():
    path="../ku_http_flooding/kitsune_features/[Normal]GoogleHome.csv"
    dataframe = pd.read_csv(path,header=0)

    raw_data=dataframe.values

    # The last element contains the labels
    labels = raw_data[:, -1]

    # The other data points are the electrocadriogram data
    data = raw_data[:, 0:-1]


    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, random_state=21
    )



    train_data=train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)

    min_val = tf.reduce_min(train_data,axis=0)
    max_val = tf.reduce_max(train_data,axis=0)

    #save min and max
    np.savetxt("../models/surrogate_max.csv",max_val,delimiter=",")
    np.savetxt("../models/surrogate_min.csv",min_val,delimiter=",")

    train_data = (train_data - min_val) / (max_val - min_val)
    test_data = (test_data - min_val) / (max_val - min_val)

    train_data=np.nan_to_num(train_data)
    test_data=np.nan_to_num(test_data)

    autoencoder = AnomalyDetector()
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(train_data, train_data,
          epochs=100,
          batch_size=1024,
          validation_data=(test_data, test_data),
          shuffle=True)

    tf.saved_model.save(autoencoder, "../models/surrogate_ae.h5")

def eval_surrogate(path, model_path, threshold=None, out_path=None, ignore_index=0):
    autoencoder=tf.keras.models.load_model(model_path)
    # path="../ku_http_flooding/kitsune_features/[HTTP_Flooding]GoogleHome_thread_800_origin.csv"
    # path="../kitsune_dataset/wiretap_malicious_hostonly.csv"
    # path="../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only.csv"
    # path="../experiment/traffic_shaping/scanning/autoencoder_1_10_3_pso0.5/csv/iter_0.csv"

    dataframe = pd.read_csv(path,header=0)

    raw_data=dataframe.values[ignore_index:]


    if raw_data.shape[1]==101:
        # The last element contains the labels
        labels = raw_data[:, -1]

        # The other data points are the electrocadriogram data
        data = raw_data[:, 0:-1]
    else:
        data=raw_data

    max_val=np.genfromtxt("../models/surrogate_max.csv",delimiter=",")
    min_val=np.genfromtxt("../models/surrogate_min.csv",delimiter=",")


    data = (data - min_val) / (max_val - min_val+1e-6)


    if out_path==None:
        out_image=path[:-4]+"_ae_rmse.png"
    else:
        out_image=out_path

    counter=0
    input_file=open(path, "r")
    input_file.readline()
    rmse_array=np.array([])

    chunks=data.shape[0]//1024
    tbar=tqdm(total=chunks)
    for fv in np.array_split(data, chunks):
        # fv=chunk

        # fv=np.array([fv[:-1]], dtype="float")
        #normalize

        # fv = (fv - min_val) / (max_val - min_val)
        fv=fv.astype(np.float32)
        # print(fv.shape)
        reconstructions = autoencoder.predict(fv)
        train_loss = tf.keras.losses.mse(reconstructions, fv)

        # print(train_loss)
        rmse_array=np.concatenate((rmse_array,train_loss))
        counter+=fv.shape[0]
        tbar.update(1)

        # feature_vector=input_file.readline()

    if threshold==None:
    # benignSample = np.log(rmse_array)
    # mean=np.mean(benignSample)
    # std= np.std(benignSample)
    # threshold=np.exp(mean+3*std)
    # print("mean {}, std {}, threshold {}".format(mean, std, threshold))
        threshold=max(rmse_array)
        print(threshold)
    # threshold = 0.12
    # mean -3.496385573778823, std 0.5348295180179533

    else:
        print(np.where(rmse_array>threshold))
        num_over=(rmse_array>threshold).sum()




    max_index=np.argmax(rmse_array)
    max_rmse=rmse_array[max_index]
    plt.figure(figsize=(10,5))
    plt.scatter(range(counter),rmse_array,s=0.1)
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.yscale("log")
    plt.title("Anomaly Scores from imposter's Execution Phase")
    # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("packet index")
    plt.tight_layout()
    plt.savefig(out_image)
    print("plot path:", out_image)

if __name__ == '__main__':
    # train()
    # eval( "../ku_http_flooding/kitsune_features/[Normal]GoogleHome.csv","../models/surrogate_ae.h5")
    eval_surrogate("../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only.csv","../models/surrogate_ae.h5",threshold=0.17)
