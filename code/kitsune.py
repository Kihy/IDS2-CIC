from KitNET.KitNET import KitNET
import pickle
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt
from matplotlib import cm
from tqdm import tqdm

def train_normal():
    # File location
    path = "../kitsune_dataset/wiretap_normal_hostonly.csv" #the pcap, pcapng, or tsv file to process.
    packet_limit = np.Inf #the number of packets to process

    # KitNET params:
    maxAE = 10 #maximum size for any autoencoder in the ensemble layer
    FMgrace = 10000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = 740000 #the number of instances used to train the anomaly detector (ensemble itself)

    # Build Kitsune
    K = KitNET(100,maxAE,FMgrace,ADgrace,0.1,0.75)

    input_file=open(path, "r")
    input_file.readline()
    count=0


    tbar=tqdm()
    rmse=[]
    while True:
        feature_vector=input_file.readline()
        fv=feature_vector.rstrip().split(",")


        if len(fv)==101:
            fv=fv[:-1]
        fv=np.array(fv, dtype="float")
        res=K.process(fv)
        count+=1
        tbar.update(1)
        if count>FMgrace+ADgrace:
            break


    # save

    model_path="../models/kitsune_video_ho.pkl"
    with open(model_path, "wb") as of:
        pickle.dump(K, of)
    #
    # out_image=path[:-4]+"_kitsune_rmse.png"
    # plt.figure(figsize=(20,10))
    # plt.scatter(range(len(rmse)),rmse,s=0.1)
    # # plt.axhline(y=threshold, color='r', linestyle='-')
    # plt.yscale("log")
    # plt.savefig(out_image)

def eval(path, model_path ,threshold=None, ignore_index=-1, out_image=None, meta_file=None):
    # path = "../ku_http_flooding/kitsune_features/[HTTP_Flooding]GoogleHome_thread_800_origin.csv" #the pcap, pcapng, or tsv file to process.
    # path = "../ku_http_flooding/kitsune_features/[Normal]GoogleHome.csv" #the pcap, pcapng, or tsv file to process.
    # path = "../experiment/traffic_shaping/normal_800.csv" #the pcap, pcapng, or tsv file to process.
     #the pcap, pcapng, or tsv file to process.
    print("evaluting", path)
    print("kitsune model path ", model_path)
    with open(model_path, "rb") as m:
        kitsune=pickle.load(m)

    if out_image==None:
        out_image=path[:-4]+"_kitsune_rmse.png"

    if meta_file is not None:
        meta=open(meta_file, "r")
        meta.readline()
        meta_row=meta.readline()
        has_meta=True
        pos_craft=0
        pos_mal=0
    else:
        has_meta= False
        pos=0


    counter=0
    input_file=open(path, "r")
    input_file.readline()
    rmse_array=[]
    feature_vector=input_file.readline()
    while feature_vector is not '':
        if counter < ignore_index:
            feature_vector=input_file.readline()

            if meta_file is not None:
                meta_row=meta.readline()

            counter+=1
            continue

        fv=feature_vector.rstrip().split(",")

        if len(fv)==101:
            index=fv[-1]
            fv=fv[:-1]

        fv=np.array(fv, dtype="float")
        rmse=kitsune.process(fv)
        rmse_array.append(rmse)
        counter+=1
        if counter%1000==0:
            print(counter)

        feature_vector=input_file.readline()

        if rmse>threshold:
            if has_meta:
                comment=meta_row.rstrip().split(",")[-1]
                if comment=="craft":
                    pos_craft+=1
                elif comment=="malicious":
                    pos_mal+=1
                else:
                    print(meta_row)
                    raise Exception

            else:
                pos+=1

        if has_meta:
            meta_row=meta.readline()


    if threshold==None:
        benignSample = np.log(rmse_array)
        mean=np.mean(benignSample)
        std= np.std(benignSample)
        threshold=np.exp(mean+2*std)
        print("mean {}, std {}, 2std {}".format(mean, std, threshold))
        # threshold=2.266930857607199
    #mean -2.628361310825707, std 0.5458639089821489


    max_rmse= max(rmse_array)
    max_index=np.argmax(rmse_array)
    plt.figure(figsize=(10,5))
    plt.scatter(range(len(rmse_array)),rmse_array,s=0.1)
    # plt.annotate("{}, {}".format(max_rmse,max_index), (max_index, max_rmse))
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.yscale("log")
    plt.title("Anomaly Scores from Kitsune's Execution Phase")
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("packet index")
    plt.tight_layout()
    plt.savefig(out_image)
    print("plot path:", out_image)
    if has_meta:
        return pos_mal, pos_craft
    else:
        return pos

if __name__ == '__main__':
    # train_normal()
    # path = "../kitsune_dataset/wiretap_normal_hostonly.csv"
    # # path = "../kitsune_dataset/wiretap_normal.csv"
    # eval(path)
    # paths = ["../ku_dataset/flooding_attacker_only.csv"]
    paths=["../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only.csv"]
    # paths=["../experiment/traffic_shaping/scanning/autoencoder_1_10_3_pso0.5/csv/iter_0.csv"]
    # paths = ["../experiment/traffic_shaping/scanning/kitsune_1_10_3_pso0.5/csv/iter_0.csv"]
    model_path="../models/kitsune.pkl"
    for path in paths:
        pos=eval(path, model_path, threshold=0.54)
        print(pos)
