from pso import *
from parse_with_kitsune import *
logging.getLogger('pyswarms').setLevel(logging.WARNING)
from kitsune import *
import pprint
from datetime import datetime
import os

def run_one(configs):



    log_file=open("logs/{}/{}.txt".format(configs["optimizer"],datetime.now().strftime("%d-%m-%Y-%H:%M:%S.txt")),"w")

    log_file.write(pprint.pformat(configs))
    netstat_path=None
    # threshold=0.1 #sub video
    # threshold=2.266930857607199 # kitsune video
    # threshold=0.3
    # threshold=0.5445 # kitsune video ho

    # malicious_file="../ku_http_flooding/pcaps/[HTTP_Flooding]GoogleHome_thread_800_origin.pcap"
    # init_file="../ku_http_flooding/pcaps/[Normal]GoogleHome.pcap"
    # malicious_file="../kitsune_dataset/wiretap_malicious.pcapng"
    # init_file="../kitsune_dataset/wiretap_normal.pcapng"
    print(pprint.pformat(configs))

    report=craft_adversary(configs["malicious_file"],configs["init_file"],configs["adv_pcap_file"], configs["mal_pcap_out"], configs["decision_type"], configs["threshold"], model_path=configs["model_path"],optimizer=configs["optimizer"],init_count=configs["init_file_len"], mutate_prob=configs["mutate_prob"], netstat_path=netstat_path, base_offset=configs["base_offset"],log_file=log_file, n_dims=configs["n_dims"],max_time_window=configs["max_time_window"],max_adv_pkt=configs["max_adv_pkt"],max_craft_pkt=configs["max_craft_pkt"], max_pkt_size=configs["max_pkt_size"], adv_csv_file=configs["adv_csv_file"], animation_folder=configs["animation_folder"])
    # # parse_kitsune(configs["adv_pcap_file"],configs["adv_csv_file"], False, parse_type="scapy")


    print("max_time_window",configs["max_time_window"])
    print("max_craft_pkt", configs["max_craft_pkt"])
    print("n_dims",configs["n_dims"])

    log_file.write(pprint.pformat(report))
    log_file.close()
    pprint.pprint(report)
    return report


def iterative_gen(max_iter, min_iter=0):
    configs={}
    configs["max_time_window"]=1
    configs["max_craft_pkt"]=10
    configs["decision_type"]="autoencoder"
    # configs["threshold"]=0.5445
    configs["threshold"]=0.12
    configs["n_dims"]=3
    configs["optimizer"]="pso"
    configs["mutate_prob"]=0.5
    if configs["optimizer"]=="de":
        configs["mutate_prob"]=""


    # configs["init_file"]="../kitsune_dataset/wiretap_normal_hostonly.pcapng"
    # configs["init_file"]="../experiment/traffic_shaping/init_pcap/wiretap.pcap"
    configs["init_file"]="../experiment/traffic_shaping/init_pcap/google_home_normal.pcap"
    # configs["model_path"]="../models/kitsune_video_ho.pkl"
    configs["model_path"]="../models/surrogate_ae.h5"
    configs["eval_model_path"]="../models/kitsune.pkl"
    configs["eval_threshold"]=0.5445
    # configs["init_file_len"]=81838
    configs["init_file_len"]=14400
    configs["max_pkt_size"]=1514

    # folder structure: experiment/traffic_shaping/{dataset}/{dt}_{max_time_window}_{max_craft}_{opt}/{pcap,csv,png}
    folder="../experiment/traffic_shaping/scanning/{}_{}_{}_{}_{}{}".format(configs["decision_type"],configs["max_time_window"], configs["max_craft_pkt"],configs["n_dims"],configs["optimizer"],configs["mutate_prob"])
    if not os.path.exists(folder):
        os.makedirs(folder)
        for i in ["craft","adv","csv","png","anim"]:
            os.mkdir(folder+"/"+i)

    for i in range(min_iter, max_iter):
        print("iteration:", i)
        # mal_pcap file will be the next malicious_file
        configs["mal_pcap_out"]=folder+"/craft/craft_iter_{}.pcap".format(i+1)
        configs["adv_pcap_file"]=folder+"/adv/iter_{}.pcap".format(i)
        configs["adv_csv_file"]=folder+"/csv/iter_{}.csv".format(i)
        configs["animation_folder"]=folder+"/anim/iter_{}".format(i)

        # first iteration uses original malicious file, and limit packets to first 10
        if i==0:
            # configs["malicious_file"]="../kitsune_dataset/wiretap_malicious_hostonly.pcapng"
            configs["malicious_file"]="../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only.pcap"
            configs["max_adv_pkt"]=100

            # base offset is the time between last normal packet and first malicious packet
            # configs["base_offset"] =-596.31862402
            configs["base_offset"] = -9422476.25
        else:
            configs["malicious_file"]=folder+"/craft/craft_iter_{}.pcap".format(i)
            configs["max_adv_pkt"]=100
            configs["base_offset"]=0

        report=run_one(configs)

        # parse_kitsune(configs["adv_pcap_file"],configs["adv_csv_file"], False, parse_type="scapy")
        # evaluate
        eval(configs["adv_csv_file"], configs["eval_model_path"],threshold=configs["eval_threshold"], ignore_index=configs["init_file_len"], out_image=folder+"/png/iter{}_kitsune_rmse.png".format(i))

        if report["num_failed"]+report["craft_failed"]==0:
            break
if __name__ == '__main__':
    iterative_gen(10)
    # run_one(9)
