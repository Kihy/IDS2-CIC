from itertools import product
import os
from surrogate_model import eval_surrogate
from datetime import datetime
import pprint
from kitsune import *
from pso import *
from parse_with_kitsune import *
logging.getLogger('pyswarms').setLevel(logging.WARNING)


def run_one(configs):

    log_file = open(configs["log_file"], "w")

    log_file.write(pprint.pformat(configs))
    netstat_path = None
    # threshold=0.1 #sub video
    # threshold=2.266930857607199 # kitsune video
    # threshold=0.3
    # threshold=0.5445 # kitsune video ho

    # malicious_file="../ku_http_flooding/pcaps/[HTTP_Flooding]GoogleHome_thread_800_origin.pcap"
    # init_file="../ku_http_flooding/pcaps/[Normal]GoogleHome.pcap"
    # malicious_file="../kitsune_dataset/wiretap_malicious.pcapng"
    # init_file="../kitsune_dataset/wiretap_normal.pcapng"
    print(pprint.pformat(configs))
    #
    report = craft_adversary(configs["malicious_file"], configs["init_file"], configs["adv_pcap_file"],
                             configs["mal_pcap_out"], configs["decision_type"], configs["threshold"], meta_path=configs["meta_path"],
                             model_path=configs["model_path"], optimizer=configs["optimizer"], init_count=configs["init_file_len"],
                             mutate_prob=configs["mutate_prob"], netstat_path=netstat_path, base_offset=configs["base_offset"],
                             log_file=log_file, n_dims=configs["n_dims"], max_time_window=configs[
        "max_time_window"], max_adv_pkt=configs["max_adv_pkt"],
        max_craft_pkt=configs["max_craft_pkt"], max_pkt_size=configs["max_pkt_size"], adv_csv_file=configs["adv_csv_file"],
        animation_folder=configs["animation_folder"], iteration=configs["iter"])

    print("max_time_window", configs["max_time_window"])
    print("max_craft_pkt", configs["max_craft_pkt"])
    print("n_dims", configs["n_dims"])

    # evaluate
    pos_mal, pos_craft = eval(configs["adv_csv_file"], configs["eval_model_path"], threshold=configs["eval_threshold"], meta_file=configs["meta_path"],
                              ignore_index=configs["init_file_len"], out_image=configs["kitsune_graph_path"])

    report["pos_mal"] = pos_mal
    report["pos_craft"] = pos_craft
    if configs["decision_type"] == "autoencoder":
        eval_surrogate(configs["adv_csv_file"], configs["model_path"], threshold=configs["threshold"],
                       ignore_index=configs["init_file_len"], out_path=configs["autoencoder_graph_path"])

    if report["num_altered"] == 0:
        log_file.write(pprint.pformat(report))
    else:
        fmt_string = """
number of malicious packets altered\t{}\taverage time delayed\t{}\tpos craft\t{}
average_reduction_ratio\t{}\taverage packet size\t{}\tpos mal\t{}
average number of craft packets\t{}
num pkt seen\t{}
"""
        log_file.write(fmt_string.format(report["num_altered"], report["av_time_delay"], report["pos_craft"],
                                         report["average_reduction_ratio"], report["av_pkt_size"], report[
                                             "pos_mal"], report["av_num_craft"],
                                         report["num_seen"]))
        log_file.write(pprint.pformat(report))
    log_file.close()
    pprint.pprint(report)

    return report


def iterative_gen(max_iter, optimizer, decision_type, n_dims, attack_configs, min_iter=0):
    configs = {}

    configs["max_time_window"] = 1
    configs["max_craft_pkt"] = 5
    configs["decision_type"] = decision_type
    # configs["threshold"]=0.5445

    configs["n_dims"] = n_dims
    configs["optimizer"] = optimizer[0]
    configs["mutate_prob"] = optimizer[1]

    # configs["init_file"]="../kitsune_dataset/wiretap_normal_hostonly.pcapng"
    # configs["init_file"]="../experiment/traffic_shaping/init_pcap/wiretap.pcap"
    configs["init_file"] = "../experiment/traffic_shaping/init_pcap/google_home_normal.pcap"

    if configs["decision_type"] == "kitsune":
        configs["model_path"] = "../models/kitsune.pkl"
        configs["threshold"] = 0.5445
    elif configs["decision_type"] == "autoencoder":
        configs["model_path"] = "../models/surrogate_ae.h5"
        configs["threshold"] = 0.17

    configs["eval_model_path"] = "../models/kitsune.pkl"
    configs["eval_threshold"] = 0.5445
    # configs["init_file_len"]=81838
    configs["init_file_len"] = 14400
    configs["max_pkt_size"] = 1514

    # folder structure: experiment/traffic_shaping/{dataset}/["craft", "adv", "csv", "png", "anim", "meta","logs"]/{dt_t_c_d_o_m}
    base_folder = "../experiment/traffic_shaping/{}".format(
        attack_configs["name"])
    experiment_folder = "{}_{}_{}_{}_{}{}".format(
        configs["decision_type"], configs["max_time_window"], configs["max_craft_pkt"], configs["n_dims"], configs["optimizer"], configs["mutate_prob"])

    for i in ["craft", "adv", "csv", "png", "anim", "meta", "logs"]:
        if not os.path.exists(os.path.join(base_folder, i, experiment_folder)):
            os.makedirs(os.path.join(base_folder, i, experiment_folder))

    for i in range(min_iter, max_iter):
        print("iteration:", i)
        # mal_pcap file will be the next malicious_file
        configs["mal_pcap_out"] = base_folder + \
            "/craft/{}/craft_iter_{}.pcap".format(experiment_folder, i + 1)
        configs["adv_pcap_file"] = base_folder + \
            "/adv/{}/iter_{}.pcap".format(experiment_folder, i)
        configs["adv_csv_file"] = base_folder + \
            "/csv/{}/iter_{}.csv".format(experiment_folder, i)
        configs["animation_folder"] = base_folder + \
            "/anim/{}/iter_{}".format(experiment_folder, i)
        configs["meta_path"] = base_folder + \
            "/meta/{}/iter_{}.csv".format(experiment_folder, i)
        configs["log_file"] = base_folder + \
            "/logs/{}/iter_{}.txt".format(experiment_folder, i)
        configs["iter"] = i
        configs["kitsune_graph_path"] = base_folder + \
            "/png/{}/iter{}_kitsune_rmse.png".format(experiment_folder, i)
        configs["autoencoder_graph_path"] = base_folder + \
            "/png/{}/iter{}_ae_rmse.png".format(experiment_folder, i)

        # first iteration uses original malicious file, and limit packets to first 10
        if i == 0:
            # configs["malicious_file"]="../kitsune_dataset/wiretap_malicious_hostonly.pcapng"
            configs["malicious_file"] = attack_configs["original_mal_file"]
            configs["max_adv_pkt"] = 1000

            # base offset is the time between last normal packet and first malicious packet
            # configs["base_offset"] =-596.31862402
            configs["base_offset"] = attack_configs["base_time_offset"]
        else:
            configs["malicious_file"] = base_folder + \
                "/craft/{}/craft_iter_{}.pcap".format(experiment_folder, i)
            configs["max_adv_pkt"] = 1000
            configs["base_offset"] = 0

        report = run_one(configs)

        #
        if report["num_altered"] == 0 or report["craft_failed"] + report["num_failed"] == 0:
            break


if __name__ == '__main__':
    optimizers = [("pso", -1), ("pso", 0.5), ("de", "")]
    n_dims = [2, 3]
    decision_types = ["autoencoder", "kitsune"]
    # iterative_gen(10, ("pso", -1), "kitsune", 2 )

    scanning = {"name": "scanning",
                "original_mal_file": "../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only.pcap",
                "base_time_offset": -9422476.25}

    arp = {"name": "arp",
           "original_mal_file": "../ku_dataset/arp_attack_only.pcap",
           "base_time_offset": -9421981.61}

    flooding = {"name": "flooding",
                "original_mal_file": "../ku_dataset/flooding_attacker_only.pcap",
                "base_time_offset": -497696}

    datasets = [scanning,flooding]
    for dataset, o, n in product(datasets, optimizers, n_dims):
        iterative_gen(10, o, "kitsune", n, dataset)

    # iterative_gen(2, optimizers[2], decision_types[0], n_dims[0], scanning)
    # run_one(9)
