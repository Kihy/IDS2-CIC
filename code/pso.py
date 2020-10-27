import numpy as np
import pyswarms as ps
from pprint import pformat
from pyswarms.utils.functions import single_obj as fx
from after_image.feature_extractor import *
from tqdm import tqdm
from scipy.spatial import distance_matrix
from scapy.all import *
from aae_dim_reduce import encode, WcLayer
from input_utils import min_max_scaler_gen
import json
import tensorflow as tf
from pyswarms.backend.topology import Random, Star, Ring
from topology.traffic import Traffic, create_swarm, plot_contour
from topology.differential_evolution import differential_evolution
from decimal import *
import pickle
from itertools import product
from KitNET.KitNET import KitNET
from adversarial import *
from sklearn.metrics import mean_squared_error
from matplotlib import animation, rc
import matplotlib.pyplot as plt
# from pyswarms.utils.plotters import (plot_cost_history, plot_contour, plot_surface)
from pyswarms.utils.plotters.formatters import Animator, Designer
from pyswarms.backend.handlers import BoundaryHandler
np.random.seed(0)

def plot_particle_position(pos_history, gbest, animation_path, limits, label):
    designer = Designer(
        limits=limits, label=label
    )

    animator = Animator(repeat=False, interval=200)

    rc('animation', html='html5')
    anim = plot_contour(pos_history=pos_history, designer=designer, animator=animator,
                        mark=gbest)
    anim.save(animation_path)
    plt.close('all')


np.set_printoptions(suppress=True,
                    formatter={'float_kind': '{:f}'.format}, linewidth=np.inf)
# np.set_printoptions(precision=18, floatmode="fixed")



def packet_gen(ori_pkt, traffic_vector):
    packet = ori_pkt.copy()
    packet.time = traffic_vector[-2]
    # print(traffic_vector)
    if packet.haslayer(TCP):
        packet[TCP].remove_payload()

        payload_size = int(traffic_vector[-1]) - len(packet)

        packet[TCP].add_payload(Raw(load="a" * payload_size))
        del packet[TCP].chksum
    # packet.show()

    elif packet.haslayer(UDP):
        packet[IP].remove_payload()

        packet[IP].add_payload(
            UDP(sport=int(traffic_vector[4]), dport=int(traffic_vector[6])))

        payload_size = int(traffic_vector[-1]) - len(packet)
        packet[UDP].add_payload(Raw(load="a" * payload_size))

    elif packet.haslayer(ARP):
        packet[IP].remove_payload()

        packet[IP].add_payload(
            ARP(psrc=int(traffic_vector[1]), pdst=int(traffic_vector[2])))

        payload_size = int(traffic_vector[-1]) - len(packet)

        packet[ARP].add_payload(Raw(load="a" * payload_size))
    # other packets
    else:
        packet[IP].remove_payload()
        payload_size = int(traffic_vector[-1]) - len(packet)
        packet.add_payload(Raw(load="a" * payload_size))
    # packet=Ether(bytes(packet))
    del packet[IP].len
    del packet[IP].chksum
    del packet.len

    # print(len(packet))
    return packet


def distance(feature, encoder, scaler, wc_layer):
    input_feature = scaler(feature)
    input_feature = np.expand_dims(input_feature, axis=0)
    style, representation, pred_label = encode(
        encoder, wc_layer, input_feature)
    # if pred_label[0][0]>0.2:
    #     print(pred_label)
    # normal label
    # if np.any(np.abs(representation.numpy()) > 3):
    #     print("repr:",representation)
    benign_label = [[0, 1]]
    mse = tf.keras.losses.binary_crossentropy(benign_label, pred_label)
    return mse.numpy()[0]


def rebuild(timestamp, num_craft, prev_pkt_time, max_pkt_size, min_pkt_len):
    """
    builds num_craft packets with random timestamp and framelen, all other fields are constant with malicious packet.
    return matrix has column as pair of crafted packets. only rebuilds 1 malicious packet
    """

    # get random time and length

    t = np.random.uniform(prev_pkt_time, timestamp, size=num_craft)

    t = np.sort(t)
    l = np.random.randint(min_pkt_len, max_pkt_size, size=num_craft)

    return np.stack((t, l))


def build_craft(timestamp, num_craft, prev_pkt_time, max_craft, n_dims, max_pkt_size=None, min_pkt_len=None):
    num_particles = timestamp.shape[0]

    # prev_pkt_time=np.full(timestamp.shape, prev_pkt_time)

    # evenly distribute craftpackets
    t = np.zeros((num_particles, max_craft))

    mask = np.full((num_particles, max_craft), True)

    for i in range(num_particles):
        mask[i][:num_craft[i]] = False
        if num_craft[i] == 0:
            continue

        if n_dims == 3:
            # indexing to remove start point
            t[i][:num_craft[i]] = np.linspace(
                prev_pkt_time, timestamp[i], num_craft[i] + 1, endpoint=False)[1:]
            t[i][:num_craft[i]] = np.around(t[i][:num_craft[i]], decimals=6)

        else:
            t = np.random.uniform(prev_pkt_time, np.expand_dims(
                timestamp, axis=1), size=(num_particles, max_craft))

    t = np.ma.masked_array(t, mask=mask)
    t = np.sort(t)

    # if min_pkt_len is int then it is 2d
    if n_dims == 3:
        l = np.tile(np.expand_dims(max_pkt_size, axis=1), (1, max_craft))
    else:
        l = np.random.randint(min_pkt_len, max_pkt_size,
                              size=(num_particles, max_craft))

    l = np.ma.masked_array(l, mask=mask)
    return np.concatenate((t.filled(0), l.filled(0)), axis=1)


def f(x, decision_func, adv_feature, decision_type, FE, max_time_window, prev_pkt_time, max_pkt_size, traffic_data, max_craft_pkt, min_pkt_len, db, verbose=False):
    """optimize function, x[:,0] is time and x[:,1] is number of crafted packets"""
    # go through each particle
    # FE.save_nstat_state()

    # f=open('tmp_nstat.txt', 'wb')
    # pickle.dump(obj=db,file=f)
    # f.close()

    # print("start f")
    # print(traffic_data)
    n_particles, n_dims = x.shape
    min_dists = np.zeros(n_particles)

    timestamps = x[:, 0]
    num_craft = np.trunc(x[:, 1]).astype(int)
    # num_craft=np.zeros(x[:,1].shape).astype(int)

    if n_dims == 3:
        pkt_sizes = np.rint(x[:, 2]).astype(int)
        craft_batch = build_craft(timestamps, num_craft, prev_pkt_time,
                                  max_craft_pkt, n_dims, max_pkt_size=pkt_sizes, min_pkt_len=pkt_sizes)

    if n_dims == 2:
        craft_batch = build_craft(timestamps, num_craft, prev_pkt_time,
                                  max_craft_pkt, n_dims, max_pkt_size=max_pkt_size, min_pkt_len=min_pkt_len)

    features = []
    for i in range(x.shape[0]):
        # dummy_db=pickle.load(f)
        # with open('tmp_nstat.txt', 'rb') as f:
        dummy_db = copy.deepcopy(db)

        # timestamp=x[:,0][i]
        # # print("random_t",timestamp)
        #
        # num_craft=np.rint(x[:,1][i]).astype(int)
        # if num_craft<0:
        #     print(num_craft)
        #     num_craft=0

        # crafted_features=rebuild(timestamp, num_craft, prev_pkt_time, max_pkt_size, min_pkt_len)

        # if crafted_features is None:
        #     print(x[:,1][i])
        # crafted features
        # print("-"*50)
        for j in range(max_craft_pkt):
            if j < num_craft[i]:
                # print((traffic_data[:-2]+craft_batch[i][:,j].tolist()))
                # print(craft_batch[i][:,j].tolist())
                time = craft_batch[i][j]
                size = craft_batch[i][j + max_craft_pkt]
                traffic_vector = traffic_data[:-2] + [time, size]

                f = FE.nstat.update_dummy_db(*traffic_vector, dummy_db)

                # f=FE.dummy_nstat.updateGetStats(*(traffic_data[:-2]+crafted_features[:,j].tolist()))

        # print(f)
        # raise Exception()

        # malicious packet
        traffic_data[-2] = timestamps[i]
        feature = FE.nstat.update_dummy_db(*(traffic_data), dummy_db)
        # feature=FE.dummy_nstat.updateGetStats(*(traffic_data))

        # add min distance
        # min_dists[i]=distance(feature, encoder, scaler,wc_layer)

        # feature = (feature - min_val) / (max_val - min_val)
        # feature=feature.astype(np.float32)
        # # print(fv.shape)
        # reconstructions = encoder.predict(np.expand_dims(feature,axis=0))
        # rmse = tf.keras.losses.mae(reconstructions, feature)
        # min_dists[i]=rmse[0]
        if decision_type == "kitsune":

            val = decision_func(feature)

            features.append(val)
        if decision_type == "autoencoder":
            features.append(feature)
        if decision_type == "adv":
            features.append(feature)
    if decision_type == "autoencoder":
        min_dists = decision_func(np.array(features))
    if decision_type == "adv":
        min_dists = tf.keras.losses.mean_squared_error(
            features, np.tile(adv_feature, (n_particles, 1)))
    if decision_type == "kitsune":
        min_dists = np.array(features)
    # min_dists[i]=decision_func(feature)
        # FE.roll_back()
    return min_dists, craft_batch


def craft_adversary(mal_pcap, init_pcap, adv_pcap, mal_pcap_out, decision_type, threshold, model_path, iteration, meta_path=None, optimizer=None, init_count=0, netstat_path=None, mutate_prob=-1, base_offset=0, log_file=None, n_dims=2, max_time_window=60, max_craft_pkt=5, num_benign_sample=100, max_pkt_size=655, max_adv_pkt=None, adv_csv_file=None, animation_folder=None):

    # output file with normal, craft and malicious pcap
    craft_pcap = PcapWriter(adv_pcap)
    log_file.write("original rmse\t original time\t mal file index\t craft file index\t adv pkt index\t best cost\t best pos\t aux\n")

    # output file with no init packets
    malicious_pcap = PcapWriter(mal_pcap_out)

    meta_file = open(meta_path, "w")
    meta_file.write("packet_index,time,comment\n")

    if decision_type == "adv":

        decision_func = get_decision_func("autoencoder", model_path)
        adv_func = get_decision_func("adv")
    else:
        decision_func = get_decision_func(decision_type, model_path)

    # whether to write features output directly to csv file
    write_to_csv = False

    if netstat_path is not None:
        with open(netstat_path, "rb") as m:
            nstat = pickle.load(m)
            init_count = nstat.num_updated
            # print("init count", init_count)

        packets = rdpcap(init_pcap)
        craft_pcap.write(packets)

    else:

        # init with benign packets
        init_extractor = FE(init_pcap, parse_type="scapy")

        if adv_csv_file is not None:
            write_to_csv = True

            # no need to create init pcap
            create_init = False
            output_csv = open(adv_csv_file, "w")
            headers = init_extractor.nstat.getNetStatHeaders()
            # print(headers)
            np.savetxt(output_csv, [headers], fmt="%s", delimiter=",")

        if create_init:
            init_pcap_file = PcapWriter(
                "../experiment/traffic_shaping/init_pcap/wiretap.pcap")

        t = tqdm(total=init_count)
        pkt_index = 0
        while pkt_index < init_count:
            try:
                traffic_data, packet = init_extractor.get_next_vector()
            except EOFError as e:
                print("EOF Reached")
                break
            t.update(1)
            pkt_index += 1
            if traffic_data == []:
                craft_pcap.write(packet)
                if create_init:
                    init_pcap_file.write(packet)
                np.savetxt(output_csv, [np.full(
                    features.shape, -1)], delimiter=",")
                meta_file.write(
                    ",".join([str(pkt_index), str(packet.time), "init_skipped\n"]))
                continue

            features = init_extractor.nstat.updateGetStats(*traffic_data)
            # init_extractor.dummy_nstat.updateGetStats(*traffic_data)
            # write init packets as is
            meta_file.write(
                ",".join([str(pkt_index), str(packet.time), "init\n"]))
            craft_pcap.write(packet)

            if create_init:
                init_pcap_file.write(packet)
            if write_to_csv:
                np.savetxt(output_csv, [features], delimiter=",")

        prev_pkt_time = float(packet.time)

        # get the database from initial fe to malware fe
        nstat = init_extractor.get_nstat()
        if create_init:
            model_path = "../models/netstat/wiretap_normal.pkl"
            with open(model_path, "wb") as of:
                pickle.dump(nstat, of)

    pkt_index = init_count
    prev_non_attack_time = None
    prev_pkt_time = nstat.prev_pkt_time
    feature_extractor = FE(mal_pcap, parse_type="scapy", nstat=nstat)

    # Set-up hyperparameters
    options = {'c1': 0.7, 'c2': 0.3, 'w': 0.5}
    # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}

    # load autoencoder
    # dataset_name="ku_flooding_kitsune_800"
    # with open("../data/{}/metadata.txt".format(dataset_name)) as file:
    #     metadata = json.load(file)
    # datamin = np.array(metadata["col_min"][:-1])
    # datamax = np.array(metadata["col_max"][:-1])
    # field_names=metadata["field_names"][:-1]

    # scaler, unscaler = min_max_scaler_gen(datamin, datamax)

    # load models
    # custom_objects = {'WcLayer': WcLayer}
    # prefix = "{}_{}_{}".format(dataset_name, 3, False)
    # aae = tf.keras.models.load_model(
    #     "../models/aae/{}_aae.h5".format(prefix), custom_objects=custom_objects)
    # encoder = tf.keras.models.load_model(
    #     "../models/aae/{}_encoder.h5".format(prefix), custom_objects=custom_objects)
    # wc_layer = aae.get_layer("wc_layer")

    t = tqdm(total=max_adv_pkt)

    # the base amount to adjust flooding traffic

    offset_time = base_offset
    total_reduction_ratio=0
    total_craft = 0
    adv_pkt_index = 0
    num_failed = 0
    craft_failed = 0
    total_craft_size = 0
    total_std = np.zeros((n_dims,))
    error_index = []
    while adv_pkt_index < max_adv_pkt:
        try:
            traffic_data, packet = feature_extractor.get_next_vector()
        except EOFError as e:
            print("EOF Reached")
            break

        pkt_index += 1

        if traffic_data == []:
            packet.time = float(packet.time) + offset_time
            craft_pcap.write(packet)
            malicious_pcap.write(packet)
            prev_pkt_time = float(packet.time)
            np.savetxt(output_csv, [np.full(
                features.shape, -1)], delimiter=",")
            meta_file.write(
                ",".join([str(pkt_index), str(packet.time), "mal_skipped\n"]))
            continue

        # non-attackers packet
        # if traffic_data[3] != "192.168.10.7":
        # if traffic_data[3] != "192.168.2.13":
        #     packet.time=float(packet.time)+offset_time
        #     craft_pcap.write(packet)
        #     malicious_pcap.write(packet)
        #     traffic_data[-2]+=offset_time
        #     features=feature_extractor.nstat.updateGetStats(*traffic_data)
        #     if write_to_csv:
        #         np.savetxt(output_csv, [features], delimiter=",")
        #     # feature_extractor.dummy_nstat.updateGetStats(*traffic_data)
        #
        #     meta_file.write(",".join([str(pkt_index),"non-attacker\n"]))
        #     prev_pkt_time=float(packet.time)
        #     continue

        # get random rows
        # rand_rows = np.random.randint(benign_feature.shape[0], size=num_benign_sample)

        tmp_pkt = packet.copy()
        if tmp_pkt.haslayer(TCP):
            tmp_pkt[TCP].remove_payload()
        elif tmp_pkt.haslayer(UDP):
            tmp_pkt[UDP].remove_payload()
        elif tmp_pkt.haslayer(ARP):
            tmp_pkt[ARP].remove_payload()
        else:
            tmp_pkt.remove_payload()
        min_pkt_len = len(tmp_pkt)

        db = feature_extractor.nstat.get_records(*traffic_data)

        # find original score
        dummy_db = copy.deepcopy(db)
        traffic_data[-2] += offset_time
        traffic_data[-2] = np.around(traffic_data[-2], decimals=6)
        features = feature_extractor.nstat.update_dummy_db(
            *(traffic_data), dummy_db, False)

        rmse_original = decision_func(features)

        if rmse_original < threshold:

            #
            # print(rmse)
            # print(features)
            # print(traffic_data)
            # print(pkt_index)
            # print(rmse)
            packet.time = traffic_data[-2]
            craft_pcap.write(packet)
            malicious_pcap.write(packet)
            features = feature_extractor.nstat.updateGetStats(*traffic_data)

            if write_to_csv:
                row = features
                np.savetxt(output_csv, [row], delimiter=",")
            # feature_extractor.dummy_nstat.updateGetStats(*traffic_data)
            #
            meta_file.write(
                ",".join([str(pkt_index), str(packet.time), "attacker_low\n"]))
            prev_pkt_time = float(packet.time)
            continue

        original_time = traffic_data[-2]
        print("original RMSE", rmse_original)
        print("original time", original_time)


        if decision_type == "adv":
            adv_feature = adv_func(features)
        else:
            adv_feature = None

        # set bounds:

        # max_craft_pkt=int((max_time-prev_pkt_time)/0.0002)
        max_bound = [original_time + max_time_window, max_craft_pkt]
        min_bound = [original_time, 0]
        if n_dims == 3:
            max_bound.append(max_pkt_size)
            min_bound.append(min_pkt_len)
        bounds = [np.array(min_bound), np.array(max_bound)]

        # PSO
        args = {"FE": feature_extractor, "adv_feature": adv_feature, "min_pkt_len": min_pkt_len, "decision_type": decision_type, "max_time_window": max_time_window,
                "prev_pkt_time": prev_pkt_time, "decision_func": decision_func, "max_pkt_size": max_pkt_size, "traffic_data": traffic_data, "max_craft_pkt": max_craft_pkt, "db": db}
        # optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=n_dims, options=options, bounds=bounds)
        # cost, pos = optimizer.optimize(f, iters=20,**args)

        mal_file_index=pkt_index - total_craft - init_count
        print("optimizing mal file index: ", mal_file_index)
        print("optimizing craft file index: ", pkt_index)

        iterations = 30

        if optimizer == "pso":
            cost, pos, aux, std, pos_history = optimize(
                options, n_dims, iterations, f, args, bounds, original_time, mutate_prob=mutate_prob)
        if optimizer == "de":
            cost, pos, aux, std, pos_history = differential_evolution(
                n_dims, iterations, f, args, bounds, original_time)

        print(cost, pos)

        log_file.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(rmse_original, original_time, mal_file_index, pkt_index, adv_pkt_index, cost, pos, aux ))
        # sum reduction ratio
        total_reduction_ratio += (rmse_original-cost)/rmse_original

        # record number of failed packets
        if cost > threshold:
            num_failed += 1
            # plot if there is insignificant decrease
            if rmse_original - cost < 0.1:
                limits=[(0, max_time_window),(min_bound[1],max_bound[1])]
                label=["time","num_craft"]
                animaition_path = animation_folder + \
                    "_{}.gif".format(pkt_index - total_craft - init_count)
                plot_particle_position(pos_history, pos, animaition_path, limits, label)

        if np.random.uniform() < 0.1:

            limits=[(0, max_time_window),(min_bound[1],max_bound[1])]
            label=["time","num_craft"]

            animaition_path = animation_folder + \
                "_{}.gif".format(pkt_index - total_craft - init_count)
            plot_particle_position(pos_history, pos, animaition_path, limits, label)

        log_file.write("-" * 100 + "\n")
        # print(aux)
        total_std += std


        # apply fake packets to feature extractor
        num_craft = pos[1].astype(int)
        total_craft += num_craft

        # num_craft=0
        # print(num_craft)
        # print(aux)
        for i in range(num_craft):

            traffic_vector = traffic_data[:-2] + \
                [aux[i], int(aux[i + max_craft_pkt])]
            features = feature_extractor.nstat.updateGetStats(*traffic_vector)

            # print("real craft feature", features)
            craft_cost = decision_func(features)

            if craft_cost > threshold:
                craft_failed += 1

            craft_packet = packet_gen(packet, traffic_vector)
            total_craft_size += aux[i + max_craft_pkt]
            # print(traffic_vector)
            # craft_packet.show()
            # packet.show()

            malicious_pcap.write(craft_packet)
            craft_pcap.write(craft_packet)
            if write_to_csv:
                np.savetxt(output_csv, [features], delimiter=",")
            meta_file.write(
                ",".join([str(pkt_index), str(craft_packet.time), "craft\n"]))

            pkt_index += 1
        t.update(1)

        # set offset
        offset_time += (pos[0] - original_time)

        # write malicious packet
        traffic_data[-2] = pos[0]

        features = feature_extractor.nstat.updateGetStats(*traffic_data)
        # print("real mal feature", features)

        if write_to_csv:
            np.savetxt(output_csv, [features], delimiter=",")

        true_cost = decision_func(features)
        try:
            np.testing.assert_almost_equal(true_cost, cost, decimal=6)
        except AssertionError as e:
            print("true cost", true_cost)
            print("cost", cost)
            print(pkt_index)
            raise
            error_index.append(pkt_index)

        # feature_extractor.dummy_nstat.updateGetStats(*traffic_data)
        packet.time = pos[0]
        craft_pcap.write(packet)
        malicious_pcap.write(packet)
        meta_file.write(
            ",".join([str(pkt_index), str(packet.time), "malicious\n"]))
        adv_pkt_index += 1
        prev_pkt_time = pos[0]

    report = {}
    if adv_pkt_index == 0:
        report["num_altered"] = 0
    else:
        report["av_time_delay"] = (offset_time - base_offset) / adv_pkt_index
        report["av_num_craft"] = total_craft / adv_pkt_index
        report["total_craft"] = total_craft
        report["num_altered"] = adv_pkt_index
        report["average_reduction_ratio"]=total_reduction_ratio/adv_pkt_index
        report["adv_mal_ratio"] = (
            pkt_index - init_count - total_craft) / adv_pkt_index
        report["av_std"] = total_std / adv_pkt_index
        report["av_pkt_size"] = total_craft_size / total_craft
        report["num_seen"] = pkt_index - total_craft - init_count
    report["num_failed"] = num_failed
    report["craft_failed"] = craft_failed
    return report


def get_decision_func(decision_type, model_path):
    if decision_type == "kitsune":
        with open(model_path, "rb") as m:
            model = pickle.load(m)

        def kitsune(features):
            # process batch
            if features.ndim == 2:
                rmse = []
                for i in range(features.shape[0]):
                    rmse.append(model.process(features[i]))
                return np.array(rmse)
            else:
                return model.process(features)
        return kitsune

    if decision_type == "autoencoder":
        model = tf.keras.models.load_model(model_path)

        max_val = np.genfromtxt("../models/surrogate_max.csv", delimiter=",")
        min_val = np.genfromtxt("../models/surrogate_min.csv", delimiter=",")

        def autoencoder(features):
            ndims = features.ndim
            if ndims == 1:
                features = np.expand_dims(features, axis=0)
            features = (features - min_val) / (max_val - min_val + 1e-6)
            features = features.astype(np.float32)
            reconstructions = model.predict(features)
            true_cost = tf.keras.losses.mse(reconstructions, features)

            if ndims == 1:
                true_cost = true_cost[0]
            return true_cost.numpy()
        return autoencoder

    if decision_type == "adv":
        model = tf.keras.models.load_model(
            "../models/surrogate_ae_video_ho.h5")

        max_val = np.genfromtxt("max.csv", delimiter=",")
        min_val = np.genfromtxt("min.csv", delimiter=",")

        def adv(features):
            ndims = features.ndim
            if ndims == 1:
                features = np.expand_dims(features, axis=0)
            features = (features - min_val) / (max_val - min_val)
            features = features.astype(np.float32)

            return adversarial_sample_ae(model, features)
        return adv


def optimize_de(f, args, bounds):
    return differential_evolution(f, bounds, args=args)


def optimize(options, n_dims, iterations, f, args, bounds, original_time, clamp=None, mutate_prob=-1):
    topology = Traffic()

    pos_history = []

    n_particles = 20

    # de params
    mutation_factor = 0.8
    crossp = 0.7
    mutation_candidates = [[idx for idx in range(
        n_particles) if idx != i] for i in range(n_particles)]
    mutation_candidates = np.array(mutation_candidates)

    swarm = create_swarm(n_particles=n_particles, dimensions=n_dims,
                         options=options, bounds=bounds, discrete_index=1)


    # set first particle to have original time and no craft packet
    swarm.position[0][1] = 0
    swarm.position[0][0] = original_time

    # swarm.position[1]=[1502269093.015671,100.000000,211.000000]

    # swarm.position[1]=np.array([1502269073.367330,18.000000,192.000000])
    # swarm.best_pos=np.array([0,0,0])

    pbar = tqdm(range(iterations), position=1)

    # file=open("tmp_pso.txt","a")
    for i in pbar:
        # pos_history.append(np.vstack((swarm.position[:,1:],np.expand_dims(swarm.best_pos[1:], axis=0))))

        # Part 1: Update personal best

        swarm.current_cost, swarm.current_aux = f(
            swarm.position, **args, verbose=False)  # Compute current cost
        # file.write("{}".format(i)+"\n")
        # file.write(pprint.pformat(swarm.current_cost)+"\n")
        # file.write(pprint.pformat(swarm.position)+"\n")

        if i == 0:
            swarm.pbest_cost, swarm.pbest_aux = swarm.current_cost, swarm.current_aux
            swarm.best_cost, swarm.best_aux = swarm.current_cost, swarm.current_aux
            swarm.best_pos = swarm.position
            swarm.pbest_iter = np.zeros((n_particles,))
        # swarm.pbest_cost, swarm.pbest_aux= f(swarm.pbest_pos, **args)  # Compute personal best pos
        # print(swarm.pbest_cost)



        # binomially mutate
        if np.random.rand() < mutate_prob:
            tmp_pos = swarm.position
            swarm.trial_pos = topology.mutate_swarm(
                swarm, mutation_factor, crossp, mutation_candidates, bounds)
            swarm.trial_cost, swarm.trial_aux = f(swarm.trial_pos, **args)
            # print("tc",swarm.trial_cost)
            # print("ta",swarm.trial_aux)
            # print("tp",swarm.trial_pos)
            swarm.position, swarm.current_cost, swarm.current_aux = topology.compute_mbest(
                swarm)
            # print("-"*50)

        swarm.pbest_pos, swarm.pbest_cost, swarm.pbest_aux, swarm.pbest_iter = topology.compute_pbest(
            swarm, i)  # Update and store

        # Part 2: Update global best
        # Note that gbest computation is dependent on your topology
        # if np.min(swarm.pbest_cost) < swarm.best_cost:
        # best index is global minimum, others are best in the neighbourhood
        swarm.best_pos, swarm.best_cost, swarm.best_aux, swarm.best_index = topology.compute_gbest_local(
            swarm, 2, 4)
        # best_iter=i
        # file.write(pprint.pformat(swarm.best_pos)+"\n")

        # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
        swarm.velocity = topology.compute_velocity(
            swarm, bounds=bounds, clamp=clamp, iter=i)
        if np.random.rand() < 0.5:
            strat = "random"
        else:
            strat = "nearest"
        swarm.position = topology.compute_position(
            swarm, bounds=bounds, bh=BoundaryHandler(strategy=strat))

        # print(swarm.position)
        # file.write(pprint.pformat(swarm.velocity)+"\n")
        #
        # file.write("-"*100+"\n")
        # post_fix="c: {:.4f}, n: {:.0f}".format(swarm.best_cost, np.trunc(swarm.best_pos[1]))
        # if n_dims==3:
        #     post_fix+=", p: {:.0f}".format(swarm.best_pos[2])
        post_fix = "c: {:.4f}".format(swarm.best_cost[swarm.best_index])
        pbar.set_postfix_str(post_fix)

        norm_pos=np.copy(swarm.position)
        norm_pos[:,0]=norm_pos[:,0]-bounds[0][0]
        norm_best=np.copy(swarm.best_pos)
        norm_best[:,0]=norm_best[:,0]-bounds[0][0]
        pos_history.append(
            np.vstack((norm_pos, norm_best)))


        # if i-best_iter>10:
        #     print("early stopping")
        #     break
    # print("best config found in {} iterations".format(best_iter))
    std = np.std(swarm.position, axis=0)
    # print("best conf", swarm.best_cost[swarm.best_index], swarm.best_pos[swarm.best_index], swarm.best_aux[swarm.best_index])

    # print(swarm.best_cost)
    #
    # bc=np.argmin(swarm.best_cost)
    # swarm.best_pos=swarm.best_pos[bc]
    # swarm.best_aux=swarm.best_aux[bc]

    # print(swarm.position)
    return swarm.best_cost[swarm.best_index], swarm.best_pos[swarm.best_index], swarm.best_aux[swarm.best_index], std, pos_history


# if __name__ == '__main__':
    # logging.getLogger('pyswarms').setLevel(logging.WARNING)
    # extract_features("../../TrafficManipulator/[Normal]GoogleHome.pcap", output_file_name="../data/kitsune/normal.csv", write_prob=0.8,add_label=True)
    # extract_features("../../TrafficManipulator/[HTTP_Flooding]GoogleHome_thread_800_origin.pcap", output_file_name="../data/kitsune/flooding.csv", write_prob=0.8, add_label=True)
    # craft_adversary("../ku_http_flooding/pcaps/[HTTP_Flooding]GoogleHome_thread_800_origin.pcap","../ku_http_flooding/pcaps/[Normal]GoogleHome.pcap", max_time_window=0.1,max_adv_pkt=1000,max_craft_pkt=3, max_pkt_size=200)


# def extract(x):
#     res=np.random.rand(10,2)
#     return res
#
# #function to optimze
# def f(x,y):
#     x=extract(x)
#     return (x[:,0]-y[:,0])**2+(x[:,1]-y[:,1])**2
#

#
# # Call instance of PSO
# optimizer = ps.single.GlobalBestPSO(n_particles=10, dimensions=3, options=options)
#
# # want x to be close to 3,4
# y=np.tile([3,4], [10,1])
# print(y)
#
# # each particle is 3d, each particle is then embedded to be 2 d and optimized to be close to 3,4
#
# # Perform optimization
