import numpy as np
import pyswarms as ps
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
import pyswarms.backend as P


def packet_gen(ori_pkt, traffic_vector):
    packet=ori_pkt.copy()
    packet.time=traffic_vector[-2]
    # print(traffic_vector)
    packet[IP].remove_payload()

    packet[IP].add_payload(TCP(sport=int(traffic_vector[4]),dport=int(traffic_vector[6])))

    payload_size=int(traffic_vector[-1])-len(packet)

    packet[TCP].add_payload(Raw(load="a"*payload_size))
    # packet.show()

    return packet

def distance(feature, encoder, scaler, wc_layer):
    input_feature = scaler(feature)
    input_feature=np.expand_dims(input_feature,axis=0)
    style, representation, pred_label = encode(encoder, wc_layer, input_feature)
    # if pred_label[0][0]>0.2:
    #     print(pred_label)
    # normal label
    if np.any(np.abs(representation.numpy()) > 3):
        print("repr:",representation)
    benign_label=[[0,1]]
    mse = tf.keras.losses.binary_crossentropy(benign_label, pred_label)
    return mse.numpy()[0]

def rebuild(timestamp, num_craft, prev_pkt_time, max_pkt_size,min_pkt_len):
    """
    builds num_craft packets with random timestamp and framelen, all other fields are constant with malicious packet.
    return matrix has column as pair of crafted packets. only rebuilds 1 malicious packet
    """

    #get random time and length

    t=np.random.uniform(prev_pkt_time,timestamp, size=num_craft)

    t=np.sort(t)
    l=np.random.randint(min_pkt_len, max_pkt_size,size=num_craft)

    return np.stack((t,l))




def f(x, encoder,wc_layer, scaler, FE, max_time_window, prev_pkt_time, max_pkt_size, traffic_data, max_craft_pkt, min_pkt_len):
    """optimize function, x[:,0] is time and x[:,1] is number of crafted packets"""
    # go through each particle
    FE.save_nstat_state()
    min_dists=[]
    for i in range(x.shape[0]):
        timestamp=x[:,0][i]
        num_craft=np.rint(x[:,1][i]).astype(int)
        if num_craft<0:
            print(num_craft)
            num_craft=0

        crafted_features=rebuild(timestamp, num_craft, prev_pkt_time, max_pkt_size, min_pkt_len)
        if crafted_features is None:
            print(x[:,1][i])
        # crafted features
        for j in range(max_craft_pkt):
            if j < num_craft:

                FE.dummy_nstat.updateGetStats(*(traffic_data[:-2]+crafted_features[:,j].tolist()))

                x[i:i+1,(j+1)*2:(j+1)*2+2]=crafted_features[:,j]
            # set to 0
            else:
                x[i:i+1,(j+1)*2:(j+1)*2+2]=0

        #malicious packet
        traffic_data[-2]=timestamp
        feature=FE.dummy_nstat.updateGetStats(*(traffic_data))
        #add min distance
        min_dists.append(distance(feature, encoder, scaler,wc_layer))
        FE.roll_back()

    return np.array(min_dists)



def craft_adversary(mal_pcap, benign_file, init_pcap, max_time_window=60, max_craft_pkt=5, num_benign_sample=100, max_pkt_size=655 , max_adv_pkt=None):



    craft_pcap = PcapWriter("../experiment/pso/crafted_pcap.pcap")

    meta_file=open("../experiment/pso/crafted_meta.csv","w")
    meta_file.write("packet_index,comment\n")
    feature_file=open("../experiment/pso/crafted_features.csv","w")


    # init with benign packets
    init_extractor=FE(init_pcap)

    headers=init_extractor.nstat.getNetStatHeaders()
    np.savetxt(feature_file,[headers], fmt="%s", delimiter=",")


    init_count=5000
    t = tqdm(total=init_count)
    pkt_index=0
    while pkt_index < init_count:
        try:
            traffic_data, packet = init_extractor.get_next_vector()
        except EOFError as e:
            print("EOF Reached")
            break
        t.update(1)
        pkt_index+=1
        if traffic_data == []:
            craft_pcap.write(packet)
            continue

        features=init_extractor.nstat.updateGetStats(*traffic_data)
        init_extractor.dummy_nstat.updateGetStats(*traffic_data)
        # write init packets as is
        np.savetxt(feature_file, [features], delimiter=",")


        meta_file.write(",".join([str(pkt_index),"init\n"]))
        craft_pcap.write(packet)

    prev_pkt_time=float(packet.time)
    prev_non_attack_time=None
    #get the database from initial fe to malware fe
    nstat, dummy_nstat=init_extractor.get_nstat()
    feature_extractor=FE(mal_pcap, nstat=nstat, dummy_nstat=dummy_nstat)

    #get benign features
    benign_feature = np.genfromtxt(benign_file, delimiter=',')

    # Set-up hyperparameters
    options = {'c1': 0.5, 'c2': 0.3, 'w':0.5}
    # options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9, 'k': 3, 'p': 2}
    n_dims=2*(max_craft_pkt+1)

    #load autoencoder
    dataset_name="ku_flooding_kitsune"
    with open("../data/{}/metadata.txt".format(dataset_name)) as file:
        metadata = json.load(file)
    datamin = np.array(metadata["col_min"][:-1])
    datamax = np.array(metadata["col_max"][:-1])
    field_names=metadata["field_names"][:-1]

    scaler, unscaler = min_max_scaler_gen(datamin, datamax)

    # load models
    custom_objects = {'WcLayer': WcLayer}
    prefix = "{}_{}_{}".format(dataset_name, 3, False)
    aae = tf.keras.models.load_model(
        "../models/aae/{}_aae.h5".format(prefix), custom_objects=custom_objects)
    encoder = tf.keras.models.load_model(
        "../models/aae/{}_encoder.h5".format(prefix), custom_objects=custom_objects)
    wc_layer = aae.get_layer("wc_layer")

    t = tqdm(total=max_adv_pkt)

    adv_pkt_index=0
    while adv_pkt_index < max_adv_pkt:
        try:
            traffic_data, packet = feature_extractor.get_next_vector()
        except EOFError as e:
            print("EOF Reached")
            break

        pkt_index+=1
        if traffic_data == []:
            packet.time=prev_pkt_time+0.1
            craft_pcap.write(packet)
            prev_pkt_time=packet.time
            continue
        # non-attackers packet
        if traffic_data[3] != "192.168.10.7":
            packet.time=prev_pkt_time+0.1
            craft_pcap.write(packet)
            np.savetxt(feature_file, [features], delimiter=",")
            meta_file.write(",".join([str(pkt_index),"non-attacker\n"]))
            prev_pkt_time=packet.time
            continue

        t.update(1)
        adv_pkt_index+=1


        # get random rows
        rand_rows = np.random.randint(benign_feature.shape[0], size=num_benign_sample)

        #set bounds:
        max_bound=[prev_pkt_time+max_time_window,max_craft_pkt]+[1 for i in range(2*max_craft_pkt)]
        min_bound=[prev_pkt_time, 0]+[0 for i in range(2*max_craft_pkt)]
        bounds=[min_bound, max_bound]

        tmp_pkt=packet.copy()
        tmp_pkt[TCP].remove_payload()
        min_pkt_len=len(tmp_pkt)

        # PSO
        args={"FE":feature_extractor,"wc_layer":wc_layer, "min_pkt_len":min_pkt_len,"scaler":scaler, "max_time_window":max_time_window, "prev_pkt_time":prev_pkt_time, "encoder":encoder, "max_pkt_size":max_pkt_size, "traffic_data":traffic_data,"max_craft_pkt":max_craft_pkt}
        # optimizer = ps.single.GlobalBestPSO(n_particles=20, dimensions=n_dims, options=options, bounds=bounds)
        # cost, pos = optimizer.optimize(f, iters=20,**args)

        cost,pos=optimize(options, n_dims, 30, f, args,bounds)
        print(cost)
        print(pos)

        # apply fake packets to feature extractor
        num_craft=np.rint(pos[1]).astype(int)
        for i in range(num_craft):
            traffic_vector=traffic_data[:-2]+pos[(i+1)*2:(i+1)*2+2].tolist()
            features=feature_extractor.nstat.updateGetStats(*traffic_vector)
            feature_extractor.dummy_nstat.updateGetStats(*traffic_vector)
            craft_packet=packet_gen(packet, traffic_vector)
            craft_pcap.write(craft_packet)
            meta_file.write(",".join([str(pkt_index),"craft\n"]))
            np.savetxt(feature_file, [features], delimiter=",")
            pkt_index+=1

        # write malicious packet
        traffic_data[-2]=pos[0]
        features=feature_extractor.nstat.updateGetStats(*traffic_data)
        feature_extractor.dummy_nstat.updateGetStats(*traffic_data)
        packet.time=pos[0]
        craft_pcap.write(packet)
        meta_file.write(",".join([str(pkt_index),"malicious\n"]))
        np.savetxt(feature_file, [features], delimiter=",")

        prev_pkt_time=pos[0]

def optimize(options, n_dims, iterations, f, args,bounds):
    topology=Star()
    swarm=P.create_swarm(n_particles=20, dimensions=n_dims, options=options, bounds=bounds)
    for i in tqdm(range(iterations)):
        # Part 1: Update personal best
        swarm.current_cost = f(swarm.position, **args) # Compute current cost
        swarm.pbest_cost = f(swarm.pbest_pos, **args)  # Compute personal best pos
        swarm.pbest_pos, swarm.pbest_cost = P.compute_pbest(swarm) # Update and store

        # Part 2: Update global best
        # Note that gbest computation is dependent on your topology
        if np.min(swarm.pbest_cost) < swarm.best_cost:
            swarm.best_pos, swarm.best_cost = topology.compute_gbest(swarm)

        # Part 3: Update position and velocity matrices
        # Note that position and velocity updates are dependent on your topology
        swarm.velocity = topology.compute_velocity(swarm)
        swarm.position = topology.compute_position(swarm)

    return swarm.best_cost, swarm.best_pos



if __name__ == '__main__':
    logging.getLogger('pyswarms').setLevel(logging.WARNING)
    # extract_features("../../TrafficManipulator/[Normal]GoogleHome.pcap", output_file_name="../data/kitsune/normal.csv", write_prob=0.8,add_label=True)
    # extract_features("../../TrafficManipulator/[HTTP_Flooding]GoogleHome_thread_800_origin.pcap", output_file_name="../data/kitsune/flooding.csv", write_prob=0.8, add_label=True)
    craft_adversary("../ku_http_flooding/pcaps/[HTTP_Flooding]GoogleHome_thread_1_origin.pcap", "../experiment/pso/benign.csv","../ku_http_flooding/pcaps/[Normal]GoogleHome.pcap", max_adv_pkt=4500, max_pkt_size=200)



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
