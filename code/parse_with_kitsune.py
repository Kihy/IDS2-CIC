from after_image.feature_extractor import *
from tqdm import tqdm
np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:f}'.format})

def parse_kitsune(pcap_file, output_file_name, add_label=False, write_prob=1, count=float('Inf'), parse_type="scapy"):
    feature_extractor=FE(pcap_file, parse_type=parse_type)
    # temp=open("tmp.txt","w")
    headers=feature_extractor.nstat.getNetStatHeaders()

    output_file=open(output_file_name,"w")
    label=output_file_name.split('/')[-1]




    if add_label:
        headers+=["label"]

    # print(headers)
    np.savetxt(output_file,[headers], fmt="%s", delimiter=",")

    skipped=0
    written=0
    t = tqdm(total=count)
    pkt_index=0
    while pkt_index < count:
        try:
            if parse_type=="scapy":
                traffic_data,packet = feature_extractor.get_next_vector()
            else:
                traffic_data = feature_extractor.get_next_vector()
        except EOFError as e:
            print("EOF Reached")
            print(e)
            break
        except ValueError as e:
            print("EOF Reached")
            print(e)
            break
        except StopIteration as e:
            print(e)
            print("EOF Reached")
            break

        pkt_index+=1
        t.update(1)
        if traffic_data == []:
            np.savetxt(output_file, [np.full(features.shape,-1)], delimiter=",")
            skipped+=1
            continue
        # print(traffic_data)
        features=feature_extractor.nstat.updateGetStats(*traffic_data)

        if np.isnan(features).any():
            print(features)
            break
        # temp.write("{}\n".format(pkt_index))
        if np.random.uniform(0, 1)<write_prob:
            if add_label:
                np.savetxt(output_file, [features], delimiter=",", newline=",")
                np.savetxt(output_file,[label],fmt="%s")
            else:
                np.savetxt(output_file, [features], delimiter=",")
        written+=1

    output_file.close()
    print("skipped:",skipped)
    print("written:",written)

if __name__ == '__main__':
    # file_name="../ku_dataset/flooding_attacker_only"
    file_name="../ku_dataset/[OS & service detection]traffic_GoogleHome_av_only"
    # file_name="../experiment/traffic_shaping/scanning/autoencoder_1_10_3_pso0.5/adv/iter_0"
    # file_name="../ku_dataset/[OS & service detection]traffic_GoogleHome_replay"
    parse_kitsune(file_name+".pcap",file_name+".csv", False, parse_type="scapy")
    # parsing normal features
    # parse_kitsune("../ku_http_flooding/pcaps/[Normal]GoogleHome.pcap", "../ku_http_flooding/kitsune_features/[Normal]GoogleHome.csv", True)
    # parsing flooding features
    # parse_kitsune("../ku_http_flooding/pcaps/[HTTP_Flooding]GoogleHome_thread_800_origin.pcap","../ku_http_flooding/kitsune_features/[HTTP_Flooding]GoogleHome_thread_800_origin.csv", True, parse_type="tsv")
    # parse_kitsune("../experiment/traffic_shaping/normal_800.pcap", "../experiment/traffic_shaping/normal_800.csv", False, parse_type="scapy")
    # parse_kitsune("../experiment/traffic_shaping/crafted_pcap.pcap", "../experiment/traffic_shaping/crafted_pcap_0.1k.csv", False, parse_type="scapy")
    # parse_kitsune("../kitsune_dataset/wiretap_normal_hostonly.pcapng", "../kitsune_dataset/wiretap_normal_hostonly.csv", True, parse_type="scapy")
    # parse_kitsune("../kitsune_dataset/wiretap_malicious_hostonly.pcapng", "../kitsune_dataset/wiretap_malicious_hostonly.csv", True, parse_type="scapy")
    # parse_kitsune("../kitsune_dataset/Active Wiretap_pcap.pcapng", "../kitsune_dataset/Active Wiretap_pcap.csv", False, parse_type="scapy")
    # parse_kitsune("../ku_dataset/[ARP]traffic_1.pcap", "../ku_dataset/[ARP]traffic_1.pcap.csv", False, parse_type="scapy")
