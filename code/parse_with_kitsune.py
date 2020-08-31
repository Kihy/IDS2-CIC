from after_image.feature_extractor import *
from tqdm import tqdm


def parse_kitsune(pcap_file, output_file_name, add_label=False, write_prob=1, count=float('Inf')):
    feature_extractor=FE(pcap_file)

    output_file=open(output_file_name,"w")
    label=output_file_name.split('/')[-1]


    headers=feature_extractor.nstat.getNetStatHeaders()
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
            traffic_data, packet = feature_extractor.get_next_vector()
        except EOFError as e:
            print("EOF Reached")
            break
        except ValueError as e:
            print("EOF Reached")
            break

        pkt_index+=1
        t.update(1)
        if traffic_data == []:
            skipped+=1
            continue

        features=feature_extractor.nstat.updateGetStats(*traffic_data)

        if np.isnan(features).any():
            print(features)
            break


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
    # parsing normal features
    # parse_kitsune("../ku_http_flooding/pcaps/[Normal]GoogleHome.pcap", "../ku_http_flooding/kitsune_features/[Normal]GoogleHome.csv", True)
    # parsing flooding features
    # parse_kitsune("../ku_http_flooding/pcaps/[HTTP_Flooding]GoogleHome_thread_1_origin.pcap","../ku_http_flooding/kitsune_features/[HTTP_Flooding]GoogleHome_thread_1_origin.csv", True)

    parse_kitsune("../experiment/pso/crafted_pcap.pcap", "../experiment/pso/crafted_pcap_new.csv", False)
