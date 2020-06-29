from adversarial import *
from eval import *
from input_utils import *
from predict import *
from train import *
from vis_utils import *

if __name__ == '__main__':
    dataset_name = "ku_flooding_800"
    model_type = "3layer"
    model_path = "../models/{}_{}".format(model_type, dataset_name)
    data_path = "../ku_httpflooding"
    percent_theta = 0.1
    target = 2
    attack_type = ["normal"]
    columns = ["protocol_type", "tl_data_len", "fin_flag", "syn_flag" ,"rst_flag", "psh_flag","ack_flag","urg_flag","ece_flag","cwr_flag","num_of_frags", "src_dst_same", "same_sip_pkt_cnt",
               "same_dip_pkt_cnt", "same_sip_sport_pkt_cnt", "same_dip_dport_pkt_cnt",
               "same_sip_pkt_dip_cnt", "same_dip_pkt_sip_cnt", "same_src_dst_pkt_sport_cnt",
               "same_src_dst_pkt_dport_cnt", "same_sip_src_bytes", "same_dip_dst_bytes",
               "same_sip_icmp_ratio", "same_dip_icmp_ratio", "same_sip_syn_ratio",
               "same_dip_syn_ratio", "same_sip_syn_ack_diff_cnt", "same_dip_syn_ack_diff_cnt","category"]
    label_col="category"
    # ingore_file=["Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"]
    # include_file=["normal.pcap_Flow.csv","slowloris.pcap_Flow.csv"]
    # fixed=["Flag","Subflow"]
    # fixed=["cnt","protocol","bwd","skew","kurtosis","min","iat"]
    # fixed=["same"]
    fixed=[]
    meta_col=["src_ip","dst_ip","timestamp","idx","fin_flag", "syn_flag" ,"rst_flag", "psh_flag","ack_flag","urg_flag","ece_flag","cwr_flag"]
    # dr = DataReader(data_path, 0.2, 0.2, dataset_name=dataset_name, protocols=["TCP"], columns=columns, label_col=label_col, use_filename_as_label=True,
    # ignore=True, files=["Normal","800"],meta_col=meta_col,
    # attack_type=None
    # )
    # dr.generate_dataframes()
    # dr.write_to_csv()
    # dr.dataset_statistics()
    # dr.start()
    # generate_fake_data(model_path, columns,10000,dataset_name)
    # train_normal_network(dataset_name, model_path, batch_size=1024, epochs=20,label_name=label_col)
    # evaluate_network(dataset_name, model_path, "{}_{}".format(model_type,dataset_name), batch_size=1024, label_name=label_col)
    adversarial_generation(dataset_name, model_path, target, "train", num_samples=1000, fixed=fixed, theta=0.05,label_name=label_col, alter=0)
    # format_converter("../pcap_data/slowloris_adv.pcap_Flow.csv", "../experiment/column_map.csv", out_dir="../experiment/attack_pcap/", metadata=True, use_filename_as_label=False)
    # add_label_col("slowloris_adv_flow.csv")
    # predict_sample("slowloris_adv_flow.csv", model_path, dataset_name)
    # vis_original_input("../IDS2017_ML/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",["BENIGN","PortScan"])
    # vis_attack_distribution("../data/patator/val.csv","../experiment/attack_pcap/cic_patator.png", attack=1, ignore_label=False)
    # vis_attack_distribution("../experiment/attack_pcap/slowloris.pcap_Flow.csv","../experiment/attack_pcap/slowloris.png",ignore_label=True)
    # vis_diff("../experiment/attack_pcap/slowloris_adv_flow.csv","../experiment/attack_pcap/slowloris_flow.csv", "../experiment/pert_vis/diff.png")
