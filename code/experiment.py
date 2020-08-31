from adversarial import *
from eval import *
from input_utils import *
from predict import *
from train import *
from vis_utils import *

if __name__ == "__main__":
    # dataset_name = "ku_flooding_800"
    dataset_name="ku_flooding_kitsune"
    model_type = "3layer"
    model_path = "../models/{}_{}".format(model_type, dataset_name)
    data_path = "../ku_http_flooding/kitsune_features"
    percent_theta = 0.1
    target = 2
    # attack_type = ["normal"]
    attack_type=None
    # columns = ["protocol_type", "tl_data_len", "fin_flag", "syn_flag" ,"rst_flag", "psh_flag","ack_flag","urg_flag","ece_flag","cwr_flag","num_of_frags", "src_dst_same", "same_sip_pkt_cnt",
    #            "same_dip_pkt_cnt", "same_sip_sport_pkt_cnt", "same_dip_dport_pkt_cnt",
    #            "same_sip_pkt_dip_cnt", "same_dip_pkt_sip_cnt", "same_src_dst_pkt_sport_cnt",
    #            "same_src_dst_pkt_dport_cnt", "same_sip_src_bytes", "same_dip_dst_bytes",
    #            "same_sip_icmp_ratio", "same_dip_icmp_ratio", "same_sip_syn_ratio",
    #            "same_dip_syn_ratio", "same_sip_syn_ack_diff_cnt", "same_dip_syn_ack_diff_cnt","category"]

    columns=["HT_MI_5_weight", "HT_MI_5_mean", "HT_MI_5_std", "HT_MI_3_weight", "HT_MI_3_mean", "HT_MI_3_std", "HT_MI_1_weight", "HT_MI_1_mean", "HT_MI_1_std", "HT_MI_0.1_weight", "HT_MI_0.1_mean", "HT_MI_0.1_std", "HT_MI_0.01_weight", "HT_MI_0.01_mean", "HT_MI_0.01_std", "HT_H_5_weight", "HT_H_5_mean", "HT_H_5_std", "HT_H_5_radius", "HT_H_5_magnitude", "HT_H_5_covariance", "HT_H_5_pcc", "HT_H_3_weight", "HT_H_3_mean", "HT_H_3_std", "HT_H_3_radius", "HT_H_3_magnitude", "HT_H_3_covariance", "HT_H_3_pcc", "HT_H_1_weight", "HT_H_1_mean", "HT_H_1_std", "HT_H_1_radius", "HT_H_1_magnitude", "HT_H_1_covariance", "HT_H_1_pcc", "HT_H_0.1_weight", "HT_H_0.1_mean", "HT_H_0.1_std", "HT_H_0.1_radius", "HT_H_0.1_magnitude", "HT_H_0.1_covariance", "HT_H_0.1_pcc", "HT_H_0.01_weight", "HT_H_0.01_mean", "HT_H_0.01_std", "HT_H_0.01_radius", "HT_H_0.01_magnitude", "HT_H_0.01_covariance", "HT_H_0.01_pcc", "HT_jit_5_weight", "HT_jit_5_mean", "HT_jit_5_std", "HT_jit_3_weight", "HT_jit_3_mean", "HT_jit_3_std", "HT_jit_1_weight", "HT_jit_1_mean", "HT_jit_1_std", "HT_jit_0.1_weight", "HT_jit_0.1_mean", "HT_jit_0.1_std", "HT_jit_0.01_weight", "HT_jit_0.01_mean", "HT_jit_0.01_std", "HT_Hp_5_weight", "HT_Hp_5_mean", "HT_Hp_5_std", "HT_Hp_5_radius", "HT_Hp_5_magnitude", "HT_Hp_5_covariance", "HT_Hp_5_pcc", "HT_Hp_3_weight", "HT_Hp_3_mean", "HT_Hp_3_std", "HT_Hp_3_radius", "HT_Hp_3_magnitude", "HT_Hp_3_covariance", "HT_Hp_3_pcc", "HT_Hp_1_weight", "HT_Hp_1_mean", "HT_Hp_1_std", "HT_Hp_1_radius", "HT_Hp_1_magnitude", "HT_Hp_1_covariance", "HT_Hp_1_pcc", "HT_Hp_0.1_weight", "HT_Hp_0.1_mean", "HT_Hp_0.1_std", "HT_Hp_0.1_radius", "HT_Hp_0.1_magnitude", "HT_Hp_0.1_covariance", "HT_Hp_0.1_pcc", "HT_Hp_0.01_weight", "HT_Hp_0.01_mean", "HT_Hp_0.01_std", "HT_Hp_0.01_radius", "HT_Hp_0.01_magnitude", "HT_Hp_0.01_covariance", "HT_Hp_0.01_pcc", "label"]
    label_col="label"
    # label_col="category"
    # ingore_file=["Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"]
    # include_file=["normal.pcap_Flow.csv","slowloris.pcap_Flow.csv"]
    # fixed=["Flag","Subflow"]
    # fixed=["cnt","protocol","bwd","skew","kurtosis","min","iat"]
    # fixed=["same"]
    fixed=[]
    # meta_col=["src_ip","dst_ip","timestamp","idx","fin_flag", "syn_flag" ,"rst_flag", "psh_flag","ack_flag","urg_flag","ece_flag","cwr_flag"]
    meta_col=[]
    # protocols=["TCP"]
    protocols=[]
    dr = DataReader(data_path, 0.2, 0.2, dataset_name=dataset_name, protocols=protocols, columns=columns, label_col=label_col, use_filename_as_label=True,
    ignore=True, files=["Normal","1_"],meta_col=meta_col, replace_label=True,
    attack_type=attack_type, type="kitsune"
    )
    # dr.generate_dataframes()
    # dr.write_to_csv()
    # dr.dataset_statistics()
    dr.start()
    # generate_fake_data(model_path, columns,10000,dataset_name)
    train_normal_network(dataset_name, model_path, batch_size=1024, epochs=20,label_name=label_col)
    # evaluate_network(dataset_name, model_path, "{}_{}".format(model_type,dataset_name), batch_size=1024, label_name=label_col)
    # adversarial_generation(dataset_name, model_path, target, "test", num_samples=10000, fixed=fixed, theta=0.05,label_name=label_col, alter=0)
    # format_converter("../pcap_data/slowloris_adv.pcap_Flow.csv", "../experiment/column_map.csv", out_dir="../experiment/attack_pcap/", metadata=True, use_filename_as_label=False)
    # add_label_col("slowloris_adv_flow.csv")
    # predict_sample("slowloris_adv_flow.csv", model_path, dataset_name)
    # vis_original_input("../IDS2017_ML/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",["BENIGN","PortScan"])
    # vis_attack_distribution("../data/patator/val.csv","../experiment/attack_pcap/cic_patator.png", attack=1, ignore_label=False)
    # vis_attack_distribution("../experiment/attack_pcap/slowloris.pcap_Flow.csv","../experiment/attack_pcap/slowloris.png",ignore_label=True)
    # vis_diff("../experiment/attack_pcap/slowloris_adv_flow.csv","../experiment/attack_pcap/slowloris_flow.csv", "../experiment/pert_vis/diff.png")
