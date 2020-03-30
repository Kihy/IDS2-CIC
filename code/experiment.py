from input_utils import *
from train import *
from eval import *
from adversarial import *
from predict import *
from vis_utils import *
if __name__ == '__main__':
    dataset_name="all"
    model_type="3layer"
    model_path="../models/{}_{}".format(model_type,dataset_name)
    data_path="../IDS2018_ML"
    percent_theta=0.1
    target=0
    attack_type=["Benign","SSH-Patator","FTP-Patator"]
    ingore_file=["Tuesday-20-02-2018_TrafficForML_CICFlowMeter.csv"]
    include_file=["Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv","Friday-23-02-2018_TrafficForML_CICFlowMeter.csv"]
    dr=DataReader(data_path, 0.2, 0.2, dataset_name=dataset_name, files=include_file, ignore=True)
    # dr.generate_dataframes()
    # dr.write_to_csv()
    # dr.dataset_statistics()
    dr.start()
    # train_normal_network(dataset_name, model_path, batch_size=1024, epochs=10)
    # evaluate_network(dataset_name, model_path, "{}_{}".format(model_type,dataset_name), batch_size=1024)
    # adversarial_generation(dataset_name, model_path, target, "val", num_samples=1024, theta=0.01)
    # format_converter("../pcap_data/patator_no_response.pcap_Flow.csv", "../experiment/column_map.csv")
    # predict_sample("patator_no_response.pcap_Flow.csv", model_path, dataset_name)
    # vis_original_input("../IDS2017_ML/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",["BENIGN","PortScan"])
    # vis_attack_distribution("../data/patator/patator_no_response.csv","../experiment/attack_pcap/cic_patator.png", attack=1, ignore_label=False)
    # vis_attack_distribution("../experiment/attack_pcap/patator_no_response.pcap_Flow.csv","../experiment/attack_pcap/patator_no_response.png",ignore_label=True)
