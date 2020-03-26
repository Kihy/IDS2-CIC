from input_utils import *
from train import *
from eval import *
from adversarial import *
from predict import *
from vis_utils import *
if __name__ == '__main__':
    dataset_name="patator"
    model_type="3layer"
    model_path="../models/{}_{}".format(model_type,dataset_name)
    percent_theta=0.1
    target=0
    # dr=DataReader("../IDS2017_ML", 78,0.2, 0.2, attack_type=["BENIGN","SSH-Patator","FTP-Patator"], dataset_name=dataset_name)
    # dr.generate_dataframes()
    # dr.write_to_csv()
    # dr.dataset_statistics()
    # dr.start()
    # train_normal_network(dataset_name, model_path, batch_size=1024, epochs=10)
    # evaluate_network(dataset_name, model_path, "{}_{}".format(model_type,dataset_name), batch_size=1024)
    # adversarial_generation(dataset_name, model_path, target, "val", num_samples=1024, theta=0.01)
    # flow_to_ml_converter("../pcap_data", "../experiment/column_map.csv")
    predict_sample("ftp patator.pcap_Flow.csv", model_path, dataset_name)
    # vis_original_input("../IDS2017_ML/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",["BENIGN","PortScan"])
    # vis_attack_distribution("../data/patator/test.csv","../experiment/attack_pcap/data_ftpp.png", attack=1, ignore_label=False)
    # vis_attack_distribution("../experiment/attack_pcap/ftp patator.pcap_Flow.csv","../experiment/attack_pcap/real_ftpp.png",ignore_label=True)
