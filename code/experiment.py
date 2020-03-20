from input_utils import *
from train import *
from eval import *
if __name__ == '__main__':
    dataset_name="top3"
    model_path="../models/3layer_top3"

    # dr=DataReader("../IDS2017_ML", 78,0.2, 0.2, attack_type=["BENIGN","DoS Hulk","PortScan"], dataset_name=dataset_name)
    # dr.generate_dataframes()
    # dr.write_to_csv()
    # dr.dataset_statistics()
    # dr.start()
    # train_normal_network(dataset_name, model_path, batch_size=1024, epochs=10)
    evaluate_network(dataset_name, model_path, "3layer_top3", batch_size=1024)
