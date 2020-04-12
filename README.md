# IDS2-CIC-IDS

This repo contains code for experiments with CIC-IDS data.

### requirements
The dependencies are listed in requirements file. run
```
pip install -r requirements.txt
```

### Directory structure
```
├── code
│   ├── adversarial.py  # code for adversarial generation (JSMA)
│   ├── eval.py         # code to evaluate neural network (classification report)
│   ├── experiment.py   # main entry point to use all other codes
│   ├── input_utils.py  # code to generate train,test,val datasets
│   ├── network_graphs  # network graphs for neural networks
│   ├── predict.py      # predicts a sample with neural network
│   ├── __pycache__     # cache file can be ignored
│   ├── train.py        # trains the neural network  
│   └── vis_utils.py    # visualization tools
├── commands.sh         # used for my docker, you probably dont need it
├── custom_data         # data folder
├── data                # folder containing generated datasets
├── experiment          
│   ├── adv_data        # output of adversarial data generation
│   ├── attack_pcap     # adversarial pcap file
│   ├── column_map.csv  # mapping from flow format to ml format
│   ├── pert_vis        # visualization of perturbations
│   └── reports         # classification reports
├── IDS2017_ML          # ml dataset for IDS2017
├── IDS2018             # dataset for IDS2018
├── IDS2018_ML          # ml version of IDS2018
├── models              # generated neural network model
├── pcap_data           # raw pcap file from simulation
└── README.md           # this file
```

### Runnning the codes
Most of the functions available are listed in experiment.py, so just modify experiment.py and run
``` python3 experiment.py
```

#### experiment.py
To read data and split them into train, test, val dataset:

``` python3
dr=DataReader(data_path, 0.2, 0.2, dataset_name=dataset_name, files=include_file, ignore=True)
dr.generate_dataframes()
dr.write_to_csv() # time consuming operation, if you just want statistics dont call
dr.dataset_statistics()
dr.start() # runs all three of the above.
```

train, evaluate and predict network

```python3
train_normal_network(dataset_name, model_path, batch_size=16, epochs=10)
evaluate_network(dataset_name, model_path, "{}_{}".format(model_type,dataset_name), batch_size=1024)
predict_sample("slowloris_adv.pcap_Flow.csv", model_path, dataset_name)
```

generating adversarial samples

```python3
adversarial_generation(dataset_name, model_path, target, "train", num_samples=100, theta=0.01, fixed=fixed,scale=False)
```

convert formats from pcap_flow to ml format (the only difference is some columns are removed)

```python3
format_converter("../pcap_data/slowloris_adv.pcap_Flow.csv", "../experiment/column_map.csv", out_dir="../experiment/attack_pcap/", metadata=True, use_filename_as_label=False)
```

visualization of inputs

```python3
vis_original_input("../IDS2017_ML/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",["BENIGN","PortScan"])
vis_attack_distribution("../data/patator/val.csv","../experiment/attack_pcap/cic_patator.png", attack=1, ignore_label=False)
vis_attack_distribution("../experiment/attack_pcap/slowloris.pcap_Flow.csv","../experiment/attack_pcap/slowloris.png",ignore_label=True)
```

### Note
All files other than code is left empty intentionally to reduce the amount of data in the repo. This means some script may need custom creation of folders for it to work. All code should have sufficient documentation for it to be useful.
