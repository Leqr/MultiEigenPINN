import os
import sys
import itertools
import subprocess
from ImportFile import *

rs = 32
N_coll = int(sys.argv[1])
N_u = int(sys.argv[2])
N_int = int(sys.argv[3])
folder_name = sys.argv[4]
validation_size = 0.0
network_properties = {
    "hidden_layers": [4],
    "neurons": [20],
    "residual_parameter": [1,10,100,1000],
    "kernel_regularizer": [1,10,100,1000],
    "regularization_parameter": [0],
    "batch_size": [(N_coll + N_u + N_int)],
    "epochs": [1],
    "max_iter": [10000],
    "activation": ["snake"],
    "optimizer": ["LBFGS"]
}
shuffle = "true"
cluster = sys.argv[5]
GPU = "None"  # GPU="GeForceGTX1080"  # GPU = "GeForceGTX1080Ti"  # GPU = "TeslaV100_SXM2_32GB"
n_retrain = 2

if not os.path.isdir(folder_name):
    os.mkdir(folder_name)
settings = list(itertools.product(*network_properties.values()))

i = 0
for setup in settings:
    print(setup)

    folder_path = folder_name + "/Setup_" + str(i)
    print("###################################")
    setup_properties = {
        "hidden_layers": setup[0],
        "neurons": setup[1],
        "residual_parameter": setup[2],
        "kernel_regularizer": setup[3],
        "regularization_parameter": setup[4],
        "batch_size": setup[5],
        "epochs": setup[6],
        "max_iter": setup[7],
        "activation": setup[8],
        "optimizer": setup[9]
    }

    arguments = list()
    arguments.append(str(rs))
    arguments.append(str(N_coll))
    arguments.append(str(N_u))
    arguments.append(str(N_int))
    arguments.append(str(folder_path))
    arguments.append(str(validation_size))
    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        arguments.append("\'" + str(setup_properties).replace("\'", "\"") + "\'")
    else:
        arguments.append(str(setup_properties).replace("\'", "\""))
    arguments.append(str(shuffle))
    arguments.append(str(cluster))
    arguments.append(str(GPU))
    arguments.append(str(n_retrain))

    if sys.platform == "linux" or sys.platform == "linux2" or sys.platform == "darwin":
        if cluster == "true":
            string_to_exec = "bsub python3 single_retraining.py "
        else:
            string_to_exec = "python3 single_retraining.py "
        for arg in arguments:
            string_to_exec = string_to_exec + " " + arg
        print(string_to_exec)
        os.system(string_to_exec)
    i = i + 1
