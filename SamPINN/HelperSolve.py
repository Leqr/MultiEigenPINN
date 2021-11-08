import torch
import sys
import json
import os
from ModClass import Pinns
import matplotlib.pyplot as plt

def initialize_inputs(len_sys_argv):
    if len_sys_argv == 1:

        # Random Seed for sampling the dataset
        sampling_seed_ = 128

        # Number of training+validation points
        n_coll_ = 2100
        n_u_ = 2
        n_int_ = 0

        # Additional Info
        folder_path_ = "EigenLapl1dTest"
        validation_size_ = 0.0  # useless
        network_properties_ = {
            "hidden_layers": 4,
            "neurons": 20,
            "residual_parameter": 1,
            "kernel_regularizer": 1.0,
            "normalization_parameter" : 100000,
            "othogonality_parameter" : 100,
            "regularization_parameter": 0.0,
            "batch_size": (n_coll_ + n_u_ + n_int_),
            "epochs": 1,
            "max_iter": 100000,
            "activation": "snake",
            "optimizer": "LBFGS"  # ADAM
        }
        retrain_ = 32

        shuffle_ = False

    else:
        print(sys.argv)
        # Random Seed for sampling the dataset
        sampling_seed_ = int(sys.argv[1])

        # Number of training+validation points
        n_coll_ = int(sys.argv[2])
        n_u_ = int(sys.argv[3])
        n_int_ = int(sys.argv[4])

        # Additional Info
        folder_path_ = sys.argv[5]
        validation_size_ = float(sys.argv[6])
        network_properties_ = json.loads(sys.argv[7])
        retrain_ = sys.argv[8]
        if sys.argv[9] == "false":
            shuffle_ = False
        else:
            shuffle_ = True

    return sampling_seed_, n_coll_, n_u_, n_int_, folder_path_, validation_size_, network_properties_, retrain_, shuffle_

#for multi solve
def load_previous_solutions(dir,input_dimension, output_dimension,
                  network_properties):
    """
    Loads the previously computed solutions so that a loss with an orthogonality
    term can be used and force the solution to another eigenvalue.
    """
    sols = dict()
    for subdir, dirs, files in os.walk(dir):
        for file in files:
            path_to_file = dir+"/"+file
            eigen = os.path.splitext(file)[0]
            extension = os.path.splitext(file)[1]
            if extension == ".pkl":
                model = Pinns(input_dimension=input_dimension, output_dimension=output_dimension,
                  network_properties=network_properties)
                model.load_state_dict(torch.load(path_to_file))
                sols[float(eigen)] = model
    return sols

def dump_to_file_eig(eigenvalue,model,path):
    torch.save(model.state_dict(), path + "/" + str(eigenvalue) + ".pkl")

def multiPlot1D(x,input_dimension, output_dimension,network_properties):
    """
    Plots the output of the MultiSolve function by going through every models
    in the Solved folder
    """
    path_to_solved = os.getcwd() + "/Solved"

    plt.figure(figsize=(14, 10), dpi=120)

    #for the laplacian eigenvalue problem with known eigenvalue
    precision = 0.05

    for subdir, dirs, files in os.walk(path_to_solved):
        for file in files:
            # go through every model
            split = os.path.splitext(file)
            eigen = split[0]
            if split[1] != "" :
                eigen_float = float(eigen)
            extension = split[1]
            path_to_file = path_to_solved + "/" + file
            if extension == ".pkl":
                if abs(eigen_float - round(eigen_float * 2) / 2) < precision:
                    model = Pinns(input_dimension=input_dimension, output_dimension=output_dimension,
                                  network_properties=network_properties)
                    model.load_state_dict(torch.load(path_to_file))
                    model.eval()
                    with torch.no_grad():
                        x_t = torch.tensor(x,dtype = torch.float32).reshape(-1,1)
                        pred = model(x_t)
                        pred = pred.numpy()
                        plt.plot(x,pred,label = "lam = " + str(eigen))
            plt.legend()
            plt.savefig("multiPlot1D.png")
