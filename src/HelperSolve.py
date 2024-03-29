import torch
import sys
import json
import os
from ModClass import Pinns
import matplotlib.pyplot as plt

from ImportFile import *

def initialize_inputs(len_sys_argv,HYPER_SOLVE = False):
    """
    Initialize the computation parameters.
    :param len_sys_argv:
    :param HYPER_SOLVE:
    :return sampling_seed_, n_coll_, n_u_, n_int_, folder_path_, validation_size_,
        network_properties_, retrain_, shuffle_:
    """
    if len_sys_argv == 1:

        # Random Seed for sampling the dataset
        sampling_seed_ = 200

        # Number of training+validation points
        n_coll_ = 2000
        n_u_ = 300
        n_int_ = 0

        network_properties_ = dict()
        # Additional Info
        folder_path_ = "SingleSol"
        validation_size_ = 0.0  # useless

        #takes into account both hyperparam optimization with ray tune and single function
        #solve
        if not HYPER_SOLVE:
            network_properties_ = {
                "hidden_layers": 4,
                "neurons": 20,
                "residual_parameter": 1000,
                "kernel_regularizer": 1.0,
                "normalization_parameter": 10000,
                "othogonality_parameter": 100,
                "regularization_parameter": 0.0,
                "batch_size": (n_coll_ + n_u_ + n_int_),
                "epochs": 1,
                "max_iter": 100000,
                "activation": "snake",
                "optimizer": "LBFGS"  # ADAM
            }
        else:
            from ray import tune
            network_properties_ = {
                "hidden_layers": tune.grid_search([4]),
                "neurons": tune.grid_search([20]),
                "residual_parameter": tune.grid_search([1,100,10000]),
                "kernel_regularizer": tune.grid_search([1.0]),
                "normalization_parameter" : tune.grid_search([10000,100000,1000000]),
                "othogonality_parameter": tune.grid_search([1,100,1000]),
                "regularization_parameter": tune.grid_search([0.0]),
                "batch_size": tune.grid_search([(n_coll_ + n_u_ + n_int_)]),
                "epochs": tune.grid_search([1]),
                "max_iter": tune.grid_search([100000]),
                "activation": tune.grid_search(["snake"]),
                "optimizer": tune.grid_search(["LBFGS"]),
                "id_retrain": tune.grid_search([1,2])
            }

        #pytorch seed
        retrain_ = 4

        # = true with batches
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

    return sampling_seed_, n_coll_, n_u_, n_int_, folder_path_, validation_size_, \
            network_properties_, retrain_, shuffle_

#for multi solve
def load_previous_solutions(dir,input_dimension, output_dimension,network_properties):
    """
    Loads the previously computed solutions so that a loss with an orthogonality
    term can be used and force the solution to another eigenvalue.
    :param dir:
    :param input_dimension:
    :param output_dimension:
    :param network_properties:
    :return solutions neural nets dictionary with eigenvalue keys:
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

def dump_to_file(model, model_path, folder_path, network_properties, data):
    """
    Output the training results in a folder.
    :param model:
    :param model_path:
    :param folder_path:
    :param network_properties:
    :param data:
    """
    torch.save(model, model_path + "/model.pkl")
    torch.save(model.state_dict(), model_path + "/model2.pkl")
    with open(model_path + os.sep + "Information.csv", "w") as w:
        keys = list(network_properties.keys())
        vals = list(network_properties.values())
        w.write(keys[0])
        for i in range(1, len(keys)):
            w.write("," + keys[i])
        w.write("\n")
        w.write(str(vals[0]))
        for i in range(1, len(vals)):
            w.write("," + str(vals[i]))

    with open(folder_path + '/InfoModel.txt', 'w') as file:
        file.write("Nu_train,"
                   "Nf_train,"
                   "Nint_train,"
                   "validation_size,"
                   "train_time,"
                   "L2_norm_test,"
                   "rel_L2_norm,"
                   "error_train,"
                   "error_vars,"
                   "error_pde\n")
        [N_u_train, N_coll_train, N_int_train, validation_size, end, L2_test,
            rel_L2_test, final_error_train, error_vars, error_pde] = data
        file.write(str(N_u_train) + "," +
                   str(N_coll_train) + "," +
                   str(N_int_train) + "," +
                   str(validation_size) + "," +
                   str(end) + "," +
                   str(L2_test) + "," +
                   str(rel_L2_test) + "," +
                   str(final_error_train) + "," +
                   str(error_vars) + "," +
                   str(error_pde))

def dump_to_file_eig(eigenvalue,model,path):
    """
    Saves the trained model corresponding to one eigenvalue in a folder as .pkl.
    The name of the file is the found eigenvalue.
    :param eigenvalue:
    :param model:
    :param path:
    """
    torch.save(model.state_dict(), path + "/" + str(eigenvalue) + ".pkl")

def remove_from_file_eig(eigenvalue,path):
    file = path + "/" + str(eigenvalue) + ".pkl"
    os.remove(file)


def multiPlot1D(x,input_dimension, output_dimension,network_properties):
    """
    Plots the output of the MultiSolve function by going through every models
    in the Solved folder.
    :param x:
    :param input_dimension:
    :param output_dimension:
    :param network_properties:
    """
    path_to_solved = os.getcwd() + "/Solved"

    plt.figure(figsize=(14, 10), dpi=120)

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

def multiPlot1DHYPER(x,errors_model,EquationClass):
    """
    Plots the output of the MultiSolve function by going through every models
    in the error_model dictionary. Overloaded for HYPER_SOLVE mode.
    :param x: Values on which the model needs to be evaluated.
    :param errors_model: errors_model[eigenvalue] =
    (model,final_error_train, error_vars, error_pde, L2_test_error, rel_L2_test_error)
    """

    #plot color generator
    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)

    plt.figure(figsize=(14, 10), dpi=120)

    colors = get_cmap(len(errors_model))
    i = 0
    for key,value in errors_model.items():
        model = value[0]
        model.eval()
        eigen = key
        with torch.no_grad():
            x_t = torch.tensor(x, dtype=torch.float32).reshape(-1, 1)
            pred = model(x_t)
            pred = pred.numpy()
            exact = EquationClass.exact(x_t,lam = round(float(eigen) * 2) / 2).numpy().reshape(-1,1)

            #compare the sign of the solutions to maybe flip --> assumes the prediction is close to the exact value
            L2_test_1 = np.mean((exact - pred) ** 2)
            L2_test_2 = np.mean((exact + pred) ** 2)
            if L2_test_2 < L2_test_1 : exact = -1.0*exact

            plt.plot(x,pred,label = "lam = " + str(eigen),c = colors(i))
            plt.plot(x,exact,label = "lam = " + str(round(float(eigen) * 2) / 2),c = colors(i))

        i = i+1
    plt.legend(loc = 'upper right')
    plt.savefig("multiPlot1D.png")


def setupEquationClass(N_coll, N_u, N_int,validation_size,network_properties):
    """
    Manages the equation type, dimensions and points generation.
    :param N_coll:
    :param N_u:
    :param N_int:
    :param validation_size:
    :param network_properties:
    :return Ec, max_iter, extrema, input_dimensions, output_dimension, space_dimensions,
                time_dimension, parameter_dimensions, N_u_train, N_coll_train,
                N_int_train, N_train, N_b_train, N_i_train:
    """
    Ec = EquationClass()
    if Ec.extrema_values is not None:
        extrema = Ec.extrema_values
        space_dimensions = Ec.space_dimensions
        time_dimension = Ec.time_dimensions
        parameter_dimensions = Ec.parameter_dimensions

        print(space_dimensions, time_dimension, parameter_dimensions)
    else:
        print("Using free shape. Make sure you have the functions:")
        print("     - add_boundary(n_samples)")
        print("     - add_collocation(n_samples)")
        print("in the Equation file")

        extrema = None
        space_dimensions = Ec.space_dimensions
        time_dimension = Ec.time_dimensions
    try:
        parameters_values = Ec.parameters_values
        parameter_dimensions = parameters_values.shape[0]
    except AttributeError:
        print("No additional parameter found")
        parameters_values = None
        parameter_dimensions = 0

    input_dimensions = parameter_dimensions + time_dimension + space_dimensions
    output_dimension = Ec.output_dimension
    mode = "none"
    if network_properties["epochs"] != 1:
        max_iter = 1
    else:
        max_iter = network_properties["max_iter"]

    N_u_train = int(N_u * (1 - validation_size))
    N_coll_train = int(N_coll * (1 - validation_size))
    N_int_train = int(N_int * (1 - validation_size))
    N_train = N_u_train + N_coll_train + N_int_train

    if space_dimensions > 0:
        N_b_train = int(N_u_train / (4 * space_dimensions))
        # N_b_train = int(N_u_train / (1 + 2 * space_dimensions))
    else:
        N_b_train = 0
    if time_dimension == 1:
        N_i_train = N_u_train - 2 * space_dimensions * N_b_train
        # N_i_train = N_u_train - N_b_train*(2 * space_dimensions)
    elif time_dimension == 0:
        N_b_train = int(N_u_train / (2 * space_dimensions))
        N_i_train = 0
    else:
        raise ValueError()

    return Ec, max_iter, extrema, input_dimensions, output_dimension, space_dimensions, \
                time_dimension, parameter_dimensions, N_u_train, N_coll_train, \
                N_int_train, N_train, N_b_train, N_i_train

def createDataSet(Ec, N_coll_train, N_b_train, N_i_train, N_int_train, batch_dim,
                  sampling_seed, shuffle):
    """
    Creates the PyTorch compatible dataset.
    :param Ec:
    :param N_coll_train:
    :param N_b_train:
    :param N_i_train:
    :param N_int_train:
    :param batch_dim:
    :param sampling_seed:
    :param shuffle:
    :return training_set_class:
    """

    training_set_class = DefineDataset(Ec, N_coll_train, N_b_train, N_i_train, N_int_train,
                            batches=batch_dim, random_seed=sampling_seed, shuffle=shuffle)
    training_set_class.assemble_dataset()

    return training_set_class

def printRecap(errors_model):
    """
    Prints a recapitulation table
    :param errors_model: The dictionary from MultiSolve.py containing the eigensolution
    and eigenvalue with some error values.
    :return:
    """
    from tabulate import tabulate
    if(errors_model[list(errors_model.keys())[0]][5] is not None):
        table = [[str(round(key,3)),'{:.3e}'.format(value[1]),'{:.3e}'.format(value[2]),'{:.3e}'.format(value[3]),
                  '{:.3e}'.format(value[4]),'{:.3e}'.format(value[5])] for key,value in errors_model.items()]
        print(tabulate(table, headers=["Eigenvalue","Total Loss", "Boundary Loss","PDE + Norm + Orth Loss",
                                       "L2 Error","Relative L2 Error"],disable_numparse=True))
    else:
        table = [[str(round(key, 3)), '{:.3e}'.format(value[1]), '{:.3e}'.format(value[2]), '{:.3e}'.format(value[3])] for key, value in errors_model.items()]
        print(tabulate(table, headers=["Eigenvalue", "Total Loss", "Boundary Loss", "PDE + Norm + Orth Loss"
                                       ], disable_numparse=True))


