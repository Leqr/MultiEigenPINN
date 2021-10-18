from ImportFile import *

torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def dump_to_file():
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


def initialize_inputs(len_sys_argv):
    if len_sys_argv == 1:

        # Random Seed for sampling the dataset
        sampling_seed_ = 128

        # Number of training+validation points
        n_coll_ = 4096
        n_u_ = 2048
        n_int_ = 0

        # Additional Info
        folder_path_ = "EigenLapl1dTest"
        validation_size_ = 0.0  # useless
        network_properties_ = {
            "hidden_layers": 4,
            "neurons": 20,
            "residual_parameter": 0.1,
            "kernel_regularizer": 2,
            "regularization_parameter": 0,
            "batch_size": (n_coll_ + n_u_ + n_int_),
            "epochs": 1,
            "max_iter": 10000,
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


sampling_seed, N_coll, N_u, N_int, folder_path, validation_size, network_properties, retrain, shuffle = initialize_inputs(len(sys.argv))

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

print("\n######################################")
print("*******Domain Properties********")
print(extrema)

print("\n######################################")
print("*******Info Training Points********")
print("Number of train collocation points: ", N_coll_train)
print("Number of initial and boundary points: ", N_u_train, N_i_train, N_b_train)
print("Number of internal points: ", N_int_train)
print("Total number of training points: ", N_train)

print("\n######################################")
print("*******Network Properties********")
pprint.pprint(network_properties)
batch_dim = network_properties["batch_size"]

print("\n######################################")
print("*******Dimensions********")
print("Space Dimensions", space_dimensions)
print("Time Dimension", time_dimension)
print("Parameter Dimensions", parameter_dimensions)
print("\n######################################")

if network_properties["optimizer"] == "LBFGS" and network_properties["epochs"] != 1 and network_properties["max_iter"] == 1 and (batch_dim == "full" or batch_dim == N_train):
    print(bcolors.WARNING + "WARNING: you set max_iter=1 and epochs=" + str(network_properties["epochs"]) + " with a LBFGS optimizer.\n"
                                                                                                            "This will work but it is not efficient in full batch mode. Set max_iter = " + str(network_properties["epochs"]) + " and epochs=1. instead" + bcolors.ENDC)

if batch_dim == "full":
    batch_dim = N_train

# #############################################################################################################################################################
# Dataset Creation
training_set_class = DefineDataset(Ec, N_coll_train, N_b_train, N_i_train, N_int_train, batches=batch_dim, random_seed=sampling_seed, shuffle=shuffle)
training_set_class.assemble_dataset()

# #############################################################################################################################################################
# Model Creation
additional_models = None
model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension, network_properties=network_properties)

# #############################################################################################################################################################
# Weights Initialization
torch.manual_seed(retrain)
init_xavier(model)


# #############################################################################################################################################################
# Model Training
start = time.time()
print("Fitting Model")
model.to(Ec.device)
model.train()
optimizer_LBFGS = optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter, max_eval=50000, history_size=100,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
optimizer_ADAM = optim.Adam(model.parameters(), lr=0.0005)

if network_properties["optimizer"] == "LBFGS":
    model.optimizer = optimizer_LBFGS
elif network_properties["optimizer"] == "ADAM":
    model.optimizer = optimizer_ADAM
else:
    raise ValueError()

errors = fit(Ec, model, training_set_class, verbose=True)
end = time.time() - start
print("\nTraining Time: ", end)

model = model.eval()
final_error_train = float(((10 ** errors[0]) ** 0.5).detach().cpu().numpy())
error_vars = float((errors[1]).detach().cpu().numpy())
error_pde = float((errors[2]).detach().cpu().numpy())
print("\n################################################")
print("Final Training Loss:", final_error_train)
print("################################################")

# #############################################################################################################################################################
# Plotting ang Assessing Performance
images_path = folder_path + "/Images"
model_path = folder_path + "/TrainedModel"

if not(os.path.exists(folder_path) and os.path.isdir(folder_path)):
    os.mkdir(folder_path)
    os.mkdir(images_path)
    os.mkdir(model_path)

L2_test, rel_L2_test = Ec.compute_generalization_error(model, extrema, images_path)
Ec.plotting(model, images_path, extrema, None)

dump_to_file()
