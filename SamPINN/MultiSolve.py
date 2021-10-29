from ImportFile import *

#create folder that will store the eigenvalue and the solution network
folder_path = "Solved"
if not(os.path.exists(folder_path) and os.path.isdir(folder_path)):
    os.mkdir(folder_path)

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

if network_properties["optimizer"] == "LBFGS" and network_properties["epochs"] != 1 and \
        network_properties["max_iter"] == 1 and (batch_dim == "full" or batch_dim == N_train):
    print(bcolors.WARNING + "WARNING: you set max_iter=1 and epochs=" + str(network_properties["epochs"]) + " with a LBFGS optimizer.\n"
        "This will work but it is not efficient in full batch mode. Set max_iter = " + str(network_properties["epochs"]) +
        " and epochs=1. instead" + bcolors.ENDC)

if batch_dim == "full":
    batch_dim = N_train

# ###################################################################################
# Dataset Creation
training_set_class = DefineDataset(Ec, N_coll_train, N_b_train, N_i_train, N_int_train,
                                   batches=batch_dim, random_seed=sampling_seed, shuffle=shuffle)
training_set_class.assemble_dataset()

n_replicates = 10

#path where the new solutions will be added
solved_path = os.getcwd() + "/Solved"

for i in range(n_replicates):
    print("Computation Number : ",i)
    sols = None
    if i != 0:
        #load the already computed orthogonal solutions
        sols = load_previous_solutions(solved_path,input_dimension=input_dimensions, output_dimension=output_dimension,
                  network_properties=network_properties)

    additional_models = None
    model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension,
                  network_properties=network_properties,other_networks=sols)

    torch.manual_seed(retrain)
    init_xavier(model)

    if network_properties["optimizer"] == "LBFGS":
        optimizer_LBFGS = optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter, max_eval=50000, history_size=100,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=1.0 * np.finfo(float).eps)
        model.optimizer = optimizer_LBFGS
    elif network_properties["optimizer"] == "ADAM":
        optimizer_ADAM = optim.Adam(model.parameters(), lr=0.0005)
        model.optimizer = optimizer_ADAM
    else:
        raise ValueError()

    start = time.time()
    print("Fitting Model")
    model.to(Ec.device)
    model.train()

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

    images_path = folder_path + "/Images"
    model_path = folder_path + "/TrainedModel"

    if not(os.path.exists(folder_path) and os.path.isdir(folder_path)):
        os.mkdir(folder_path)
        os.mkdir(images_path)
        os.mkdir(model_path)

    L2_test, rel_L2_test = Ec.compute_generalization_error(model, extrema, images_path)
    Ec.plotting(model, images_path, extrema, None)

    eigenval = model.lam.detach().numpy()[0]
    print("Eigenvalue : {}".format(eigenval))

    dump_to_file_eig(eigenval, model, solved_path)
