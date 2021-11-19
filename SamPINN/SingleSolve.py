from ImportFile import *

torch.manual_seed(42)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sampling_seed, N_coll, N_u, N_int, folder_path, validation_size\
    , network_properties, retrain, shuffle = initialize_inputs(len(sys.argv))

[Ec, max_iter, extrema, input_dimensions, output_dimension, space_dimensions, \
                time_dimension, parameter_dimensions, N_u_train, N_coll_train, \
                N_int_train, N_train, N_b_train, N_i_train] = \
                setupEquationClass(N_coll, N_u, N_int,validation_size,network_properties)

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
    print(bcolors.WARNING + "WARNING: you set max_iter=1 and epochs=" + str(
        network_properties["epochs"]) + " with a LBFGS optimizer.\n"
                                        "This will work but it is not efficient in full batch mode. Set max_iter = " + str(
        network_properties["epochs"]) +
          " and epochs=1. instead" + bcolors.ENDC)

if batch_dim == "full":
    batch_dim = N_train

# ###################################################################################
# Dataset Creation
training_set_class = createDataSet(Ec, N_coll_train, N_b_train, N_i_train, N_int_train,
                                   batch_dim, sampling_seed, shuffle)

# ####################################################################################
# Model Creation
additional_models = None
model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension,
              network_properties=network_properties)

# ####################################################################################
# Weights Initialization
torch.manual_seed(retrain)
init_xavier(model)

# ####################################################################################
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

# #####################################################################################
# Plotting ang Assessing Performance
images_path = folder_path + "/Images"
model_path = folder_path + "/TrainedModel"

if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
    os.mkdir(folder_path)
    os.mkdir(images_path)
    os.mkdir(model_path)

L2_test, rel_L2_test = Ec.compute_generalization_error(model, extrema, images_path)
Ec.plotting(model, images_path, extrema, None)

eigenval = model.lam.detach().numpy()[0]
print("Eigenvalue : {}".format(eigenval))
print("{} iterations".format(model.iter))

data = [N_u_train, N_coll_train, N_int_train, validation_size, end, L2_test,
        rel_L2_test, final_error_train, error_vars, error_pde]
dump_to_file(model, model_path, folder_path, network_properties, data)
