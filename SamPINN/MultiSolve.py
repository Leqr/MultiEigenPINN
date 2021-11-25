from ImportFile import *
import itertools
from ray import tune
from functools import partial
import ray

#reduces the output of warnings
ray.init(log_to_driver=False) #,local_mode=True) #sequential run param

# manage the hyperparameter optimization mode
HYPER_SOLVE = True

# create folder that will store the eigenvalue and the solution network
folder_path = "Solved"
if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
    os.mkdir(folder_path)

sampling_seed, N_coll, N_u, N_int, folder_path, validation_size, network_properties, retrain, shuffle = \
    initialize_inputs(len(sys.argv), HYPER_SOLVE=HYPER_SOLVE)

# unfold the network properties into single setup
if HYPER_SOLVE:
    settings = list(itertools.product(*network_properties.values()))

[Ec, max_iter, extrema, input_dimensions, output_dimension, space_dimensions, \
 time_dimension, parameter_dimensions, N_u_train, N_coll_train, \
 N_int_train, N_train, N_b_train, N_i_train] = \
    setupEquationClass(N_coll, N_u, N_int, validation_size, network_properties)

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

if HYPER_SOLVE:
    batch_dim = network_properties["batch_size"]["grid_search"][0]
else:
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

n_replicates = 20

# path where the new solutions will be added
solved_path = os.getcwd() + "/Solved"

def training_function(config, params):
    """
    Training function used by ray tune for the hyperparameter optimization
    :param config:
    :return: loss:
    """
    #get the parameters
    solved_path = params["solved_path"]
    input_dimensions= params["input_dimensions"]
    output_dimension= params["output_dimension"]
    retrain= params["retrain"]
    max_iter= config["max_iter"]
    Ec= params["Equation"]
    training_set_class= params["training_set_class"]
    sols = None
    i = params["i"]
    HYPER_SOLVE = params["HYPER_SOLVE"]
    if i != 0:
        # load the already computed orthogonal solutions
        sols = load_previous_solutions(solved_path, input_dimension=input_dimensions,
                                       output_dimension=output_dimension,
                                       network_properties=config)
    model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension,
                  network_properties=config, other_networks=sols)
    torch.manual_seed(retrain)
    init_xavier(model)

    if config["optimizer"] == "LBFGS":
        optimizer_LBFGS = optim.LBFGS(model.parameters(), lr=0.8, max_iter=max_iter, max_eval=50000, history_size=100,
                                      line_search_fn="strong_wolfe",
                                      tolerance_change=1.0 * np.finfo(float).eps)
        model.optimizer = optimizer_LBFGS
    elif config["optimizer"] == "ADAM":
        optimizer_ADAM = optim.Adam(model.parameters(), lr=0.0005)
        model.optimizer = optimizer_ADAM
    else:
        raise ValueError()

    model.to(Ec.device)
    model.train()


    errors = fit(Ec, model, training_set_class, verbose = not HYPER_SOLVE)
    if HYPER_SOLVE:
        tune.report(loss_tot = float(errors[0].detach().cpu().numpy()),
                    loss_vars= float(errors[1].detach().cpu().numpy()),
                    loss_pde = float(errors[2].detach().cpu().numpy()),
                    model = model,
                    moditer = model.iter,
                    eigen = model.lam.detach().numpy()[0])
    else: return errors,model


for i in range(n_replicates):
    print("Computation Number : ", i)

    #parameters packed in a dict to enable ray tune
    params_training_function = {
        "i": i,
        "solved_path": solved_path,
        "input_dimensions": input_dimensions,
        "output_dimension": output_dimension,
        "retrain": retrain,
        "Equation": Ec,
        "training_set_class": training_set_class,
        "HYPER_SOLVE": HYPER_SOLVE
    }

    start = time.time()
    print("Fitting Model")
    if HYPER_SOLVE:
        analysis = tune.run(partial(training_function,params = params_training_function),
                            config=network_properties,metric = 'loss_pde', mode = 'min',
                            verbose = 1,
                            raise_on_failed_trial = False)
        best_trial = analysis.best_trial
        print("Best trial config: {}".format(best_trial.config))
        final_error_train = ((10 ** best_trial.last_result["loss_tot"]) ** 0.5)
        print("Best trial final total loss: {}".format(
            final_error_train))
        print("Best trial final pde loss: {}".format(
            best_trial.last_result["loss_pde"]))
        model = best_trial.last_result["model"]
    else:
        [errors,model] = training_function(config=network_properties,params=params_training_function)
        final_error_train = float(((10 ** errors[0]) ** 0.5).detach().cpu().numpy())
        error_vars = float((errors[1]).detach().cpu().numpy())
        error_pde = float((errors[2]).detach().cpu().numpy())
        print("\n################################################")
        print("Final Training Loss:", final_error_train)
        print("################################################")
    end = time.time() - start
    print("\nTraining Time: ", end)

    model = model.eval()

    # only keep solutions that have a low enough loss
    if final_error_train < 0.8:

        images_path = folder_path + "/Images"
        model_path = folder_path + "/TrainedModel"

        if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
            os.mkdir(folder_path)
            os.mkdir(images_path)
            os.mkdir(model_path)

        L2_test, rel_L2_test = Ec.compute_generalization_error(model, extrema, images_path)

        eigenval = model.lam.detach().numpy()[0]
        print("Eigenvalue : {}".format(eigenval))

        dump_to_file_eig(eigenval, model, solved_path)

# plot all the solutions on one figure for 1D problems
if Ec.space_dimensions == 1 and not HYPER_SOLVE:
    x = np.linspace(Ec.extrema_values[0][0], Ec.extrema_values[0][1], 1000)
    multiPlot1D(x, input_dimensions, output_dimension, network_properties)
