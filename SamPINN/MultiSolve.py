
from ImportFile import *
import itertools
from functools import partial

# manage the hyperparameter optimization mode
HYPER_SOLVE = False

# uses the previous network to compute a new eigenvalue and eigenfunction (transfer learning)
TRANSFER_LEARNING = True

if HYPER_SOLVE:
    from ray import tune
    import ray
    #reduces the output of warnings
    ray.init(log_to_driver=False) #,local_mode=True) #sequential run param

# create folder that will store the eigenvalue and the solution network
folder_path = "Solved"
if not (os.path.exists(folder_path) and os.path.isdir(folder_path)):
        #careful as this clears the previously found solutions
        os.mkdir(folder_path)
else :
    os.system("rm -r Solved")
    os.mkdir(folder_path)

sampling_seed, N_coll, N_u, N_int, folder_path, validation_size, network_properties, retrain, shuffle = \
    initialize_inputs(len(sys.argv), HYPER_SOLVE=HYPER_SOLVE)

# unfold the network properties into single setup
if HYPER_SOLVE:
    from ray.tune import CLIReporter
    reporter = CLIReporter(max_progress_rows=10,print_intermediate_tables = False,
                           metric=None,mode=None,max_report_frequency=5)
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

# path where the new solutions will be added
solved_path = os.getcwd() + "/Solved"

def training_function(config, params):
    """
    Training function used by ray tune for the hyperparameter optimization
    :param config:
    :param params:
    :return errors:
    """
    #get the parameters
    solved_path = params["solved_path"]
    input_dimensions= params["input_dimensions"]
    output_dimension= params["output_dimension"]
    max_iter= config["max_iter"]
    Ec= params["Equation"]
    training_set_class= params["training_set_class"]
    sols = None
    i = params["i"]
    HYPER_SOLVE = params["HYPER_SOLVE"]
    TRANSFER_LEARNING = params["TRANSFER_LEARNING"]
    errors_model = params["errors_model"]

    if len(errors_model) != 0:
        # load the already computed orthogonal solutions
        sols = load_previous_solutions(solved_path, input_dimension=input_dimensions,
                                       output_dimension=output_dimension,
                                       network_properties=config)

    if HYPER_SOLVE:
        #in this case the id_retrain parameter of config gives the number of
        #retraining needed, the random seed is then reported to tune for reproducibility
        retrain = random.randint(1, 10000)
        torch.manual_seed(retrain)

    model = None
    # deal with the model initilisation given that we are doing transfer learning or not
    if TRANSFER_LEARNING:
        if len(errors_model) != 0:
            # randomly take one of the already found solution as starting network
            eigenvalue, model = random.choice(list(sols.items()))

            # reset the model
            model.reset()

            # assign the previously found solutions
            model.other_networks = sols

            #add noise to the neural net weights to help get out of the previous local minima
            model.noise()

    if not TRANSFER_LEARNING or (TRANSFER_LEARNING and len(errors_model) == 0):
        model = Pinns(input_dimension=input_dimensions, output_dimension=output_dimension,
                      network_properties=config, other_networks=sols)
        init_xavier(model)

    """
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    """

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
                    eigen = model.lam.detach().numpy()[0],
                    torch_seed = retrain)
    else: return errors,model

n_replicates = 5

torch.manual_seed(retrain)

# keep in memory the erros of the models so that we can discard regarding to
# the error of the model for the same eigenvalue
errors_model = dict()

for i in range(n_replicates):
    print("Computation Number : ", i)

    #parameters packed in a dict to enable ray tune
    params_training_function = {
        "i": i,
        "solved_path": solved_path,
        "input_dimensions": input_dimensions,
        "output_dimension": output_dimension,
        "Equation": Ec,
        "training_set_class": training_set_class,
        "HYPER_SOLVE": HYPER_SOLVE,
        "TRANSFER_LEARNING": TRANSFER_LEARNING,
        "errors_model": errors_model
    }

    start = time.perf_counter()
    print("Fitting Model")
    if HYPER_SOLVE:
        analysis = tune.run(partial(training_function,params = params_training_function),
                            config=network_properties,metric = 'loss_tot', mode = 'min',
                            verbose=2,
                            raise_on_failed_trial = False
                            )
        best_trial = analysis.best_trial
        print("Best trial config: {}".format(best_trial.config))

        final_error_train = ((10 ** best_trial.last_result["loss_tot"]) ** 0.5)
        error_vars = float(best_trial.last_result["loss_vars"])
        error_pde = float(best_trial.last_result["loss_pde"])

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
    end = time.perf_counter() - start
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



        #iterate through the previously computed solutions to check if we already found
        #this eigenvalue
        match = False

        #temporary dictionary in order to update in the loop
        temp_errors_model = errors_model.copy()

        #compute the models generalization error (return None if no analyticalm solution)
        L2_test, rel_L2_test = Ec.compute_generalization_error(model,extrema)

        for key,value in errors_model.items():
            if np.isclose(key,eigenval, 0.1):
                if errors_model[key][1] > final_error_train:
                    #keep the best solution
                    remove_from_file_eig(key,solved_path)
                    del temp_errors_model[key]

                    #save the best solution for this eigenvalue
                    dump_to_file_eig(eigenval, model, solved_path)
                    temp_errors_model[eigenval] =  (model,final_error_train, error_vars, error_pde, L2_test, rel_L2_test)

                match = True

        errors_model = temp_errors_model.copy()

        if not match:
            #no match --> new eigenvalue --> save the model
            dump_to_file_eig(eigenval, model, solved_path)
            errors_model[eigenval] =  (model,final_error_train, error_vars, error_pde, L2_test, rel_L2_test)


# plot all the solutions on one figure for 1D problems
if Ec.space_dimensions == 1:
    x = np.linspace(Ec.extrema_values[0][0], Ec.extrema_values[0][1], 1000)
    multiPlot1D(x, input_dimensions, output_dimension, network_properties)

if True :
    printRecap(errors_model)

if HYPER_SOLVE:
    #remove log dir to free up space
    os.system("rm -r $HOME/ray_results/")
