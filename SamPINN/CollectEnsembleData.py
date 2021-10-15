from ImportFile import *

torch.nn.Module.dump_patches = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def select_over_retrainings(folder_path, selection="error_train", mode="min", compute_std=False, compute_val=False, rs_val=0):
    retrain_models = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

    models_list = list()
    for retraining in retrain_models:
        # print("Looking for ", retraining)
        rs = int(folder_path.split("_")[-1])
        retrain_path = folder_path + "/" + retraining
        number_of_ret = retraining.split("_")[-1]

        if os.path.isfile(retrain_path + "/InfoModel.txt"):
            models = pd.read_csv(retrain_path + "/InfoModel.txt", header=0, sep=",")
            models["retraining"] = number_of_ret
            models["selection"] = models["error_vars"].values[0] + models["error_pde"].values[0]

            models_list.append(models)
            # print(models)

        else:
            print("No File Found")

    retraining_prop = pd.concat(models_list, ignore_index=True)
    print(retraining_prop)
    retraining_prop = retraining_prop.sort_values(selection)
    # print("#############################################")
    # print(retraining_prop)
    # print("#############################################")
    # quit()
    if mode == "min":
        # print("#############################################")
        # print(retraining_prop.iloc[0])
        # print("#############################################")
        return retraining_prop.iloc[0]
    else:
        retraining = retraining_prop["retraining"].iloc[0]
        # print("#############################################")
        # print(retraining_prop.mean())
        # print("#############################################")
        retraining_prop = retraining_prop.mean()
        retraining_prop["retraining"] = retraining
        return retraining_prop


np.random.seed(42)

base_path_list = ["Test"]

for base_path in base_path_list:
    print("#################################################")
    print(base_path)

    b = False
    compute_std = False
    directories_model = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    sensitivity_df = pd.DataFrame(columns=["batch_size",
                                           "regularization_parameter",
                                           "kernel_regularizer",
                                           "neurons",
                                           "hidden_layers",
                                           "residual_parameter",
                                           "L2_norm_test",
                                           "error_train",
                                           "error_val",
                                           "error_test"])
    # print(sensitivity_df)

    selection_criterion = "selection"
    eval_criterion = "rel_L2_norm"

    Nu_list = []
    Nf_list = []

    L2_norm = []
    criterion = []
    best_retrain_list = []
    list_models_setup = list()

    for subdirec in directories_model:
        model_path = base_path

        sample_path = model_path + "/" + subdirec
        retrainings_fold = [d for d in os.listdir(sample_path) if os.path.isdir(os.path.join(sample_path, d))]

        retr_to_check_file = None
        for ret in retrainings_fold:
            if os.path.isfile(sample_path + "/" + ret + "/TrainedModel/Information.csv"):
                retr_to_check_file = ret
                break

        setup_num = int(subdirec.split("_")[1])
        if retr_to_check_file is not None:
            info_model = pd.read_csv(sample_path + "/" + retr_to_check_file + "/TrainedModel/Information.csv", header=0, sep=",")
            best_retrain = select_over_retrainings(sample_path, selection=selection_criterion, mode="min", compute_std=compute_std, compute_val=False, rs_val=0)
            info_model["error_train"] = best_retrain["error_train"]
            info_model["train_time"] = best_retrain["train_time"]
            info_model["selection"] = best_retrain["selection"]
            info_model["error_val"] = 0
            info_model["error_test"] = 0
            info_model["L2_norm_test"] = best_retrain["L2_norm_test"]
            info_model["rel_L2_norm"] = best_retrain["rel_L2_norm"]
            info_model["setup"] = setup_num
            info_model["retraining"] = best_retrain["retraining"]

            if info_model["batch_size"].values[0] == "full":
                info_model["batch_size"] = best_retrain["Nu_train"] + best_retrain["Nf_train"]

            sensitivity_df = sensitivity_df.append(info_model, ignore_index=True)
        else:
            print(sample_path + "/TrainedModel/Information.csv not found")

    sensitivity_df = sensitivity_df.sort_values(selection_criterion)
    best_setup = sensitivity_df.iloc[0]
    best_setup.to_csv(base_path + "/best.csv", header=0, index=False)
    # print(sensitivity_df)
    print("Best Setup:", best_setup["setup"])
    print(best_setup)

    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.scatter(sensitivity_df[selection_criterion], sensitivity_df[eval_criterion])
    plt.xlabel(r'$\varepsilon_T$')
    plt.ylabel(r'$\varepsilon_G$')
    plt.savefig(base_path + "/et_vs_eg.png", dpi=400)

    total_list = list()

    var_list = ["hidden_layers",
                "neurons",
                "residual_parameter",
                "kernel_regularizer",
                "regularization_parameter",
                "activation"]

    labels_list = ["hidden-layers",
                   "neurons",
                   "residual-parameter",
                   "kernel-regularizer",
                   "regularization-parameter",
                   "activation"]
    for var in var_list:
        print("=======================================================")
        print(var)
        params = sensitivity_df[var].values
        params = list(set(params))
        params.sort()
        df_param_list = list()
        for value in params:
            index_list_i = sensitivity_df.index[sensitivity_df[var] == value]
            new_df = sensitivity_df.loc[index_list_i]
            df_param_list.append(new_df)
        total_list.append(df_param_list)

    if not b:
        out_var_vec = list()
        out_var_vec.append(eval_criterion)

        for out_var in out_var_vec:
            for j in range(len(total_list)):
                print("-------------------------------------------------------")
                var = var_list[j]
                lab = labels_list[j]
                print(var)
                print(lab)
                # name = name_list[j]
                sens_list = total_list[j]
                Nf_dep_fig = plt.figure()
                axes = plt.gca()
                max_val = 0
                plt.grid(True, which="both", ls=":")
                for i in range(len(sens_list)):
                    df = sens_list[i]
                    print(df)

                    value = df[var].values[0]
                    label = lab + " = " + str(value).replace("_", "-")

                    sns.distplot(df[out_var], label=label, kde=True, hist=True, norm_hist=False, kde_kws={'shade': True, 'linewidth': 2})
                plt.xlabel(r'$\varepsilon_G$')

                plt.legend(loc=1)
                plt.savefig(base_path + "/Sensitivity_" + var + ".png", dpi=500)
