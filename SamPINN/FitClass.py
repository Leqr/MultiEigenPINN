import torch
import torch.nn as nn
import numpy as np


class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, Ec, network, x_u_train, u_train, x_b_train, u_b_train, x_f_train, dataclass):
        lambda_residual = network.lambda_residual
        lambda_reg = network.regularization_param
        order_regularizer = network.kernel_regularizer

        u_pred_var_list = list()
        u_train_var_list = list()

        if x_b_train.shape[0] != 0:
            Ec.apply_bc(network, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)
        if x_u_train.shape[0] != 0:
            Ec.apply_ic(network, x_u_train, u_train, u_pred_var_list, u_train_var_list)

        u_pred_tot_vars = torch.cat(u_pred_var_list, 0).to(Ec.device)
        u_train_tot_vars = torch.cat(u_train_var_list, 0).to(Ec.device)

        assert not torch.isnan(u_pred_tot_vars).any()

        res = Ec.compute_res(network, x_f_train, None).to(Ec.device)

        loss_res = (torch.mean(abs(res) ** 2))
        loss_vars = (torch.mean(abs(u_pred_tot_vars - u_train_tot_vars) ** 2))
        loss_reg = regularization(network, order_regularizer)

        loss_v = torch.log10(lambda_residual * loss_vars + loss_res + lambda_reg * loss_reg)
        print("Total Loss:", loss_v.detach().cpu().numpy().round(4), "| Function Loss:", torch.log10(loss_vars).detach().cpu().numpy().round(4), "| PDE Loss:", torch.log10(loss_res).detach().cpu().numpy().round(4), "\n")

        return loss_v, loss_vars, loss_res


def fit(Ec, model, training_set_class, verbose=False):
    num_epochs = model.num_epochs
    optimizer = model.optimizer

    train_losses = list([np.NAN, np.NAN, np.NAN])
    freq = 50

    training_coll = training_set_class.data_coll
    training_boundary = training_set_class.data_boundary
    training_initial_internal = training_set_class.data_initial_internal

    model.train()

    def closure():
        optimizer.zero_grad()
        loss_f, loss_vars, loss_pde = CustomLoss().forward(Ec, model, x_u_train_, u_train_, x_b_train_, u_b_train_, x_coll_train_, training_set_class)
        loss_f.backward()
        train_losses[0] = loss_f
        train_losses[1] = loss_vars
        train_losses[2] = loss_pde
        return loss_f

    for epoch in range(num_epochs):
        if verbose and epoch % freq == 0:
            print("################################ ", epoch, " ################################")

        if len(training_boundary) != 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_), (x_u_train_, u_train_)) in enumerate(zip(training_coll, training_boundary, training_initial_internal)):
                if verbose and epoch % freq == 0:
                    print("Batch Number:", step)

                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                optimizer.step(closure=closure)

        elif len(training_boundary) == 0 and len(training_initial_internal) != 0:
            for step, ((x_coll_train_, u_coll_train_), (x_u_train_, u_train_)) in enumerate(zip(training_coll, training_initial_internal)):
                x_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)
                u_b_train_ = torch.full((0, x_u_train_.shape[1]), 0)

                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                optimizer.step(closure=closure)

        elif len(training_boundary) != 0 and len(training_initial_internal) == 0:
            for step, ((x_coll_train_, u_coll_train_), (x_b_train_, u_b_train_)) in enumerate(zip(training_coll, training_boundary)):
                x_u_train_ = torch.full((0, 1), 0)
                u_train_ = torch.full((0, 1), 0)

                x_coll_train_ = x_coll_train_.to(Ec.device)
                x_b_train_ = x_b_train_.to(Ec.device)
                u_b_train_ = u_b_train_.to(Ec.device)
                x_u_train_ = x_u_train_.to(Ec.device)
                u_train_ = u_train_.to(Ec.device)

                optimizer.step(closure=closure)

    return train_losses


def regularization(model, p):
    reg_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            reg_loss = reg_loss + torch.norm(param, p)
    return reg_loss



