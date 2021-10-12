#!/bin/env python
# %%
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import DataLoader
import torch.linalg as tl
import numpy.linalg as nl
import math

class Sin(nn.Module):
    #sin activation function
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)

class Encoder(nn.Module):
    def __init__(self,d = 4) -> None:
        super().__init__()
        self.d = d

    def forward(self,x):
        #encodes an x tensor of dimension [N,1] into a tensor of dimension [N,2*d] and therefore contains d sin and d cos   
        
        n = x.shape[0]
        encoded = torch.zeros(n,2*self.d)
        pi = torch.tensor(math.pi)

        for i in range(n):
            for j in range(self.d):
                encoded[i,2*j] = torch.cos(pi*x[i]*2**j)
                encoded[i,2*j+1] = torch.sin(pi*x[i]*2**j)
        
        #encoded = x.apply_(self.fourier)
        return encoded


class NeuralNet(nn.Module):
    def __init__(self, input_dimension, output_dimension, n_hidden_layers, neurons, encode = True):
        super(NeuralNet, self).__init__()

        #fourier encoder 
        self.encode = encode
        self.encoder = Encoder(d = 6)

        # Number of input dimensions n
        if self.encode : 
            self.input_dimension = 2*self.encoder.d
        else:
            self.input_dimension = 1
        # Number of output dimensions m
        self.output_dimension = output_dimension
        # Number of neurons per layer
        self.neurons = neurons
        # Number of hidden layers
        self.n_hidden_layers = n_hidden_layers
        # Activation function
        self.activation = Sin()

        self.input_layer = nn.Linear(self.input_dimension, self.neurons)
        self.hidden_layers = nn.ModuleList([nn.Linear(self.neurons, self.neurons) for _ in range(n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)


    def forward(self, x):
        # The forward function performs the set of affine and non-linear transformations defining the network
        # (see equation above)

        if self.encode:
            x = self.encoder(x)

        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)


def init_xavier(model):
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            g = nn.init.calculate_gain('tanh')
            torch.nn.init.xavier_uniform_(m.weight, gain=g)
            # torch.nn.init.xavier_normal_(m.weight, gain=g)
            m.bias.data.fill_(0)

    model.apply(init_weights)

def fit(pinns_class, training_set_boundary, training_set_collocation, num_epochs, optimizer, verbose=True):
    history = list()

    # Loop over epochs
    for epoch in range(num_epochs):
        if verbose: print("################################ ", epoch, " ################################")

        running_loss = list([0])

        # Loop over batches
        for j, ((inp_train_b, u_train_b), (inp_train_c, u_train_c)) in enumerate(zip(training_set_boundary, training_set_collocation)):
            def closure():
                optimizer.zero_grad()
                loss = pinns_class.compute_loss(inp_train_b, u_train_b, inp_train_c, u_train_c,verbose)
                loss.backward()
                running_loss[0] = loss.item()
                return loss
            # Item 3. below
            optimizer.step(closure=closure)

        print('Loss: ', running_loss[0])
        history.append(running_loss[0])

    return history


class Pinn:
    def __init__(self, eigen = 1.0, hidden = 4, neurons = 20):
        self.domain_extrema = torch.tensor([0, 2*np.pi])

        self.approximate_solution = NeuralNet(input_dimension=1, output_dimension=1, n_hidden_layers=hidden, neurons=neurons)

        torch.manual_seed(12)
        init_xavier(self.approximate_solution)

        #eigenvalue
        self.lam = eigen


    # Function returning the training set S_sb corresponding to the spatial boundary
    def add_boundary_points(self):
        x0 = self.domain_extrema[0]
        xL = self.domain_extrema[1]
        bd_value = 0.0
        return torch.tensor([x0,xL]).reshape(-1,1),torch.tensor([bd_value,bd_value]).reshape(-1,1)

    # Function returning the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_collocation_points(self, n_collocation):
        self.soboleng = torch.quasirandom.SobolEngine(dimension=1)
        input_collocation = self.soboleng.draw(n_collocation)
        # torch.random.manual_seed(random_seed)
        # input_collocation = torch.rand([n_collocation, 2]).type(torch.FloatTensor)
        input_collocation = self.convert(input_collocation)

        output_collocation = torch.zeros((n_collocation, 1))
        return input_collocation.reshape(-1,1), output_collocation.reshape(-1,1)

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    def apply_boundary_conditions(self, input_boundary, output_boundary):
        pred_output_boundary = self.approximate_solution(input_boundary)
        return output_boundary, pred_output_boundary

    # Function to compute the PDE residuals
    def compute_pde_residual(self, input_collocation):
        input_collocation.requires_grad = True
        u = self.approximate_solution(input_collocation).reshape(-1, )

        grad_u_x = torch.autograd.grad(u.sum(), input_collocation, create_graph=True)[0][:,0]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), input_collocation, create_graph=True)[0][:,0]

        residual = grad_u_xx + u*self.lam**2
        #enforce function normalisation
        loss_L2 = 10000*torch.abs(torch.mean(u**2)-0.5).reshape(1,)
        residual = torch.cat([residual,loss_L2]) 

        return residual.reshape(-1, )

    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        return tens * (self.domain_extrema[1] - self.domain_extrema[0]) + self.domain_extrema[0]

    def compute_loss(self, inp_train_b, u_train_b, inp_train_c, u_train_c, verbose = True):
        u_train_b, u_pred_b = self.apply_boundary_conditions(inp_train_b, u_train_b)

        r_int = self.compute_pde_residual(inp_train_c)  
        r_sb = u_train_b - u_pred_b

        loss_sb = torch.mean(r_sb**2)
        loss_int = torch.mean(r_int**2)

        lambda_b = 1
        lambda_int = 10

        loss = torch.log10(lambda_b*loss_sb + lambda_int*loss_int) 
        if verbose :
            print("Total loss: ", round(loss.item(), 4), "| PDE Log Residual: ", round(torch.log10(loss_sb).item(), 4), "| Boundary Log Error: ", round(torch.log10(loss_int).item(), 4))

        return loss

def fit_with_lam(pinn,training_set_b, training_set_c,eigen = 1.0):
    pinn.lam = eigen
    n_epochs = 1
    optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(), lr=float(1.5), max_iter=1000, max_eval=50000, history_size=150,
                                line_search_fn="strong_wolfe",
                                tolerance_change=1.0 * np.finfo(float).eps)
    hist = fit(pinn, training_set_b, training_set_c, num_epochs=n_epochs, optimizer=optimizer_LBFGS, verbose=True)

def eigenTest(pinn,training_set_b, training_set_c, input_c_ ,eigenmax = 20.0):
    hists = []
    true_sol_errs = [] 
    for i in range(eigenmax):
        print("Lambda = ",i+1)
        n_epochs = 1
        optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(), lr=float(1.5), max_iter=1000, max_eval=50000, history_size=150,
                                    line_search_fn="strong_wolfe",
                                    tolerance_change=1.0 * np.finfo(float).eps)
        hist = fit(pinn, training_set_b, training_set_c, num_epochs=n_epochs, optimizer=optimizer_LBFGS, verbose=False)
        hists.append(hist)

        #compare with true solution
        true_sol = torch.sin((i+1)*input_c_)
        pred = pinn.approximate_solution(input_c_)
        norm_err = float(tl.norm(true_sol - pred)**2)
        if norm_err > 10.0 :
            norm_err = float(tl.norm(true_sol + pred)**2)
            pred = -1.0*pred
        true_sol_errs.append(norm_err)
        print("Error to real solution : ", norm_err)

        if True:
            #show numerical solution
            pred = pred.detach().numpy()
            plt.scatter(input_c_,pred,marker = ".",label = "prediction")
            plt.scatter(input_c_,true_sol,marker = ".",label = "true")
            plt.ylim(min(true_sol),max(true_sol))
            plt.legend()
            plt.savefig("figures/eigenPINN-{}.png".format(i+1))
            plt.clf()
        
        #reset the PINN in order to get new initial weight values
        pinn = Pinn()
        pinn.lam = i+2

    return true_sol_errs,hists


#Initialize PINN
pinn = Pinn()

# Generate S_sb, S_tb, S_int
input_b_, output_b_ = pinn.add_boundary_points()  # S_sb

n_coll = 300
input_c_, output_c_ = pinn.add_collocation_points(n_coll)  # S_int

#create dataset for pytorch model
training_set_b = DataLoader(torch.utils.data.TensorDataset(input_b_, output_b_), batch_size=2, shuffle=False)
training_set_c = DataLoader(torch.utils.data.TensorDataset(input_c_, output_c_), batch_size=n_coll, shuffle=False)


fit_with_lam(pinn,training_set_b, training_set_c, eigen = 1.0)
#show numerical solution
pred = pinn.approximate_solution(input_c_)
pred = pred.detach().numpy()
plt.scatter(input_c_,pred,marker = ".")
plt.ylim(min(pred),max(pred))
plt.show()


#true_sol_errs, history = eigenTest(pinn,training_set_b, training_set_c, input_c_, eigenmax=20)

# %%
