import matplotlib.pyplot as plt
import torch
from ImportFile import *
from EquationBaseClass import EquationBaseClass
from GeneratorPoints import generator_points
from SquareDomain import SquareDomain
from BoundaryConditions import PeriodicBC, DirichletBC, AbsorbingBC, NoneBC


class EquationClass(EquationBaseClass):
    """
    ∆u + E*u = 0 on [0,2*pi] x ... x [0,2*pi]
    u(x) = 0.0 on the boundary (free particle in a box with infinite potential outside)
    """

    def __init__(self, dimension = 2):
        EquationBaseClass.__init__(self)

        self.type_of_points = "sobol"
        self.output_dimension = 1
        self.space_dimensions = dimension
        self.time_dimensions = 0
        self.parameters_values = None
        self.parameter_dimensions = 0
        self.extrema_values = torch.tensor([[0, 2*np.pi] for i in range(self.space_dimensions)])
        self.list_of_BC = list([[self.ub0, self.ub1] for i in range(self.space_dimensions)])
        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          self.list_of_BC,
                                          self.extrema_values,
                                          self.type_of_points)

        #use a tensor so that it can be made into a trainable parameter
        self.lam = torch.tensor(1.0)


    def add_collocation_points(self, n_coll, random_seed):
        return self.square_domain.add_collocation_points(n_coll, random_seed)

    def add_boundary_points(self, n_boundary, random_seed):
        return self.square_domain.add_boundary_points(n_boundary, random_seed)

    def add_initial_points(self, n_initial, random_seed):
        #time indepedent pde --> no initial points
        return torch.tensor([]), torch.tensor([])

    def apply_bc(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):

        self.square_domain.apply_boundary_conditions(model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)

    def compute_res(self, network, x_f_train, solid_object, lambda_norm = 10, lambda_orth = 100, verbose = False):
        """
        Computes the PDE residual. It is constituted of the pde loss ∆u = u*lambda,
        the normalization loss |||u||^2 - c| and the orthogonal condition when some
        eigensolutions were already found sum over the previous solutions of <u,u_prev>.
        :param network:
        :param x_f_train:
        :param solid_object:
        :param lambda_norm: hyperparameter for the normalization loss value
        :param lambda_orth: hyperparameter for the orthogonal loss value
        :param verbose:
        :return pde residual:
        """
        x_f_train.requires_grad = True
        u = network(x_f_train)[:, 0].reshape(-1, )

        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]

        grads = []
        #iterate over the space dimensions to get the gardients
        for i in range(self.space_dimensions):
            grads.append(torch.autograd.grad(grad_u[:,i], x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,i])

        hbar = 1.054571e-34
        melectron = 9.109383e-31
        residual = sum(grads) + network.lam**u

        #enforce probability density normalisation a la QM
        #compute the volume of Omega for monte carlo integration
        volume = 1
        for val in self.extrema_values:
            length = val[1]-val[0]
            volume *= length

        norm_loss = lambda_norm*torch.abs(torch.mean(u**2)-1/volume).reshape(1,)

        #othogonal condition when trying to solve for multiple eigenvalues
        loss_orth = torch.tensor([0.0])
        if network.other_networks is not None:
            for eig, func in network.other_solutions.items():
                assert func.shape == u.shape
                loss_orth += torch.mean(u*func)**2
            loss_orth = lambda_orth*loss_orth

        residual = torch.cat([residual, norm_loss, loss_orth]).reshape(-1, )
        assert not torch.isnan(residual).any()

        #show eigenvalue
        trained_lam = network.lam.detach().numpy()[0]
        if verbose and (network.iter % 10 == 0): print("Eigenvalue = {}".format(trained_lam))
        self.lam = trained_lam

        return residual

    def exact(self, x, lam = None):
        return None

    def ub0(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub1(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def compute_generalization_error(self, model, extrema, images_path=None):
        return None,None

    def plotting(self, model, images_path, extrema, solid):
        if self.space_dimensions == 2:
            model.cpu()
            model = model.eval()
            x = extrema[0,1]*torch.rand((1000,2))
            pred = model(x)
            pred = pred.detach().numpy()
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.scatter(x[:,0], x[:,1], pred)
            plt.savefig(images_path + "/Samples.png", dpi=500)
        else :
            return None
