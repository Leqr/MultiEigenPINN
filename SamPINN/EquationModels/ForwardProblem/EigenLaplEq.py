import torch
from ImportFile import *
from EquationBaseClass import EquationBaseClass
from GeneratorPoints import generator_points
from SquareDomain import SquareDomain
from BoundaryConditions import PeriodicBC, DirichletBC, AbsorbingBC, NoneBC


class EquationClass(EquationBaseClass):
    """
    ∆u + lambda^2*u = 0
    u(0) = u(2*pi) = 0.0
    """

    def __init__(self):
        EquationBaseClass.__init__(self)

        self.type_of_points = "sobol"
        self.output_dimension = 1
        self.space_dimensions = 1
        self.time_dimensions = 0
        self.parameters_values = None
        self.parameter_dimensions = 0
        self.extrema_values = torch.tensor([[0, 2*np.pi]])
        self.list_of_BC = list([[self.ub0, self.ub1]])
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
        Computes the PDE residual. It is constituted of the pde loss ∆u + u*lambda^2,
        the normalization loss |||u||^2 - 0.5| and the orthogonal condition when some
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

        grad_u_x = torch.autograd.grad(u.sum(), x_f_train, create_graph=True)[0][:,0]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), x_f_train, create_graph=True)[0][:,0]

        residual = grad_u_xx + (network.lam**2)*u

        #enforce function normalisation
        norm_loss = lambda_norm*torch.abs(torch.mean(u**2)-0.5).reshape(1,)

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
        """
        Exact solution of the pde. Takes care of the case where the eigenvalue is given
        and where the eigenvalue is learned.
        :param x: domain value to evaluate the function
        :param lam: eigenvalue
        :return exact solution:
        """
        if lam is not None :
            return torch.sin(lam*x)
        else:
            eigen = round(float(self.lam) * 2) / 2
            return torch.sin(eigen*x)

    def ub0(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub1(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def compute_generalization_error(self, model, extrema, images_path=None):
        """
        Compute the generalization error given the exact solution of the problem.
        :param model:
        :param extrema:
        :param images_path:
        :return L2_test, rel_L2_test: L2 and relative L2 test errors.
        """
        model.eval()
        test_inp = self.convert(torch.rand([100000, extrema.shape[0]]), extrema)
        Exact = (self.exact(test_inp)).numpy().reshape(-1, 1)
        test_out = model(test_inp).detach().numpy().reshape(-1, 1)
        assert (Exact.shape[1] == test_out.shape[1])
        L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
        print("Error :", L2_test)

        rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
        print("Relative Error :", rel_L2_test)

        if images_path is not None:
            plt.figure()
            plt.grid(True, which="both", ls=":")
            plt.scatter(Exact, test_out)
            plt.xlabel(r'Exact Values')
            plt.ylabel(r'Predicted Values')
            plt.savefig(images_path + "/Score.png", dpi=400)
        return L2_test, rel_L2_test

    def plotting(self, model, images_path, extrema, solid):
        """
        Plots the numerical and exact solutions of the pde.
        :param model:
        :param images_path:
        :param extrema:
        :param solid:
        """
        model.cpu()
        model = model.eval()
        x = torch.linspace(extrema[0, 0], extrema[0, 1], 500).reshape(-1,1)
        pred = model(x)
        pred = pred.detach().numpy()
        exact = self.exact(x)
        plt.figure()
        plt.scatter(x,pred,marker = ".", label = "prediction")
        plt.scatter(x,exact,marker = ".", label = "exact solution")
        plt.ylim(min(pred),max(pred))
        plt.legend()
        plt.savefig(images_path + "/Samples.png", dpi=500)


