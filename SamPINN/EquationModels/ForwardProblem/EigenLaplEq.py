import torch
from ImportFile import *
from EquationBaseClass import EquationBaseClass
from GeneratorPoints import generator_points
from SquareDomain import SquareDomain
from BoundaryConditions import PeriodicBC, DirichletBC, AbsorbingBC, NoneBC


class EquationClass(EquationBaseClass):

    def __init__(self, eigenvalue = 1.0):
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

    def compute_res(self, network, x_f_train, solid_object, lambda_norm = 10):
        x_f_train.requires_grad = True
        u = network(x_f_train)[:, 0].reshape(-1, )

        grad_u_x = torch.autograd.grad(u.sum(), x_f_train, create_graph=True)[0][:,0]
        grad_u_xx = torch.autograd.grad(grad_u_x.sum(), x_f_train, create_graph=True)[0][:,0]

        residual = grad_u_xx + (network.lam**2)*u

        #enforce function normalisation
        norm_loss = lambda_norm*torch.abs(torch.mean(u**2)-0.5).reshape(1,)
        residual = torch.cat([residual,norm_loss]).reshape(-1,)
        assert not torch.isnan(residual).any()

        print("Eigenvalue = {}".format(network.lam.detach().numpy()[0]))

        return residual

    def exact(self, x):
        return torch.sin(self.lam*x)

    def ub0(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub1(self, t):
        type_BC = [DirichletBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def compute_generalization_error(self, model, extrema, images_path=None):
        model.eval()
        test_inp = self.convert(torch.rand([100000, extrema.shape[0]]), extrema)
        Exact = (self.exact(test_inp)).numpy().reshape(-1, 1)
        test_out = model(test_inp).detach().numpy().reshape(-1, 1)
        assert (Exact.shape[1] == test_out.shape[1])
        L2_test = np.sqrt(np.mean((Exact - test_out) ** 2))
        print("Error Test:", L2_test)

        rel_L2_test = L2_test / np.sqrt(np.mean(Exact ** 2))
        print("Relative Error Test:", rel_L2_test)

        if images_path is not None:
            plt.figure()
            plt.grid(True, which="both", ls=":")
            plt.scatter(Exact, test_out)
            plt.xlabel(r'Exact Values')
            plt.ylabel(r'Predicted Values')
            plt.savefig(images_path + "/Score.png", dpi=400)
        return L2_test, rel_L2_test

    def plotting(self, model, images_path, extrema, solid):
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

