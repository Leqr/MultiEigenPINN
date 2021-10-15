from ImportFile import *
from EquationBaseClass import EquationBaseClass
from GeneratorPoints import generator_points
from SquareDomain import SquareDomain
from BoundaryConditions import PeriodicBC, DirichletBC, AbsorbingBC, NoneBC


class EquationClass(EquationBaseClass):

    def __init__(self, ):
        EquationBaseClass.__init__(self)

        self.type_of_points = "sobol"
        self.output_dimension = 1
        self.space_dimensions = 1
        self.time_dimensions = 1
        self.parameters_values = None
        self.parameter_dimensions = 0
        self.extrema_values = torch.tensor([[0, 1],
                                            [-1, 1]])
        self.list_of_BC = list([[self.ub0, self.ub1]])
        self.square_domain = SquareDomain(self.output_dimension,
                                          self.time_dimensions,
                                          self.space_dimensions,
                                          self.list_of_BC,
                                          self.extrema_values,
                                          self.type_of_points,
                                          vel_wave=self.a)

        ###########
        self.sigma = 1 / 8
        self.a_type = "constant"

    def add_collocation_points(self, n_coll, random_seed):
        return self.square_domain.add_collocation_points(n_coll, random_seed)

    def add_boundary_points(self, n_boundary, random_seed):
        return self.square_domain.add_boundary_points(n_boundary, random_seed)

    def add_initial_points(self, n_initial, random_seed):
        extrema_0 = self.extrema_values[:, 0]
        extrema_f = self.extrema_values[:, 1]
        x_time_0 = generator_points(n_initial, self.time_dimensions + self.space_dimensions, random_seed, self.type_of_points, True)
        x_time_0[:, 0] = torch.full(size=(n_initial,), fill_value=0.0)
        x_time_0 = x_time_0 * (extrema_f - extrema_0) + extrema_0

        y_time_0 = self.u0(x_time_0[:, 1:])

        return x_time_0, y_time_0

    def apply_bc(self, model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list):

        self.square_domain.apply_boundary_conditions(model, x_b_train, u_b_train, u_pred_var_list, u_train_var_list)

    def apply_ic(self, model, x_u_train, u_train, u_pred_var_list, u_train_var_list):
        for j in range(self.output_dimension):
            if x_u_train.shape[0] != 0:
                x_u_train.requires_grad = True
                u = model(x_u_train)[:, j]
                v = self.v0(x_u_train)[:, j]
                grad_u = torch.autograd.grad(u, x_u_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
                grad_u_t = grad_u[:, 0]
                u_pred_var_list.append(u)
                u_train_var_list.append(u_train[:, j])
                u_pred_var_list.append(grad_u_t)
                u_train_var_list.append(v)

    def a(self, inputs):
        x = inputs[:, 1]
        if self.a_type == "constant":
            a = 1 * torch.ones_like(x)
            return a
        if self.a_type == "pwc":
            n = 4
            a = 0.5 * torch.ones_like(x)
            a_mod = a
            for i in range(n):
                x_i = -1 + 2 / n * i
                x_i1 = -1 + 2 / n * (i + 1)
                a_i = (0.5 - 0.25) / (1 + 1) * (x_i + 1) + 0.25
                a_mod_new = torch.where((x < x_i1) & (x > x_i), torch.tensor(a_i), a_mod)
                a_mod = a_mod_new
            return a_mod
        if self.a_type == "swc":
            a = 0.25 / (1 + torch.exp(-150 * x)) + 0.25
            return a
        if self.a_type == "linear":
            a = x
            return a

    def compute_res(self, network, x_f_train, solid_object):
        x_f_train.requires_grad = True
        u = network(x_f_train)[:, 0].reshape(-1, )
        grad_u = torch.autograd.grad(u, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        grad_u_t = grad_u[:, 0]
        grad_u_x = grad_u[:, 1]
        grad_u_tt = torch.autograd.grad(grad_u_t, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0]
        grad_u_xx = torch.autograd.grad(grad_u_x, x_f_train, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1]

        residual = grad_u_tt.reshape(-1, ) - self.a(x_f_train) ** 2 * grad_u_xx.reshape(-1, )

        return residual

    def exact(self, x):
        x_sh_1 = x[:, 1] - self.a(x) * x[:, 0]
        x_sh_2 = x[:, 1] + self.a(x) * x[:, 0]

        u = 0.5 * (self.u0(x_sh_1)[:, 0] + self.u0(x_sh_2)[:, 0])
        v = -2 * pi * torch.sin(2 * pi * x[:, 1]) * torch.sin(2 * pi * x[:, 0])
        return torch.cat([u.reshape(-1, 1), v.reshape(-1, 1)], 1)

    def v0(self, x):

        return torch.zeros((x.shape[0], 1))

    def ub0(self, t):
        type_BC = [NoneBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def ub1(self, t):
        type_BC = [NoneBC()]
        u = torch.full(size=(t.shape[0], 1), fill_value=0.0)
        return u, type_BC

    def u0(self, x):

        u0 = torch.exp(-x ** 2 / (2 * self.sigma ** 2)).reshape(-1, 1)
        return u0

    def compute_generalization_error(self, model, extrema, images_path=None):
        model.eval()
        test_inp = self.convert(torch.rand([100000, extrema.shape[0]]), extrema)
        Exact = (self.exact(test_inp)[:, 0]).numpy().reshape(-1, 1)
        test_out = model(test_inp)[:, 0].detach().numpy().reshape(-1, 1)
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
        x = torch.reshape(torch.linspace(extrema[1, 0], extrema[1, 1], 100), [100, 1])
        time_steps = [0.0, 0.25, 0.5, 0.75, 1]
        scale_vec = np.linspace(0.65, 1.55, len(time_steps))

        plt.figure()
        plt.grid(True, which="both", ls=":")
        for val, scale in zip(time_steps, scale_vec):
            plot_var = torch.cat([torch.tensor(()).new_full(size=(100, 1), fill_value=val), x], 1)
            plt.plot(x, self.exact(plot_var)[:, 0], 'b-', linewidth=2, label=r'Exact, $t=$' + str(val) + r'$s$', color=self.lighten_color('grey', scale), zorder=0)
            plt.scatter(plot_var[:, 1].detach().numpy(), model(plot_var)[:, 0].detach().numpy(), label=r'Predicted, $t=$' + str(val) + r'$s$', marker="o", s=14,
                        color=self.lighten_color('C0', scale), zorder=10)

        plt.xlabel(r'$x$')
        plt.ylabel(r'u')
        plt.legend()
        plt.savefig(images_path + "/Samples.png", dpi=500)
