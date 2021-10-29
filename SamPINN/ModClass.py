from ImportFile import *


class Swish(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Sin(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class Snake(nn.Module):
    def __init__(self):
        super().__init__()
        # self.alpha = torch.nn.Parameter(torch.tensor(0.5), requires_grad=True)
        self.alpha = 0.5

    def forward(self, x):
        return x + torch.sin(self.alpha * x) ** 2 / self.alpha

class Encoder(nn.Module):

    def __init__(self, d_max):
        super().__init__()

        self.d = d_max
        self.omega = 2 ** np.arange(0, self.d)


    def forward(self, x):

        pe = torch.zeros(x.shape[0], x.shape[1], 2, self.d)
        for n_col in range(x.shape[1]):

            for i in range(self.omega.shape[0]):
                pe[:, n_col, 0, i] = torch.sin(self.omega[i] * x[:, n_col])
                pe[:, n_col, 1, i] = torch.cos(self.omega[i] * x[:, n_col])

        pe = pe.view(x.shape[0], -1)
        return pe


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['swish']:
        return Swish()
    elif name in ['sin']:
        return Sin()
    elif name in ['snake']:
        return Snake()
    else:
        raise ValueError('Unknown activation function')


class Pinns(nn.Module):

    def __init__(self, input_dimension, output_dimension, network_properties, other_networks = None):
        super(Pinns, self).__init__()

        #eigenvalue problems
        max_eigenvalue = 10.0
        self.lam0 = max_eigenvalue*torch.rand(1)
        self.lam = nn.Parameter(self.lam0,requires_grad=True)

        # positional encoding
        self.encode = False
        if self.encode:
            self.encoder = Encoder(d_max=6)
            self.input_dimension = int(2*self.encoder.d)
        else:
            self.input_dimension = input_dimension

        #neural net architecture
        self.output_dimension = output_dimension
        self.n_hidden_layers = int(network_properties["hidden_layers"])
        self.neurons = int(network_properties["neurons"])

        #weight losses
        self.lambda_residual = float(network_properties["residual_parameter"])
        self.kernel_regularizer = int(network_properties["kernel_regularizer"])
        self.lambda_norm = network_properties["normalization_parameter"]
        self.regularization_param = float(network_properties["regularization_parameter"])

        #param
        self.num_epochs = int(network_properties["epochs"])
        self.act_string = str(network_properties["activation"])
        self.optimizer = network_properties["optimizer"]

        #layers
        self.input_layer = nn.Linear(self.input_dimension, self.neurons)

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.neurons, self.neurons) for _ in range(self.n_hidden_layers - 1)])
        self.output_layer = nn.Linear(self.neurons, self.output_dimension)

        #activation
        self.activation = activation(self.act_string)

        #full_solve
        self.other_networks = other_networks


    def forward(self, x):
        if self.encode:
            x = self.encoder(x)
        x = self.activation(self.input_layer(x))
        for k, l in enumerate(self.hidden_layers):
            x = self.activation(l(x))
        return self.output_layer(x)

    def evaluate_other_functions(self,x_coll):
        self.other_solutions = dict()
        for eigen,sol in self.other_networks.items():
            self.other_solutions[eigen] = sol(x_coll)


def init_xavier(model):
    def init_weights(m):
        if type(m) == nn.Linear and m.weight.requires_grad and m.bias.requires_grad:
            # gain = nn.init.calculate_gain('tanh')
            gain = 1
            torch.nn.init.xavier_uniform_(m.weight, gain=gain)
            torch.nn.init.uniform_(m.bias, 0, 1)

            # torch.nn.init.xavier_uniform_(m.bias)
            # m.bias.data.fill_(0)

    model.apply(init_weights)

