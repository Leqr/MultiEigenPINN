from ImportFile import *

n_coll_ = 2000
n_u_ = 2
n_int_ = 0
input_dimensions = 1
output_dimension = 1
#may not be the right parameters but are just here to initialize the pinn to evaluare it
network_properties_ = {
                "hidden_layers": 4,
                "neurons": 20,
                "residual_parameter": 100,
                "kernel_regularizer": 1.0,
                "normalization_parameter": 10000,
                "othogonality_parameter": 100,
                "regularization_parameter": 0.0,
                "batch_size": (n_coll_ + n_u_ + n_int_),
                "epochs": 1,
                "max_iter": 100000,
                "activation": "snake",
                "optimizer": "LBFGS"  # ADAM
}
x = np.linspace(0,2*np.pi, 1000)
multiPlot1D(x, input_dimensions, output_dimension, network_properties_)