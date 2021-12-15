from ImportFile import *

sampling_seed, N_coll, N_u, N_int, folder_path, \
validation_size, network_properties, \
retrain, shuffle = initialize_inputs(len(sys.argv))

x = np.linspace(0,2.0*np.pi,1000)

multiPlot1D(x,1,1,network_properties)
