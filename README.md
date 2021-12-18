# MultiEigenPINN
### Physics Informed Neural Network for Differential Eigenvalue Problems

## Setup the environment :
```bash
python3 -m venv vpinn
source vpinn/bin/activate
python3 -m pip install -r requirements.txt
```

## Running the multi-eigenpair solver
For finding mutliple eigenpairs, run [MultiSolve.py](src/MultoSolve.py)

#### A few settings for this solver :

**HYPER_SOLVE** in [MultiSolve.py](src/MultiSolve.py) defines if the program will run with RayTune hyperparameter optimization.

**TRANSFER_LEARNING** in [MultiSolve.py](src/MultiSolve.py) defines if the program will use transfer learning to find solutions.

**n_replicates** in [MultiSolve.py](src/MultiSolve.py) is the number of trial for finding eigenpairs.

The pytorch neural network solutions are stored in the **src/Solved** folder with the eigenvalue as file name.

## Running the single-eigenpair solver
For finding a single eigenpair, run [SingleSolve.py](src/SingleSolve.py)

Most of the settings mentionned above apply for this solver.


## Most of the simulation settings can be changed in the initialize_inputs function in [HelperSolve.py](src/HelperSolve.py)

**n_coll** defines the number of quasi-random collocation points.

**n_u_** defines the number of boundary points (for 1D problem use 2).

**network_properties_** are the various hyperparameters of the network.

**retrain_** defines the random seed for the solver. There is no way to set a random seed if **HYPER_SOLVE** is true (randomly chosen for different trials).

The equation to solve is defined by a class in **src/EquationModels/ForwardProblem**, in order to change the equation, the wanted equation class file need to be mentioned in [ImportFile.py](src/ImportFile.py) to define the wanted **EquationClass**.


## Running on an lsf scheduled cluster
There are in **src/utils** a few useful bash scripts to launch the solver with the lsf batch system.

