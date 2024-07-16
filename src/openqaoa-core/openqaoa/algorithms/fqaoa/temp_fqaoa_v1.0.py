# The simplest Open FQAOA workflow
import networkx
import numpy as np

# The portfolio optimization
def portfolio(num_assets, Budget):
    from docplex.mp.model import Model
    np.random.seed(1)
    num_days = 15
    q = 0.01
    # Creating a random history of the forcasting for the expected return 
    hist_exp = (1 - 2 * np.random.rand(num_assets)).reshape(-1,1) * (np.array([np.arange(num_days) for i in range(num_assets)]) + np.random.randint(10)) + (1 - 2 * np.random.rand(num_assets,  num_days))
    mu = hist_exp.mean(axis=1)
    sigma = np.cov(hist_exp)
    #Start the docplex model with Model("name of the model")
    from openqaoa.problems.converters import FromDocplex2IsingModel
    mdl = Model('Portfolio-Optimization')
    x = np.array(mdl.binary_var_list(num_assets, name='asset'))
    objective_function = - mu @ x + x.T @ sigma @ x
    mdl.minimize(objective_function)
    mdl.add_constraint(x.sum() == Budget, ctname='budget')
    qubo_po = FromDocplex2IsingModel(mdl)
    ising_encoding_po = qubo_po.ising_model
    return ising_encoding_po, Budget

# create a conventional FQAOA workflow
from openqaoa import FQAOA
from openqaoa.backends import create_device # for qiskit

# parameters for fqaoa
num_assets = 10
Budget = 5
hopping = 1.0
device_list = [create_device(location='local', name='qiskit.statevector_simulator'),
               create_device(location='local', name='qiskit.qasm_simulator'),
               create_device(location='local', name='qiskit.shot_simulator'),
               create_device(location='local', name='vectorized'),
               create_device(location='local', name='analytical_simulator')]
for device in device_list:
    print('device: ', device.device_name)    
    fqaoa = FQAOA(device)
    fqaoa.set_circuit_properties(p=2, init_type='ramp')
# optional: fqaoa.set_circuit_properties(p=2, init_type='ramp', mixer_qubit_connectivity='chain')
# optional: fqaoa.set_classical_optimizer(method='bfgs', jac="finite_difference")
# optional: fqaoa.set_backend_properties(n_shots = 100, seed_simulator = 1)
    fqaoa.compile(portfolio(num_assets, Budget))
    fqaoa.optimize()
    opt_results = fqaoa.result
    cost = opt_results.optimized['cost']
    print('intermediate', opt_results.intermediate['cost'])
    print('cost using FQAOA: ', cost, 'compared to 27.028662')
    print('angles: ', opt_results.optimized['angles'])
    print()
    
