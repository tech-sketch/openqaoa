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
    objective_function = mu @ x - x.T @ sigma @ x
    mdl.maximize(objective_function)
    mdl.add_constraint(x.sum() == Budget, ctname='budget')
    qubo_po = FromDocplex2IsingModel(mdl)
    ising_encoding_po = qubo_po.ising_model
    return ising_encoding_po

# create a device

# create a conventional FQAOA workflow
from openqaoa import FQAOA
from openqaoa.backends import create_device # for qiskit
# parameters for fqaoa
num_assets = 10
Budget = 5
hopping = 1.0
lattice = 'cyclic'
device_dict = {'qiskit': create_device(location='local', name='qiskit.statevector_simulator'),
               'local': create_device(location='local', name='vectorized')}
for backend, device in device_dict.items():
    print('device: ', device.device_name)    
    fqaoa = FQAOA(device)
    fqaoa.set_fqaoa_parameters(num_assets, Budget, hopping, lattice)
    fqaoa.set_circuit_properties(p=2, init_type='ramp')
#    fqaoa.set_classical_optimizer(maxiter=100, method='cobyla') #
#    fqaoa.set_circuit_properties(p=2, init_type='custom', variational_params_dict =
#                                 {'betas':[0.585006801179, 0.266641182597],
#                                  'gammas':[0.07350407864, 0.530566945246]})
#    fqaoa.set_classical_optimizer(maxiter=0, method='cobyla') #    
    fqaoa.compile(portfolio(num_assets, Budget))
    fqaoa.optimize()
    # get resutls
    opt_results = fqaoa.result
    cost = opt_results.optimized['cost']
    print('cost using FQAOA on cyclic lattice: ', cost, 'compared to 27.028662')
    print('angles: ', opt_results.optimized['angles'])
