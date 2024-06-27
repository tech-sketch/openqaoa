# The simplest Open FQAOA workflow
import networkx
import numpy as np
from openqaoa.problems import MinimumVertexCover

# The portfolio optimization
from docplex.mp.model import Model
np.random.seed(1)
num_assets = 10
num_days = 15
q = 0.01
Budget = 5
# Creating a random history of the forcasting for the expected return 
hist_exp = (1 - 2 * np.random.rand(num_assets)).reshape(-1,1) * (np.array([np.arange(num_days) for i in range(num_assets)]) + np.random.randint(10)) + (1 - 2 * np.random.rand(num_assets,  num_days))
mu = hist_exp.mean(axis=1)
sigma = np.cov(hist_exp)

#Start the docplex model with Model("name of the model")
from openqaoa.problems.converters import FromDocplex2IsingModel
mdl = Model('Portfolio Optimization')
x = np.array(mdl.binary_var_list(num_assets, name='asset'))
objective_function = mu @ x - x.T @ sigma @ x
mdl.maximize(objective_function)
mdl.add_constraint(x.sum() == Budget, ctname='budget')
qubo_po = FromDocplex2IsingModel(mdl)
ising_encoding_po = qubo_po.ising_model 


g = networkx.circulant_graph(6, [1])
vc = MinimumVertexCover(g, field=1.0, penalty=10)
qubo_problem = vc.qubo

# create a FQAOA workflow
from openqaoa import QAOA, FQAOA
from openqaoa.backends import create_device # for qiskit
device = create_device('local', 'qiskit.statevector_simulator') # for qiskit

qaoa_po = QAOA(device)
qaoa_po.set_circuit_properties(p=2, init_type='ramp')
qaoa_po.compile(ising_encoding_po)
qaoa_po.optimize()
opt_results = qaoa_po.result
cost = opt_results.optimized['cost']
print(cost)

