# https://openqaoa.entropicalabs.com/workflows/run-workflows-on-qpus-qpu/#a-simple-example-connecting-to-the-ibmq-cloud-qasm-simulator
from qiskit_ibm_provider import IBMProvider
import os
IBMProvider.save_account(os.getenv('IBM_TOKEN'), overwrite=True)
provider = IBMProvider()
provider.backends()

from openqaoa import FQAOA
from openqaoa.backends import create_device
from temp_PO import portfolio

# device setting
device = create_device(location='ibmq', name='ibm_osaka')

# parameters for fqaoa
num_assets = 10
Budget = 5
hopping = 1.0
po_problem = portfolio(num_assets, Budget)
print('device: ', device.device_location, device.device_name)
fqaoa = FQAOA(device)
fqaoa.set_circuit_properties(p=1, param_type = 'standard', init_type='custom',
                             variational_params_dict={'betas': [0.0], 'gammas': [0.0]})
fqaoa.set_backend_properties(n_shots = 10000)
fqaoa.set_classical_optimizer(maxiter=0)
fqaoa.fermi_compile(po_problem)
fqaoa.optimize()
opt_results = fqaoa.result
cost = opt_results.optimized['cost']
print('cost using FQAOA: ', cost)    
print()
    
