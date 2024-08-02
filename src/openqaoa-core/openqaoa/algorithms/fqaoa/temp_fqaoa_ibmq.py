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
PO = portfolio(num_assets, Budget)
print('device: ', device.device_location, device.device_name)
fqaoa = FQAOA(device)
fqaoa.set_backend_properties(n_shots = 1)
fqaoa.set_classical_optimizer(method='cobyla', maxiter=1, tol=0.05)
fqaoa.fermi_compile(PO)
fqaoa.optimize()
opt_results = fqaoa.result
cost = opt_results.optimized['cost']
print('cost using FQAOA: ', cost)    
print()
    
