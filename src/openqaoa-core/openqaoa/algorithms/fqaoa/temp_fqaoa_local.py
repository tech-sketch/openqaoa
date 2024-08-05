# The simplest Open FQAOA workflow
from openqaoa import FQAOA
from openqaoa.backends import create_device
from temp_PO import portfolio

# parameters for fqaoa
num_assets = 10
Budget = 5
hopping = 1.0
po_problem = portfolio(num_assets, Budget)
# https://openqaoa.entropicalabs.com/devices/qiskit/
device_list = [create_device(location='local', name='qiskit.statevector_simulator'),
               create_device(location='local', name='qiskit.qasm_simulator'),
               create_device(location='local', name='qiskit.shot_simulator'),
               create_device(location='local', name='vectorized'),
               ]
for device in device_list:
    print('device: ', device.device_location, device.device_name)
    fqaoa = FQAOA(device)
    fqaoa.set_backend_properties(n_shots=10000)
    fqaoa.fermi_compile(po_problem)
    fqaoa.optimize()
    opt_results = fqaoa.result
    cost = opt_results.optimized['cost']
    print('cost using FQAOA: ', cost)    
    
