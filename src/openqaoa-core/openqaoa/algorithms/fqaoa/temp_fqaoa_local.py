# The simplest Open FQAOA workflow
from openqaoa import FQAOA
from openqaoa.backends import create_device
from temp_PO import portfolio

# parameters for fqaoa
num_assets = 10
Budget = 5
hopping = 1.0
PO = portfolio(num_assets, Budget)
# https://openqaoa.entropicalabs.com/devices/qiskit/
device_list = [create_device(location='local', name='qiskit.statevector_simulator'),
               create_device(location='local', name='qiskit.qasm_simulator'),
               create_device(location='local', name='qiskit.shot_simulator'),
               create_device(location='local', name='vectorized'),
               ]
for device in device_list:
    print('device: ', device.device_location, device.device_name)
    fqaoa = FQAOA(device)
    fqaoa.fermi_compile(PO)
    fqaoa.optimize()
    opt_results = fqaoa.result
    cost = opt_results.optimized['cost']
    print('cost using FQAOA: ', cost)    
    
