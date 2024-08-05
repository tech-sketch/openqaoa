# https://openqaoa.entropicalabs.com/devices/amazon-braket/
from openqaoa import FQAOA
from openqaoa.backends import create_device
from temp_PO import portfolio

# device setting
device1 = create_device(location='aws', name='arn:aws:braket:::device/quantum-simulator/amazon/sv1', aws_region='us-east-1')
device2 = create_device(location='aws', name='arn:aws:braket:eu-north-1::device/qpu/iqm/Garnet',     aws_region='eu-north-1')
device = device1

# parameters for fqaoa
num_assets = 10
Budget = 5
hopping = 1.0
po_problem = portfolio(num_assets, Budget)
print('device: ', device.device_location, device.device_name)
fqaoa = FQAOA(device)
fqaoa.set_backend_properties(n_shots=10000)
fqaoa.fermi_compile(po_problem)
fqaoa.optimize()
opt_results = fqaoa.result
print('lowest_cost', opt_results.lowest_cost_bitstrings(5))
cost = opt_results.optimized['cost']
print('intermediate', opt_results.intermediate['cost'])
print('cost using FQAOA: ', cost)
print()
