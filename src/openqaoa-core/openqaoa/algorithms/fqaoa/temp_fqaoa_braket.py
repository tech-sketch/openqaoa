# https://openqaoa.entropicalabs.com/devices/amazon-braket/
from openqaoa import FQAOA
from openqaoa.backends import create_device
from temp_PO import portfolio

# device setting
device = create_device(location='aws', 
                            name='arn:aws:braket:::device/quantum-simulator/amazon/sv1', 
                            aws_region='us-west-1')

# parameters for fqaoa
num_assets = 10
Budget = 5
hopping = 1.0
PO = portfolio(num_assets, Budget)
print('device: ', device.device_location, device.device_name)
fqaoa = FQAOA(device)
fqaoa.compile(PO)
fqaoa.optimize()
opt_results = fqaoa.result
cost = opt_results.optimized['cost']
print('intermediate', opt_results.intermediate['cost'])
print('cost using FQAOA: ', cost)    
print()
    
