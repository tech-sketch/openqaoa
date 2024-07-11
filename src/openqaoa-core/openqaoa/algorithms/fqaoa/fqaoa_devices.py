from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from braket.circuits import Circuit
from openqaoa_qiskit.backends.gates_qiskit import QiskitGateApplicator
from openqaoa_braket.backends.gates_braket import BraketGateApplicator


class QuantumDevice(ABC):
    """
    Abstract base class representing a quantum computing device.

    Attributes:
        n_qubits (int): Number of qubits targeted by the device.

    Methods:
        __init__(n_qubits):
            Initializes the QuantumDevice with a given number of qubits.

        initial_circuit():
            Abstract method to be implemented in subclasses for initializing the quantum circuit.

        gate_applicator():
            Abstract method to be implemented in subclasses for setting the gate applicator.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        
    @abstractmethod
    def initial_circuit(self):
        pass
    
    @abstractmethod
    def gate_applicator(self):
        pass


class QiskitDevice(QuantumDevice):
    def __init__(self, n_qubits): 
        super().__init__(n_qubits)
        
    def initial_circuit(self):
        return QuantumCircuit(self.n_qubits)
    
    def gate_applicator(self):
        return QiskitGateApplicator()

class BraketDevice(QuantumDevice):
    def __init__(self, n_qubits):
        super().__init__(n_qubits)

    def initial_circuit(self):
        return Circuit()
    
    def gate_applicator(self):
        return BraketGateApplicator()
        
def set_device(device_location, n_qubits):
    """
    Factory function to create a quantum device based on the specified location.

    Args:
        device_location (str): Location where the quantum device is located ('ibmq', 'azure', 'braket').
        n_qubits (int): Number of qubits targeted by the device.

    Returns:
        tuple: Tuple containing the initialized quantum circuit object and gate applicator object.
    Raises:
        ValueError: If an unsupported device location is provided.
    """
    if device_location in ["local", "ibmq", "azure"]:
        device = QiskitDevice(n_qubits)
    elif device_location == "braket":
        device = BraketDevice()
    else:
        raise ValueError("Unsupported device")
    return device.initial_circuit(), device.gate_applicator()

