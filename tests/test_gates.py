import unittest
import numpy as np

from qiskit import QuantumCircuit
from pyquil import Program, quilbase
from pyquil.gates import RX as p_RX
from pyquil.gates import RY as p_RY
from pyquil.gates import RZ as p_RZ
from pyquil.gates import CZ as p_CZ
from pyquil.gates import CNOT as p_CX
from pyquil.gates import XY as p_XY
from pyquil.gates import CPHASE as p_CPHASE
from braket.circuits import gates as braketgates
from braket.circuits import Circuit
from braket.circuits.free_parameter import FreeParameter

from openqaoa.qaoa_components.ansatz_constructor.gates import (
    RY,
    RX,
    RZ,
    CZ,
    CX,
    RXX,
    RYY,
    RZZ,
    RZX,
    CPHASE,
    RiSWAP,
)
from openqaoa.qaoa_components.ansatz_constructor.rotationangle import RotationAngle
from openqaoa_braket.backends.gates_braket import BraketGateApplicator
from openqaoa_qiskit.backends.gates_qiskit import QiskitGateApplicator
from openqaoa_pyquil.backends.gates_pyquil import PyquilGateApplicator


class TestingGate(unittest.TestCase):
    
    def setUp(self):
        
        self.braket_gate_applicator = BraketGateApplicator()
        self.qiskit_gate_applicator = QiskitGateApplicator()
        self.pyquil_gate_applicator = PyquilGateApplicator()
    
    def test_braket_gates_1q(self):
        
        # Braket Gate Applicator
        gate_applicator = self.braket_gate_applicator

        # One Qubit Gate Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], FreeParameter("test_angle"))
        
        empty_circuit = Circuit()
        llgate = RY(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.Ry.ry(0, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = RX(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.Rx.rx(0, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = RZ(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.Rz.rz(0, np.pi)

        self.assertEqual(test_circuit, output_circuit)

    def test_braket_gates_2q(self):
        
        # Braket Gate Applicator
        gate_applicator = self.braket_gate_applicator

        # Two Qubit Gate Tests
        empty_circuit = Circuit()
        llgate = CZ(gate_applicator, 0, 1)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = Circuit()
        test_circuit += braketgates.CZ.cz(0, 1)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = CX(gate_applicator, 0, 1)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = Circuit()
        test_circuit += braketgates.CNot.cnot(0, 1)

        self.assertEqual(test_circuit, output_circuit)

    def test_braket_gates_2q_w_gates(self):
        
        # Braket Gate Applicator
        gate_applicator = self.braket_gate_applicator

        # Two Qubit Gate with Angles Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], FreeParameter("test_angle"))

        empty_circuit = Circuit()
        llgate = RXX(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.XX.xx(0, 1, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = RYY(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.YY.yy(0, 1, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = RZZ(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.ZZ.zz(0, 1, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = CPHASE(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.CPhaseShift.cphaseshift(0, 1, np.pi)

        self.assertEqual(test_circuit, output_circuit)

        empty_circuit = Circuit()
        llgate = RiSWAP(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)
        output_circuit = output_circuit.make_bound_circuit({"test_angle": np.pi})

        test_circuit = Circuit()
        test_circuit += braketgates.XY.xy(0, 1, np.pi)

        self.assertEqual(test_circuit, output_circuit)

    def test_ibm_gates_1q(self):
        
        # Qiskit Gate Applicator
        gate_applicator = self.qiskit_gate_applicator

        # One Qubit Gate Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)

        empty_circuit = QuantumCircuit(1)
        llgate = RY(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(1)
        test_circuit.ry(np.pi, 0)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(1)
        llgate = RX(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(1)
        test_circuit.rx(np.pi, 0)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(1)
        llgate = RZ(gate_applicator, 0, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(1)
        test_circuit.rz(np.pi, 0)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

    def test_ibm_gates_2q(self):
        
        # Qiskit Gate Applicator
        gate_applicator = self.qiskit_gate_applicator

        # Two Qubit Gate Tests
        empty_circuit = QuantumCircuit(2)
        llgate = CZ(gate_applicator, 0, 1)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.cz(0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(2)
        llgate = CX(gate_applicator, 0, 1)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.cx(0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

#         empty_circuit = QuantumCircuit(2)
#         llgate = CX(gate_applicator)
#         output_circuit = llgate.apply_ibm_gate([0, 1], empty_circuit)

#         test_circuit = QuantumCircuit(2)
#         test_circuit.ry(np.pi / 2, 1)
#         test_circuit.rx(np.pi, 1)
#         test_circuit.cz(0, 1)
#         test_circuit.ry(np.pi / 2, 1)
#         test_circuit.rx(np.pi, 1)

#         self.assertEqual(
#             test_circuit.to_instruction().definition,
#             output_circuit.to_instruction().definition,
#         )

    def test_ibm_gates_2q_w_gates(self):
        
        # Qiskit Gate Applicator
        gate_applicator = self.qiskit_gate_applicator

        # Two Qubit Gate with Angles Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)

        empty_circuit = QuantumCircuit(2)
        llgate = RXX(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.rxx(np.pi, 0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(2)
        llgate = RYY(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.ryy(np.pi, 0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(2)
        llgate = RZZ(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.rzz(np.pi, 0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(2)
        llgate = RZX(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.rzx(np.pi, 0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

        empty_circuit = QuantumCircuit(2)
        llgate = CPHASE(gate_applicator, 0, 1, rotation_angle_obj)
        output_circuit = llgate.apply_gate(empty_circuit)

        test_circuit = QuantumCircuit(2)
        test_circuit.crz(np.pi, 0, 1)

        self.assertEqual(
            test_circuit.to_instruction().definition,
            output_circuit.to_instruction().definition,
        )

    def test_pyquil_gates_1q(self):
        
        # Pyquil Gate Applicator
        gate_applicator = self.pyquil_gate_applicator

        # One Qubit Gate Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)

        empty_program = Program()
        llgate = RY(gate_applicator, 0, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_RY(np.pi, 0))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

        empty_program = Program()
        llgate = RX(gate_applicator, 0, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_RX(np.pi, 0))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

        empty_program = Program()
        llgate = RZ(gate_applicator, 0, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_RZ(np.pi, 0))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

    def test_pyquil_gates_2q(self):
        
        # Pyquil Gate Applicator
        gate_applicator = self.pyquil_gate_applicator

        # Two Qubit Gate Tests
        empty_program = Program()
        llgate = CZ(gate_applicator, 0, 1)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_CZ(0, 1))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

        empty_program = Program()
        llgate = CX(gate_applicator, 0, 1)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_CX(0, 1))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

    def test_pyquil_gates_2q_w_gates(self):
        
        # Pyquil Gate Applicator
        gate_applicator = self.pyquil_gate_applicator

        # Two Qubit Gate with Angles Tests
        rotation_angle_obj = RotationAngle(lambda x: x, [], np.pi)

        empty_program = Program()
        llgate = CPHASE(gate_applicator, 0, 1, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_CPHASE(np.pi, 0, 1))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)

        empty_program = Program()
        llgate = RiSWAP(gate_applicator, 0, 1, rotation_angle_obj)
        output_program = llgate.apply_gate(empty_program)

        test_program = Program().inst(p_XY(np.pi, 0, 1))

        output_gate_names = [
            instr.name for instr in output_program if type(instr) == quilbase.Gate
        ]
        test_gate_names = [
            instr.name for instr in test_program if type(instr) == quilbase.Gate
        ]
        self.assertEqual(output_gate_names, test_gate_names)


if __name__ == "__main__":
    unittest.main()