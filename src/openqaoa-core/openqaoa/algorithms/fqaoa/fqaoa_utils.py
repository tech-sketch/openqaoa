from braket.circuits import Circuit
from qiskit import QuantumCircuit, QuantumRegister

# yoshioka temp
from typing import List, Union, Tuple
from enum import Enum
from openqaoa.qaoa_components.ansatz_constructor.gatemap import GateMap
from openqaoa.qaoa_components.ansatz_constructor.gatemaplabel import GateMapType, GateMapLabel
from openqaoa.qaoa_components.ansatz_constructor.gates import *
from openqaoa_qiskit.backends.gates_qiskit import QiskitGateApplicator

import numpy as np
from scipy import linalg

class FermiInitialGateMap(GateMap):
    def __init__(self, n_qubits: int, n_fermions: int, gtheta: list[float]):
        self.n_qubits = n_qubits
        self.n_fermions = n_fermions
        self.gtheta = gtheta
        self.gate_label = GateMapLabel(n_qubits=n_qubits, gatemap_type=GateMapType.FIXED)
        
    @property
    def _decomposition_standard(self) -> List[Tuple]:
        excitation = [(X, [i]) for i in range(self.n_fermions)]
        givens_rotations = []
        it = (self.n_qubits-self.n_fermions)*self.n_fermions
        for irow in range(self.n_fermions-1, -1, -1):
                for icol in range(irow+1, self.n_qubits-self.n_fermions+irow+1):
                    it = it-1
                    givens_rotations.append((RZ, [icol-1, RotationAngle(lambda x: x, self.gate_label, np.pi / 2)]))
                    givens_rotations.append((RZ, [icol,   RotationAngle(lambda x: x, self.gate_label, np.pi / 2)]))
                    givens_rotations.append((RY, [icol,   RotationAngle(lambda x: x, self.gate_label, np.pi / 2)]))
                    givens_rotations.append((X, [icol]))
                    givens_rotations.append((CX, [icol, icol-1]))
                    givens_rotations.append((RY, [icol-1, RotationAngle(lambda x: x, self.gate_label, self.gtheta[it])]))
                    givens_rotations.append((RY, [icol  , RotationAngle(lambda x: x, self.gate_label, self.gtheta[it])]))
                    givens_rotations.append((CX, [icol, icol-1]))                    
                    givens_rotations.append((RY, [icol,   RotationAngle(lambda x: x, self.gate_label, np.pi / 2)]))
                    givens_rotations.append((X, [icol]))
                    givens_rotations.append((RZ, [icol-1, RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)]))
                    givens_rotations.append((RZ, [icol,   RotationAngle(lambda x: x, self.gate_label, -np.pi / 2)]))
        return [*excitation, *givens_rotations]

class FQAOAMixer:
    def __init__(self, n_qubits, n_fermions, hopping, mixer_qubit_connectivity):
        # 
        self.n_qubits = n_qubits
        self.n_fermions = n_fermions
        self.hopping = hopping
        
        # Is mixer hamiltonian already implemented in openqaoa?
        # make mixer Hamiltonian
        HM = np.zeros((n_qubits, n_qubits))
        for jw in range(1, self.n_qubits):
            HM[jw, jw-1] = 1.0
        # add periodic boundary condition for cyclic lattice
        if mixer_qubit_connectivity == 'cyclic':
            HM[n_qubits-1, 0] = -(-1.0)**n_fermions 
        HM = HM*hopping
        
        self.HM = HM
        
    def get_eigenvalue(self):
        # diagonalize HM
        eig = linalg.eigh(self.HM) # default setting lower = True
        return eig

class FQAOAInitial:
    def __init__(self, n_qubits, n_fermions, hopping, mixer_qubit_connectivity, device_name):
        self.n_qubits = n_qubits
        self.n_fermions = n_fermions
        self.hopping = hopping
        self.mixer_qubit_connectivity = mixer_qubit_connectivity
        self.device_name = device_name

        # driver Hamiltonian
        self.HM = FQAOAMixer(n_qubits, n_fermions, self.hopping, self.mixer_qubit_connectivity) 
        self.gtheta = self._get_givens_rotation_angle()
      
    def _get_fermi_orbitals(self):
        phi0 = np.zeros((self.n_fermions, self.n_qubits), dtype = np.float64)
        eig = self.HM.get_eigenvalue()
        for jw in range(self.n_qubits):
            for k in range(self.n_fermions):
                phi0[k][jw] = eig[1][jw][k]
        return phi0

    def get_statevector(self):
        phi0 = self._get_fermi_orbitals()
        statevector = np.zeros(2**self.n_qubits, dtype = np.complex128)
        cof = np.zeros((self.n_fermions, self.n_fermions), dtype = np.complex128) 
        for ibit in range(2**self.n_qubits):
            if bin(ibit)[2:].count('1') == self.n_fermions: # Conditions that satisfy the constraint conditions
                bit_str = bin(ibit)[2:].zfill(self.n_qubits)
                for i, j in enumerate([j for j, bit in enumerate(reversed(bit_str)) if bit == '1']):
                    for k in range(self.n_fermions):
                        cof[k][i] = phi0[k][j]
                statevector[ibit] = linalg.det(cof)
        if round(np.linalg.norm(statevector), 10) != 1:
            print('norm error', np.linalg.norm(statevector))
            exit()
        return(statevector)
                
    def _get_givens_rotation_angle(self):
        n_fermions = self.n_fermions
        n_qubits = self.n_qubits
        phi0 = self._get_fermi_orbitals()

        # zeroed triangular region        
        for it in range(n_fermions-1): # column to be zeroed n-1, n-2, ..., n-num+1
            icol = n_qubits - 1 - it
            for irot in range(n_fermions-1-it):
                if phi0[irot+1][icol] == 0.0:
                    for jw in range(n_qubits):
                        abc = phi0[irot][jw]
                        phi0[irot][jw] = phi0[irot+1][jw]
                        phi0[irot+1][jw] = abc
                else:
                    rate = phi0[irot][icol]/phi0[irot+1][icol]
                    for jw in range(n_qubits):
                        abc = phi0[irot][jw]
                        phi0[irot][jw]   = (abc-rate*phi0[irot+1][jw])/np.sqrt(1+rate**2)
                        phi0[irot+1][jw] = (phi0[irot+1][jw]+rate*abc)/np.sqrt(1+rate**2)
                    abc = 0.0
                    for jw in range(n_qubits):
                        abc += phi0[irot][jw]**2
                        
        # diagonalize
        gtheta = []
        it = -1
        for irow in range(n_fermions):
            for icol in range(n_qubits-n_fermions+irow, 0+irow, -1):
                it = it + 1
                if phi0[irow][icol-1] == 0.0:
                    gtheta.append(np.pi/2.0)
                else:
                    rate = phi0[irow][icol]/phi0[irow][icol-1]
                    uk =  1.0/np.sqrt(1+rate**2)
                    vk = -rate/np.sqrt(1+rate**2)
                    if   vk >= 0.0: gtheta.append(np.arccos(uk))
                    elif vk  < 0.0: gtheta.append(-np.arccos(uk))
                for ik in range(n_fermions):
                    abc = phi0[ik][icol-1] 
                    phi0[ik][icol-1] =  abc*np.cos(gtheta[it]) - phi0[ik][icol]*np.sin(gtheta[it])
                    phi0[ik][icol]   =  abc*np.sin(gtheta[it]) + phi0[ik][icol]*np.cos(gtheta[it])

        return gtheta
        
    def get_initial_circuit(self): 
        n_qubits = self.n_qubits
        n_fermions = self.n_fermions
        gtheta = self.gtheta

        FermiInitialGateMap(n_qubits, n_fermions, gtheta)
        gate_applicator = QiskitGateApplicator()
        parametric_circuit = QuantumCircuit(n_qubits)        
        for each_tuple in FermiInitialGateMap(n_qubits, n_fermions, gtheta).decomposition('standard'):
            gate = each_tuple[0](gate_applicator, *each_tuple[1])
            gate.apply_gate(parametric_circuit)
        print(parametric_circuit)
        return parametric_circuit

