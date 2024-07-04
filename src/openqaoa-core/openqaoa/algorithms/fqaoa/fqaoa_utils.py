from braket.circuits import Circuit
from qiskit import QuantumCircuit

import numpy as np
from scipy import linalg

def get_encoding(n):
    enc = []
    for j in range(n): 
        enc.append(j)
    return enc

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

        # encoding
        self.enc = get_encoding(n_qubits)
        
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

        if self.device_name == 'braket':
            circ = Circuit()
            for i in range(n_fermions):
                circ.x(i)
            it = (n_qubits-n_fermions)*n_fermions
            for irow in range(n_fermions-1, -1, -1):
                for icol in range(irow+1, n_qubits-n_fermions+irow+1):
                    it = it-1
                    circ.s(icol-1)
                    circ.s(icol)
                    circ.h(icol)
                    circ.cnot(icol, icol-1)
                    circ.ry(icol-1, gtheta[it])
                    circ.ry(icol, gtheta[it])
                    circ.cnot(icol, icol-1)
                    circ.h(icol)
                    circ.si(icol-1)
                    circ.si(icol)
        if self.device_name == 'qiskit.statevector_simulator':
            circ = QuantumCircuit(n_qubits)
            for i in range(n_fermions):
                circ.x(i)
            it = (n_qubits-n_fermions)*n_fermions
            for irow in range(n_fermions-1, -1, -1):
                for icol in range(irow+1, n_qubits-n_fermions+irow+1):
                    it = it-1
                    circ.s(icol-1)
                    circ.s(icol)
                    circ.h(icol)
                    circ.cnot(icol, icol-1)
                    circ.ry(gtheta[it], icol-1)
                    circ.ry(gtheta[it], icol)
                    circ.cnot(icol, icol-1)
                    circ.h(icol)
                    circ.sdg(icol-1)
                    circ.sdg(icol)
        return circ

       





