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
    def __init__(self, n, num, hopping, mixer_qubit_connectivity):
        self.n = n
        self.num = num
        self.hopping = hopping

        # make driver Hamiltonian
        Hd = np.zeros((n, n))
        for jw in range(1, self.n):
            Hd[jw, jw-1] = 1.0
        # add periodic boundary condition for cyclic lattice
        if mixer_qubit_connectivity == 'cyclic':
            Hd[n-1, 0] = -(-1.0)**num 
        Hd = Hd*hopping
        
        self.Hd = Hd
        
    def get_eigenvalue(self):
        # diagonalize Hd
        eig = linalg.eigh(self.Hd) # default setting lower = True
        return eig

class FQAOAInitial:
    def __init__(self, n, num, hopping, mixer_qubit_connectivity, backend):
        self.n = n
        self.num = num
        self.hopping = hopping
        self.mixer_qubit_connectivity = mixer_qubit_connectivity
        self.backend = backend

        # encoding
        self.enc = get_encoding(n)
        
        # driver Hamiltonian
        self.Hd = FQAOAMixer(n, num, self.hopping, self.mixer_qubit_connectivity) 
        self.gtheta = self.get_givens_rotation_angle()
      
    def get_wave_function(self):
        phi0 = np.zeros((self.num, self.n), dtype = np.float64)
        eig = self.Hd.get_eigenvalue()
        for jw in range(self.n):
            for k in range(self.num):
                phi0[k][jw] = eig[1][jw][k]
        return phi0

    def get_statevector(self):
        phi0 = self.get_wave_function()
        statevector = np.zeros(2**self.n, dtype = np.float64)
        cof = np.zeros((self.num, self.num), dtype = np.float64) 
        for ibit in range(2**self.n):
            if bin(ibit)[2:].count('1') == self.num: # Conditions that satisfy the constraint conditions
                bit_str = bin(ibit)[2:].zfill(self.n)
                for i, j in enumerate([j for j, bit in enumerate(reversed(bit_str)) if bit == '1']):
                    for k in range(self.num):
                        cof[k][i] = phi0[k][j]
                statevector[ibit] = linalg.det(cof)
        if round(np.linalg.norm(statevector), 10) != 1:
            print('norm error', np.linalg.norm(statevector))
            exit()
        return(statevector)
                
    def get_givens_rotation_angle(self):
        num = self.num
        n = self.n
        phi0 = self.get_wave_function()

        # zeroed triangular region        
        for it in range(num-1): # column to be zeroed n-1, n-2, ..., n-num+1
            icol = n - 1 - it
            for irot in range(num-1-it):
                if phi0[irot+1][icol] == 0.0:
                    for jw in range(n):
                        abc = phi0[irot][jw]
                        phi0[irot][jw] = phi0[irot+1][jw]
                        phi0[irot+1][jw] = abc
                else:
                    rate = phi0[irot][icol]/phi0[irot+1][icol]
                    for jw in range(n):
                        abc = phi0[irot][jw]
                        phi0[irot][jw]   = (abc-rate*phi0[irot+1][jw])/np.sqrt(1+rate**2)
                        phi0[irot+1][jw] = (phi0[irot+1][jw]+rate*abc)/np.sqrt(1+rate**2)
                    abc = 0.0
                    for jw in range(n):
                        abc += phi0[irot][jw]**2
                        
        # diagonalize
        gtheta = []
        it = -1
        for irow in range(num):
            for icol in range(n-num+irow, 0+irow, -1):
                it = it + 1
                if phi0[irow][icol-1] == 0.0:
                    gtheta.append(np.pi/2.0)
                else:
                    rate = phi0[irow][icol]/phi0[irow][icol-1]
                    uk =  1.0/np.sqrt(1+rate**2)
                    vk = -rate/np.sqrt(1+rate**2)
                    if   vk >= 0.0: gtheta.append(np.arccos(uk))
                    elif vk  < 0.0: gtheta.append(-np.arccos(uk))
                for ik in range(num):
                    abc = phi0[ik][icol-1] 
                    phi0[ik][icol-1] =  abc*np.cos(gtheta[it]) - phi0[ik][icol]*np.sin(gtheta[it])
                    phi0[ik][icol]   =  abc*np.sin(gtheta[it]) + phi0[ik][icol]*np.cos(gtheta[it])

        return gtheta
        
    def get_initial_circuit(self):
        n = self.n
        num = self.num
        enc = self.enc
        gtheta = self.gtheta

        if self.backend == 'braket':
            circ = Circuit()
            for i in range(num):
                circ.x(enc[i])
            it = (n-num)*num
            for irow in range(num-1, -1, -1):
                for icol in range(irow+1, n-num+irow+1):
                    it = it-1
                    circ.s(enc[icol-1])
                    circ.s(enc[icol])
                    circ.h(enc[icol])
                    circ.cnot(enc[icol], enc[icol-1])
                    circ.ry(enc[icol-1], gtheta[it])
                    circ.ry(enc[icol], gtheta[it])
                    circ.cnot(enc[icol], enc[icol-1])
                    circ.h(enc[icol])
                    circ.si(enc[icol-1])
                    circ.si(enc[icol])
        if self.backend == 'qiskit':
            circ = QuantumCircuit(n)
            for i in range(num):
                circ.x(enc[i])
            it = (n-num)*num
            for irow in range(num-1, -1, -1):
                for icol in range(irow+1, n-num+irow+1):
                    it = it-1
                    circ.s(enc[icol-1])
                    circ.s(enc[icol])
                    circ.h(enc[icol])
                    circ.cnot(enc[icol], enc[icol-1])
                    circ.ry(gtheta[it], enc[icol-1])
                    circ.ry(gtheta[it], enc[icol])
                    circ.cnot(enc[icol], enc[icol-1])
                    circ.h(enc[icol])
                    circ.sdg(enc[icol-1])
                    circ.sdg(enc[icol])
        return gtheta, circ

       





