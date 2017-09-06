4
#This is a quantum fourier transform circuit for 4 qubits
#Assumes that all states are initialized randomly, are not normalized, and are in the computational basis

H 0
CRk 1 0
CRk 2 0
CRk 3 0
H 1
CRk 2 1
CRk 3 1
H 2
CRk 3 2
H 3
CNOT 0 3
CNOT 3 0
CNOT 0 3
CNOT 1 2
CNOT 2 1
CNOT 1 2