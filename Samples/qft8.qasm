8
#This is a quantum fourier transform circuit for 8 qubits
#Assumes that all states are initialized randomly, are not normalized, and are in the computational basis

Rx 1.6 0
Rx 1.7 1
Rx 1.8 2
Rx 1.9 3
Rx 2.0 4
Rx 2.1 5
Rx 2.2 6
Rx 2.3 7
H 0
CRk 1 0
CRk 2 0
CRk 3 0
CRk 4 0
CRk 5 0
CRk 6 0
CRk 7 0
H 1
CRk 2 1
CRk 3 1
CRk 4 1
CRk 5 1
CRk 6 1
CRk 7 1
H 2
CRk 3 2
CRk 4 2
CRk 5 2
CRk 6 2
CRk 7 2
H 3
CRk 4 3
CRk 5 3
CRk 6 3
CRk 7 3
H 4
CRk 5 4
CRk 6 4
CRk 7 4
H 5
CRk 6 5
CRk 7 5
H 6
CRk 7 6
H 7
CNOT 0 7
CNOT 7 0
CNOT 0 7
CNOT 1 6
CNOT 6 1
CNOT 1 6
CNOT 2 5
CNOT 5 2
CNOT 2 5
CNOT 3 4
CNOT 4 3
CNOT 3 4