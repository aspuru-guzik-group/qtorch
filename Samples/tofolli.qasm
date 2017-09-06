3
def1 S Samples/s.gate
def1 T Samples/t.gate
def1 Tdag Samples/tdagger.gate
H 2
CNOT 1 2
Tdag 2
CNOT 0 2
T 2
CNOT 1 2
Tdag 2
CNOT 0 2
Tdag 1
T 2
CNOT 0 1
H 2
Tdag 1
CNOT 0 1
T 0
S 1