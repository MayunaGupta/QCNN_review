import pennylane as qml

# dev = qml.device('default.qubit', wires=3)

hadamard_qasm = 'OPENQASM 2.0;' \
                'include "qelib1.inc";' \
                'qreg q[1];' \
                'h q[0];'

apply_hadamard = qml.from_qasm(hadamard_qasm)

# @qml.qnode(dev)
def circuit_with_hadamards():
    apply_hadamard(wires=[0])
    apply_hadamard(wires=[1])
    qml.Hadamard(wires=[1])
    return qml.expval(qml.PauliX(0)), qml.expval(qml.PauliX(1))

result = circuit_with_hadamards()

