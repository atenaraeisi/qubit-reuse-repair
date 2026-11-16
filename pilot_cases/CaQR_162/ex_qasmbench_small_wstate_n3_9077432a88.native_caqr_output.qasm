OPENQASM 2.0;
include "qelib1.inc";
gate cH q0,q1 { h q1; sdg q1; cx q0,q1; h q1; t q1; cx q0,q1; t q1; h q1; s q1; x q1; s q0; }
qreg q[3];
creg c[3];
u3(1.91063,0,0) q[0];
cH q[0],q[1];
ccx q[0],q[1],q[2];
x q[0];
x q[1];
cx q[0],q[1];
measure q[0] -> c[0];
measure q[1] -> c[1];
measure q[2] -> c[2];
