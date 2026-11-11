"""Historical Experiment-3 normalize_for_routing rule, for depth parity diagnostics.

Only use on static-circuit materializations. Uses the original GidNET experiment's
reset -> measured conditional X lowering, but DOES NOT invoke GidNET compilation.
"""
def normalize_reset_for_depth(qc):
    from qiskit import QuantumCircuit, ClassicalRegister
    out=QuantumCircuit(*qc.qregs,*qc.cregs)
    out.global_phase=qc.global_phase
    last_measure_clbit={}
    clbit_owner={}
    total_resets=sum(1 for x in qc.data if x.operation.name=='reset')
    aux=None
    next_aux=0
    for item in qc.data:
        instr=item.operation
        qargs=list(item.qubits)
        cargs=list(item.clbits)
        qis=[qc.find_bit(q).index for q in qargs]
        if instr.name=='measure':
            qi=qc.find_bit(qargs[0]).index
            ci=qc.find_bit(cargs[0]).index
            previous=clbit_owner.get(ci)
            if previous is not None:
                last_measure_clbit.pop(previous,None)
            out.append(instr,qargs,cargs)
            last_measure_clbit[qi]=out.clbits[ci]
            clbit_owner[ci]=qi
        elif instr.name=='reset':
            qi=qc.find_bit(qargs[0]).index
            if qi in last_measure_clbit:
                cbit=last_measure_clbit[qi]
            else:
                if aux is None:
                    aux=ClassicalRegister(max(total_resets,1),'reset_aux')
                    out.add_register(aux)
                cbit=aux[next_aux]
                next_aux+=1
                out.measure(qargs[0],cbit)
            out.x(qargs[0]).c_if(cbit,1)
            last_measure_clbit.pop(qi,None)
        else:
            out.append(instr,qargs,cargs)
            for qi in qis:
                last_measure_clbit.pop(qi,None)
    return out
