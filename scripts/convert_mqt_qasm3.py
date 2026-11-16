#!/usr/bin/env python3
"""Convert MQT Bench OpenQASM3 to OpenQASM2 with Qiskit >=2 in a SEPARATE env.

No original files are edited. Each converted QASM2 is immediately parsed back
by Qiskit; the conversion is logged. Conversion failures are NOT silently kept.
Run outside the legacy project's Qiskit 0.45 environment.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import zipfile
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True,help='MQT results.zip or a folder of *.qasm')
    p.add_argument('--output',type=Path,required=True,help='New, separate directory of QASM2')
    a=p.parse_args()
    try:
        from qiskit import qasm2, qasm3, QuantumCircuit
    except ImportError as e:
        raise SystemExit('Install Qiskit >=2 and qiskit-qasm3-import in a separate Python environment: '+str(e))
    if a.input.is_file() and a.input.suffix.lower()=='.zip':
        with zipfile.ZipFile(a.input) as z:
            sources=[(n,z.read(n)) for n in z.namelist() if n.lower().endswith('.qasm') and not n.startswith('/')]
    elif a.input.is_dir():
        sources=[(p.relative_to(a.input).as_posix(),p.read_bytes()) for p in sorted(a.input.rglob('*.qasm'))]
    else:raise SystemExit('Input must be a ZIP or directory')
    a.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,bytes_ in sources:
        row=dict(source=name,status='failed',error='')
        try:
            content=bytes_.decode('utf-8-sig')
            # qasm3.loads uses the optional qiskit-qasm3-import dependency.
            qc=qasm3.loads(content)
            qasm2_text=qasm2.dumps(qc)
            # The reverse parse checks the written syntax is usable by Qiskit.
            parsed=QuantumCircuit.from_qasm_str(qasm2_text)
            if qc.num_qubits!=parsed.num_qubits or qc.num_clbits!=parsed.num_clbits:
                raise ValueError('QASM2 roundtrip changed register widths')
            stem=Path(name).stem
            digest=hashlib.sha256(bytes_).hexdigest()[:10]
            dest=a.output/f'{stem}_{digest}.qasm'
            dest.write_text(qasm2_text,encoding='utf-8')
            row.update(status='converted',output=str(dest),qubits=parsed.num_qubits,
                       instructions=len(parsed.data),source_sha256=hashlib.sha256(bytes_).hexdigest(),
                       output_sha256=hashlib.sha256(qasm2_text.encode()).hexdigest())
        except Exception as e:
            row['error']=f'{type(e).__name__}: {str(e)[:500]}'
        rows.append(row)
        print(name,':',row['status'],row['error'][:120],flush=True)
    with (a.output/'conversion_report.csv').open('w',encoding='utf-8-sig',newline='') as f:
        keys=list(dict.fromkeys(k for row in rows for k in row))
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader();writer.writerows(rows)
    print('Converted:',sum(r['status']=='converted' for r in rows),'/',len(rows))
    print('Report:',a.output/'conversion_report.csv')


if __name__=='__main__':main()
