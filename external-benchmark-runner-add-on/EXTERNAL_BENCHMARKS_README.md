# External benchmark campaigns (CaQR + new Materializer Ridge)

This add-on runs the **native CaQR plan generator**, then the existing
fixed-width Repair, per-objective Materializer Ridge, SABRE router, Final Guard,
and Best-of-2 control on external QASM2 source circuits. It does **not** run DQCC.
The existing 162-case CaQR results are never used or overwritten.

## A. QASMBench, available locally on Windows

Source: https://github.com/pnnl/QASMBench (use the `small/` and `medium/`
folders under `QASMBench-master`). Source files ending `_transpiled.qasm`
are excluded to avoid evaluating already mapped/compiled inputs.

Copy the two new Python files into `scripts/` of the current
`qubit-reuse-repair` project, or unzip the provided complete project.
Open PowerShell **inside** `qubit-reuse-repair` with your existing
`(caqr-qiskit045-env)` active, then:

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) "src")
$caqr = "C:\Users\ASUS\Documents\Quantum computing\Qiskit\CaQR-main\CaQR-main"
$qasmbench = "C:\Users\ASUS\Downloads\QASMBench-master"
$out = "results_external_qasmbench_v1"

python scripts\external_caqr_campaign.py prepare `
  --source-dir "$qasmbench" `
  --caqr-root "$caqr" `
  --suite QASMBench `
  --out-dir "$out" `
  --categories small,medium `
  --max-qubits 16 `
  --max-gates 3000 `
  --holdout-percent 20

# Safe first check: five circuits, then resume all remaining later.
python scripts\external_caqr_campaign.py screen `
  --out-dir "$out" --caqr-root "$caqr" --batch-size 5

python scripts\external_caqr_campaign.py screen `
  --out-dir "$out" --caqr-root "$caqr" --batch-size 0

# Six routing jobs first (a job = circuit * objective * seed).
python scripts\external_caqr_campaign.py evaluate `
  --out-dir "$out" --model-set materializer --batch-size 6 `
  --seeds 200,211,223 --topology heavy_hex7 `
  --budget 120 --max-depth 5 --routing-timeout 420

# Resume all remaining eligible routing jobs.
python scripts\external_caqr_campaign.py evaluate `
  --out-dir "$out" --model-set materializer --batch-size 0 `
  --seeds 200,211,223 --topology heavy_hex7 `
  --budget 120 --max-depth 5 --routing-timeout 420

python scripts\external_caqr_campaign.py summary --out-dir "$out"
```

**Change** `$qasmbench` if the folder is somewhere else. Native CaQR will
copy input files to `CaQR-main/benchmarks/` with an `ex_qasmbench_...` prefix;
this is intentional. Each input stays unchanged, with its content hash
recorded in `frozen_roster.json`.

The default preparation preserves ~20% of selected **families** as an
untouched final-test split. Screening and evaluation exclude them; do not
release them while changing model/search parameters. To choose a broader
set later, use a NEW output directory and explicitly increase
`--max-qubits` and/or `--max-gates`.

### Files

- `frozen_roster.json`: the exact source paths, SHA256 hashes and train/test split.
- `external_roster.csv`, `excluded_inputs.csv`: selected and excluded inputs with reasons.
- `screening_external.csv`: CaQR reuse screening; `ready` means at least one observed valid neighbor.
- `per_circuit_seed.csv`, `per_circuit_objective.csv`: matched routing metrics.
- `summary_external.json`: summary and important scientific caveats.
- `screen/cases/`, `evaluation/cases/`: checkpoints for safe resumption.

**Controls:** topology `heavy_hex7`, objective-specific Model Ridge, SABRE,
seeds 200 / 211 / 223, and Best-of-2 seed offset +1000. Models are fixed.
No good-looking circuits are cherry-picked into the frozen evaluation set.

**Coverage:** Several circuits will have no reuse, no valid structural
neighbors, unsupported features, or native CaQR failures. Do not discard
these silently; report both coverage and gains conditional on eligibility.
The separate `training_filename_overlap` and `overlaps_caqr_input_sha` flags
are not sufficient to prove training independence; check the actual training
pairs and family overlaps as well.

## B. Your uploaded `results.zip` (MQT Bench)

Your uploaded ZIP has **16 OpenQASM 3** files: seven Grover, seven QAOA,
one graph-state and one random-circuit, at 6–12 qubits. These are **not**
directly readable by the `QuantumCircuit.from_qasm_file` QASM2 loader used
by the CaQR Qiskit 0.45 environment.

**Do not upgrade the legacy CaQR virtual environment.** For conversion, use
a *separate* modern Qiskit >=2 Python environment, e.g.:

```powershell
py -3 -m venv .venv-qasm3-converter
.\.venv-qasm3-converter\Scripts\python.exe -m pip install "qiskit>=2,<3" qiskit-qasm3-import

.\.venv-qasm3-converter\Scripts\python.exe scripts\convert_mqt_qasm3.py `
  --input "C:\Users\ASUS\Downloads\results.zip" `
  --output "mqt_qasm2_converted"
```

The converter reads QASM3, exports QASM2, attempts to parse it again, and
writes `conversion_report.csv`. Failures are explicitly reported. A
successful round-trip parse is a **syntax and register** check, not full
formal semantic equivalence. Verify a few representative circuits against
the originals before treating converted circuits as validated benchmarks.

Use a **separate** output directory for MQT results, and set `--categories ""`
because these files are at the root of the converted directory:

```powershell
$mqt = "mqt_qasm2_converted"
$out = "results_external_mqt_v1"
python scripts\external_caqr_campaign.py prepare `
  --source-dir "$mqt" --caqr-root "$caqr" --suite MQT_Bench `
  --out-dir "$out" --categories "" --max-qubits 16 --max-gates 3000 `
  --holdout-percent 20
```

Then run `screen`, `evaluate`, and `summary` as in part A, using that new
`$out`. The converter can fail for unsupported QASM3 features; never relabel
QASM3 as QASM2 or remove gates to force inclusion.

### Additional benchmark source

MQT Bench: https://mqt-bench.app/ ; choose target-independent or algorithmic,
**unmapped** circuits with OpenQASM2 export where available. Do not mix QASM2
and QASM3 without an explicit conversion and validation log.

## Scope of validation in this package

The new external runner has been smoke-tested with tiny local synthetic
QASM2 and QASM3 files. The native CaQR and routing workloads cannot be
executed in the artifact-building environment because Qiskit is not
installed there. Execute the full project tests and a few native cases in
`caqr-qiskit045-env` before launching the full set.
