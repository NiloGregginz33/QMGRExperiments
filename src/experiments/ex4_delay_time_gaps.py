"""
ex4_delay_time_gaps.py

Recreates the ex4 time-gap construction but replaces idle identity gates with
explicit Delay instructions. Runs via CGPTFactory.run (no Aer) and logs results
under experiment_logs/ex4_delay_time_gaps/instance_YYYYMMDD_HHMMSS/.

Usage (simulation):
  python src/experiments/ex4_delay_time_gaps.py --num_injections 10 \
      --num_radiation_qubits 4 --gap_ns 100 --device simulator --shots 2048

Usage (hardware):
  python src/experiments/ex4_delay_time_gaps.py --device ibm_brisbane
"""

import os
import json
from datetime import datetime
import argparse

from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister, Delay
from qiskit import transpile
try:
    from qiskit_ibm_runtime import QiskitRuntimeService
except Exception:
    QiskitRuntimeService = None

# Ensure src/ is on sys.path for CGPTFactory import
import sys, os as _os
_src_path = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), '..'))
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)

# Use CGPTFactory runner per repo rules
from CGPTFactory import run as cgpt_run


def build_circuit_with_delays(num_injections: int, num_radiation_qubits: int, gap_ns: int) -> QuantumCircuit:
    """Build a black-hole + radiation circuit with explicit Delay time gaps.

    - Qubit 0: black hole
    - Qubits 1..N: radiation
    - Alternating charge injection (X, Z) on q0
    - Sequential entangle to radiation qubits
    - Insert Delay(gap_ns, 'ns') after each injection/entangle block
    """
    n = int(num_radiation_qubits) + 1
    qc = QuantumCircuit(n)

    for i in range(int(num_injections)):
        # Alternating charge injection
        if i % 2 == 0:
            qc.x(0)
        else:
            qc.z(0)

        # Entangle BH with each radiation qubit
        for j in range(1, n):
            qc.h(0)
            qc.cx(0, j)

        # Explicit time gap using a Delay instruction (ns units)
        if gap_ns and gap_ns > 0:
            # Apply delay on the black-hole qubit (and minimally on all to keep scheduler busy)
            d = Delay(int(gap_ns), unit="ns")
            qc.append(d, [0])

    # Measure all
    creg = ClassicalRegister(n)
    qc.add_register(creg)
    qc.measure(range(n), range(n))
    return qc


def _coerce_device(device: str) -> str:
    d = (device or "").lower()
    if d in {"sim", "simulator", "simulation"}:
        return "simulation"
    return device


def _load_latest_mi_json(dir_path: str):
    try:
        files = [f for f in os.listdir(dir_path) if f.startswith("cgpt_mi_values_") and f.endswith(".json")]
        if not files:
            return None
        files.sort()
        target = os.path.join(dir_path, files[-1])
        with open(target, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    p = argparse.ArgumentParser(description="ex4 with explicit Delay time gaps (no Aer)")
    p.add_argument("--num_injections", type=int, default=10)
    p.add_argument("--num_radiation_qubits", type=int, default=4)
    p.add_argument("--gap_ns", type=int, default=100, help="Delay gap in ns after each block (single-run)")
    p.add_argument("--gap_ns_list", type=str, default="", help="Comma-separated gaps in ns, e.g. '50,100,200'")
    p.add_argument("--gap_ns_start", type=int, default=None)
    p.add_argument("--gap_ns_stop", type=int, default=None)
    p.add_argument("--gap_ns_step", type=int, default=None)
    p.add_argument("--device", type=str, default="simulation", help="'simulation' or IBM backend name")
    p.add_argument("--shots", type=int, default=2048)
    args = p.parse_args()

    device = _coerce_device(args.device)

    # Setup experiment logs
    experiment_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "experiment_logs", "ex4_delay_time_gaps"))
    os.makedirs(base, exist_ok=True)
    instance_dir = os.path.join(base, f"instance_{experiment_timestamp}")
    os.makedirs(instance_dir, exist_ok=True)

    # Build sweep list if requested
    gaps = None
    if args.gap_ns_list:
        try:
            gaps = [int(x.strip()) for x in args.gap_ns_list.split(",") if x.strip()]
        except Exception:
            gaps = None
    if gaps is None and args.gap_ns_start is not None and args.gap_ns_stop is not None and args.gap_ns_step:
        try:
            gaps = list(range(int(args.gap_ns_start), int(args.gap_ns_stop) + 1, int(args.gap_ns_step)))
        except Exception:
            gaps = None
    if gaps is None:
        gaps = [int(args.gap_ns)]

    sweep_rows = []

    for gap in gaps:
        gap_dir = os.path.join(instance_dir, f"gap_{int(gap):05d}ns")
        os.makedirs(gap_dir, exist_ok=True)

        # Expose per-gap log dir for CGPTFactory auto-save hook
        import __main__ as _m
        setattr(_m, "experiment_log_dir", gap_dir)

        # Build and run
        qc = build_circuit_with_delays(args.num_injections, args.num_radiation_qubits, int(gap))

        # If running on hardware, attempt a backend-aware transpile to satisfy coupling map
        if device and device not in {"sim", "simulation", "simulator"} and QiskitRuntimeService is not None:
            try:
                svc = QiskitRuntimeService()
                backend = svc.backend(device)
                qc = transpile(qc, backend=backend, optimization_level=1)
            except Exception:
                pass

        out = cgpt_run(qc, device=device, shots=args.shots)
        counts = (out or {}).get("counts", {})

        # Save per-gap results
        results = {
            "spec": {
                "num_injections": args.num_injections,
                "num_radiation_qubits": args.num_radiation_qubits,
                "gap_ns": int(gap),
                "device": device,
                "shots": args.shots,
            },
            "timestamp": experiment_timestamp,
            "counts": counts,
        }
        with open(os.path.join(gap_dir, "results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Try to load MI autosave, if any
        mi = _load_latest_mi_json(gap_dir)
        mi_vals = (mi or {}).get("mutual_information") if mi else None
        sweep_rows.append({"gap_ns": int(gap), "mi": mi_vals, "counts_nonzero": sum(counts.values())})

        # Optional per-gap summary
        with open(os.path.join(gap_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join([
                "ex4_delay_time_gaps (per-gap):", f"gap_ns={int(gap)}", f"device={device}", f"shots={args.shots}",
                "MI autosave: present" if mi_vals else "MI autosave: none",
            ]))

    # Write sweep-level summary/results at instance root
    with open(os.path.join(instance_dir, "sweep_results.json"), "w", encoding="utf-8") as f:
        json.dump({"gaps": sweep_rows}, f, indent=2, ensure_ascii=False)

    with open(os.path.join(instance_dir, "summary.txt"), "w", encoding="utf-8") as f:
        header = [
            "ex4_delay_time_gaps: Delay sweep (no Aer)",
            f"num_injections={args.num_injections}",
            f"num_radiation_qubits={args.num_radiation_qubits}",
            f"device={device}",
            f"shots={args.shots}",
            f"gaps_ns={','.join(str(r['gap_ns']) for r in sweep_rows)}",
            "", "Each gap run logs under a per-gap subdirectory.",
            "MI autosaves from CGPTFactory are stored alongside per-gap results.",
        ]
        f.write("\n".join(header))

    print(f"[OK] Sweep logged to: {instance_dir}")


if __name__ == "__main__":
    main()
