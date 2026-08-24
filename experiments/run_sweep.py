"""Parallel sweep runner: launches one run_single.py subprocess per
(parsimony lambda, GP seed) cell, capped at --max-workers concurrent
processes.

Resumable by construction: a cell is skipped if its result JSON already
exists, so re-invoking after a partial run (or a crash) only fills the gaps.
This is what let seeds 1's lambda={0, 0.2, 0.5} cells (already run by hand)
get skipped for free.

Run me (background, unbuffered so the log updates live):
    python experiments/run_sweep.py

    --dry-run   print the grid and which cells are already done, then exit
"""

import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
RESULTS_DIR = os.path.join(ROOT, "results")
RUN_SINGLE = os.path.join(ROOT, "experiments", "run_single.py")
PYTHON = sys.executable


def tag_for(n_pop, n_gen, gp_seed, n_train_seeds, parsimony):
    # MUST match run_single.py's default tag exactly, or "already done"
    # detection silently breaks and cells get rerun for nothing.
    return (f"pop{n_pop}_gen{n_gen}_seed{gp_seed}"
            f"_ts{n_train_seeds}_lam{parsimony:g}")


def build_grid(lambdas, gp_seeds, n_pop, n_gen, n_train_seeds):
    return [dict(n_pop=n_pop, n_gen=n_gen, seed=gs,
                 n_train_seeds=n_train_seeds, parsimony=lam)
            for lam in lambdas for gs in gp_seeds]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.5])
    ap.add_argument("--gp-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--n-pop", type=int, default=200)
    ap.add_argument("--n-gen", type=int, default=50)
    ap.add_argument("--n-train-seeds", type=int, default=20)
    ap.add_argument("--heldout-every", type=int, default=0,
                    help="forwarded to run_single.py; record the champion's "
                         "held-out return every N generations (0 = off). "
                         "Set 1 for the convergence experiment.")
    ap.add_argument("--max-workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    grid = build_grid(args.lambdas, args.gp_seeds, args.n_pop, args.n_gen,
                      args.n_train_seeds)

    pending = []
    done = []
    for cfg in grid:
        tag = tag_for(cfg["n_pop"], cfg["n_gen"], cfg["seed"],
                      cfg["n_train_seeds"], cfg["parsimony"])
        out_path = os.path.join(RESULTS_DIR, f"single_{tag}.json")
        if os.path.exists(out_path):
            done.append(tag)
        else:
            pending.append((tag, cfg))

    print(f"grid: {len(grid)} cells total, {len(done)} already done, "
          f"{len(pending)} to run")
    print(f"max concurrent workers: {args.max_workers}\n")
    for tag in done:
        print(f"  done   : {tag}")
    for tag, _ in pending:
        print(f"  pending: {tag}")

    if args.dry_run or not pending:
        return

    running = {}  # Popen -> (tag, log file handle)
    queue = list(pending)
    t0 = time.time()
    completed = 0
    failed = []

    print(f"\nlaunching {len(pending)} runs, {args.max_workers} at a time...\n",
          flush=True)

    while queue or running:
        while queue and len(running) < args.max_workers:
            tag, cfg = queue.pop(0)
            log_path = os.path.join(RESULTS_DIR, f"run_{tag}.log")
            cmd = [PYTHON, "-u", RUN_SINGLE,
                  "--n-pop", str(cfg["n_pop"]),
                  "--n-gen", str(cfg["n_gen"]),
                  "--seed", str(cfg["seed"]),
                  "--n-train-seeds", str(cfg["n_train_seeds"]),
                  "--parsimony", str(cfg["parsimony"]),
                  "--heldout-every", str(args.heldout_every)]
            fh = open(log_path, "w")
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
            running[proc] = (tag, fh)
            print(f"[{time.time()-t0:7.1f}s] launched {tag}  (pid {proc.pid})",
                  flush=True)

        time.sleep(10)
        for proc in list(running):
            ret = proc.poll()
            if ret is not None:
                tag, fh = running.pop(proc)
                fh.close()
                completed += 1
                if ret == 0:
                    status = "OK"
                else:
                    status = f"FAILED (exit {ret})"
                    failed.append(tag)
                print(f"[{time.time()-t0:7.1f}s] finished {tag}  {status}  "
                      f"({completed}/{len(pending)} done)", flush=True)

    elapsed = time.time() - t0
    print(f"\nsweep complete: {completed} runs in {elapsed/60:.1f} min "
          f"({len(failed)} failed)", flush=True)
    if failed:
        print("failed tags:")
        for tag in failed:
            print(f"  {tag}")
    print("SWEEP DONE")


if __name__ == "__main__":
    main()
