"""Compile a GP-evolved tree's printed string back into a callable, using the
ENGINE's actual protected_div semantics -- never raw Python `/`.

Why this exists: a printed tree is infix Python-ish source (e.g. "x / (angle
- angle)" for a subtree that divides by a self-cancelling denominator). Hand-
evaluating that with raw `/` is WRONG -- the engine defines "/" as
protected_div (EPS-guarded: returns 1.0 near zero, never raises, never
produces inf/nan), so raw division can both crash on inputs the real run
never crashed on, and silently disagree with what actually got scored.
(This module replaces the one-off script that first caught that mistake.)
"""
import ast
import json
import os

import numpy as np

from engine.tree import protected_div, VARIABLES, Node

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "results")


class _DivRewriter(ast.NodeTransformer):
    """Rewrites every `a / b` into `pdiv(a, b)` before compiling."""

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.Call(func=ast.Name(id="pdiv", ctx=ast.Load()),
                            args=[node.left, node.right], keywords=[])
        return node


def compile_expr(expr):
    """Printed tree string -> f(obs) -> float, obs a length-8 array/sequence
    in VARIABLES order (x, y, vx, vy, angle, ang_vel, leg_l, leg_r)."""
    tree = ast.parse(expr, mode="eval")
    tree = _DivRewriter().visit(tree)
    ast.fix_missing_locations(tree)
    code = compile(tree, "<gp-tree>", "eval")

    def f(obs):
        env = {"pdiv": protected_div, "tanh": np.tanh}
        env.update(zip(VARIABLES, obs))
        return float(eval(code, {"__builtins__": {}}, env))
    return f


def load_controller(tag):
    """Sweep result tag -> (f_main, f_lat, meta). meta carries the numbers
    already reported by run_single.py (train/heldout return, size, lambda)
    so callers can print them alongside the render without recomputing.

    Prefers results/exact_<tag>.json if extract_exact.py has been run for
    this tag: that file stores the real Node trees (full float64 precision,
    no string round-trip), so f_main/f_lat there evaluate identically to the
    original run. meta["exact"] tells the caller which path was used.
    Falls back to reconstructing from the lossy 3-sig-fig single_<tag>.json
    string otherwise -- returns/behaviour from that path are close but not
    exact (see engine/tree.py Node.__str__).
    """
    exact_path = os.path.join(RESULTS_DIR, f"exact_{tag}.json")
    if os.path.exists(exact_path):
        with open(exact_path) as fh:
            data = json.load(fh)
        tree_main = Node.from_dict(data["tree_main_dict"])
        tree_lat = Node.from_dict(data["tree_lat_dict"])
        f_main = lambda obs: float(tree_main.evaluate(obs))
        f_lat = lambda obs: float(tree_lat.evaluate(obs))
        meta = {
            "fitness": data["fitness"],
            "size": data["size"],
            "train_return": data["train_return"],
            "heldout_return": data["heldout_return"],
            "tree_main": data["tree_main_full"],
            "tree_lat": data["tree_lat_full"],
            "parsimony": data["config"]["parsimony"],
            "exact": True,
        }
        return f_main, f_lat, meta

    path = os.path.join(RESULTS_DIR, f"single_{tag}.json")
    with open(path) as fh:
        data = json.load(fh)
    best = data["best"]
    if "tree_main_dict" in best:
        # Runs written after the lossy-string fix carry the exact trees
        # inline (run_single.py write_payload) -- same fidelity as an
        # exact_ file, no re-evolution needed.
        tree_main = Node.from_dict(best["tree_main_dict"])
        tree_lat = Node.from_dict(best["tree_lat_dict"])
        f_main = lambda obs: float(tree_main.evaluate(obs))
        f_lat = lambda obs: float(tree_lat.evaluate(obs))
        meta = dict(best)
        meta["parsimony"] = data["config"]["parsimony"]
        meta["exact"] = True
        return f_main, f_lat, meta
    f_main = compile_expr(best["tree_main"])
    f_lat = compile_expr(best["tree_lat"])
    meta = dict(best)
    meta["parsimony"] = data["config"]["parsimony"]
    meta["exact"] = False
    return f_main, f_lat, meta
