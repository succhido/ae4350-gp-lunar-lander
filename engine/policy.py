"""SymbolicPolicy: the bridge between GP trees and rollout.py.

rollout.py already expects two plain callables f_main(obs) -> float and
f_lat(obs) -> float. Nothing in rollout.py needs to change when trees replace
the hand-written linear controller -- that was the point of building it first.
"""

from .tree import ramped_half_and_half


class SymbolicPolicy:
    """One GP individual = a PAIR of trees, one per engine.

    tree_main -> f_main -> action[0] (main engine)
    tree_lat  -> f_lat  -> action[1] (lateral engines)

    Note rollout.py applies the tanh squashing itself, so the trees output
    RAW values here. Do not tanh twice.
    """

    def __init__(self, tree_main, tree_lat):
        self.tree_main = tree_main
        self.tree_lat = tree_lat
        self.fitness = None   # cached; set by the evaluator in gp.py

    def as_callables(self):
        """Return (f_main, f_lat), each f(obs) -> float, ready for rollout()."""
        return (lambda obs: float(self.tree_main.evaluate(obs)),
                lambda obs: float(self.tree_lat.evaluate(obs)))

    def size(self):
        """Total node count across BOTH trees.

        This is the |f_main| + |f_lat| term in the parsimony penalty
        (plan section 3.3).
        """
        return self.tree_main.size() + self.tree_lat.size()

    def copy(self):
        """Deep copy both trees. Fitness resets to None -- a mutated child's
        inherited fitness value would be a lie.
        """
        return SymbolicPolicy(self.tree_main.copy(), self.tree_lat.copy())

    def __str__(self):
        return (f"a_main = tanh({self.tree_main})\n"
                f"a_lat  = tanh({self.tree_lat})")


def random_population(rng, n, min_depth=2, max_depth=5):
    """Build n SymbolicPolicy individuals.

    Generates 2n trees and shuffles before pairing. Without the shuffle, the
    two trees of one individual would share a (depth, method) slot in the RHH
    schedule -- both engines starting from the same shape at gen 0 is a
    correlation evolution has no reason to be handed.
    """
    trees = ramped_half_and_half(rng, 2 * n, min_depth, max_depth)
    rng.shuffle(trees)
    return [SymbolicPolicy(trees[2 * i], trees[2 * i + 1]) for i in range(n)]
