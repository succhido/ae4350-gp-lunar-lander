"""Tree representation + initialization. The foundation everything else uses.

A GP individual for this project is an expression tree over:
  function set F = {+, -, *, protected_div, tanh}   (optionally sin, exp)
  terminal set T = the 8 observation variables + ephemeral random constants

obs = [x, y, vx, vy, angle, angular_vel, leg_left, leg_right]
"""

import numpy as np

# Terminal names, in the same order as the observation vector. Indexing into
# obs by position is what makes a tree callable on a raw gym observation.
VARIABLES = ("x", "y", "vx", "vy", "angle", "ang_vel", "leg_l", "leg_r")

EPS = 1e-6  # protected-division guard


# --------------------------------------------------------------------------
# Function set
# --------------------------------------------------------------------------

def protected_div(a, b):
    return a/b if np.abs(b) > EPS else 1.0


# name -> (callable, arity). Arity is what random_tree() needs to know how
# many children to generate.
FUNCTIONS = {
    "+":    (np.add, 2),
    "-":    (np.subtract, 2),
    "*":    (np.multiply, 2),
    "/":    (protected_div, 2),
    "tanh": (np.tanh, 1),
}


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------

class Node:
    """One node: either a function (has children) or a terminal (leaf).

    Represent a terminal as either
      - a variable: label is one of VARIABLES, children == []
      - a constant: label is a float,      children == []
    and a function as label in FUNCTIONS with len(children) == arity.
    """

    def __init__(self, label, children=None):
        self.label = label
        self.children = children if children is not None else []

    def is_terminal(self):
        return True if len(self.children) == 0 else False

    def evaluate(self, obs):
        """recursively compute this subtree's value for one observation.

        Three cases:
          1. label is a float          -> return it
          2. label is in VARIABLES     -> return obs[VARIABLES.index(label)]
          3. label is in FUNCTIONS     -> look up (fn, arity), recurse on each
                                          child, apply fn to the results

        This is the hot path -- it runs once per node per timestep per episode
        per individual per generation. Keep it simple.
        """
        if isinstance(self.label, float):
            return self.label

        elif self.label in VARIABLES:
            return obs[VARIABLES.index(self.label)]

        elif self.label in FUNCTIONS:
            fn, arity = FUNCTIONS[self.label]
            results = []
            for i in range(arity):
                results.append(self.children[i].evaluate(obs))
            return fn(*results)

    def size(self):
        """Total node count of this subtree (1 + sum of children sizes).

        Needed for the parsimony penalty and the bloat plots.
        """
        return 1 + sum(c.size() for c in self.children)

    def depth(self):
        """0 for a leaf, else 1 + max(child depths)."""
        if self.is_terminal():
            return 0
        return 1 + max(c.depth() for c in self.children)

    def copy(self):
        """Deep copy. Crossover/mutation MUST NOT alias parent subtrees,
        or mutating a child silently corrupts its parent.
        """
        return Node(self.label, [c.copy() for c in self.children])

    def all_nodes(self):
        """Flat list of every node in this subtree (self first).

        Crossover and mutation both work by picking a uniformly random node
        from this list, so it is worth having in one place.
        """
        nodes = [self]
        for c in self.children:
            nodes.extend(c.all_nodes())
        return nodes

    def nodes_with_depth(self, _depth=0):
        """[(node, depth-below-this-root)] for every node in this subtree.

        A Node has no parent pointer -- deliberately, since that is what makes
        the crossover swap a cheap involution. The cost is that a node cannot
        report how deep it sits. This walks DOWN from a known root instead,
        carrying the depth as it descends.

        Used by subtree_mutation: grafting a height-h tree onto a node sitting
        at depth d yields depth d + h, so the graft budget has to know d.
        """
        out = [(self, _depth)]
        for c in self.children:
            out.extend(c.nodes_with_depth(_depth + 1))
        return out

    def __str__(self):
        """Readable infix string, e.g. "tanh((vy * -0.5) + y)".

        This is the interpretability payoff -- the evolved controller gets
        printed into the report from here.
        """
        if self.is_terminal():
            # variables print as their name, constants to 3 significant figures
            return self.label if isinstance(self.label, str) else f"{self.label:.3g}"
        if len(self.children) == 1:
            return f"{self.label}({self.children[0]})"
        return f"({self.children[0]} {self.label} {self.children[1]})"

    def full_str(self):
        """Like __str__ but constants keep full float64 precision (repr, not
        :.3g). Use this when a constant needs to be quoted exactly (e.g. in
        the report) -- __str__ is lossy by design and must stay that way for
        readability, so this is a separate method rather than a flag on it.
        """
        if self.is_terminal():
            return self.label if isinstance(self.label, str) else repr(self.label)
        if len(self.children) == 1:
            return f"{self.label}({self.children[0].full_str()})"
        return f"({self.children[0].full_str()} {self.label} {self.children[1].full_str()})"

    def to_dict(self):
        """Exact JSON-serializable form (label + children, full precision --
        json round-trips python floats exactly via repr). Pairs with
        from_dict() to save/load a tree with no information loss at all,
        unlike the printed string forms above.
        """
        return {"label": self.label, "children": [c.to_dict() for c in self.children]}

    @staticmethod
    def from_dict(d):
        return Node(d["label"], [Node.from_dict(c) for c in d["children"]])


# --------------------------------------------------------------------------
# Initialization: GROW / FULL / Ramped Half-and-Half
# --------------------------------------------------------------------------

def random_terminal(rng, const_prob=0.3, const_range=(-6.0, 6.0)):
    """A leaf: either an ephemeral random constant or one of the variables."""
    if rng.random() < const_prob:
        return Node(float(rng.uniform(*const_range)))
    return Node(VARIABLES[rng.integers(len(VARIABLES))])


def random_tree(rng, max_depth, method="grow", const_prob=0.3):
    """Build one random tree.

    method="full": every branch runs to exactly max_depth.
    method="grow": each level may stop early at a terminal, so branches end
                   at varying depths.
    """
    # Base case: out of depth budget, must stop.
    if max_depth == 0:
        return random_terminal(rng, const_prob)

    # GROW may stop early; FULL never does.
    if method == "grow" and rng.random() < 0.5:
        return random_terminal(rng, const_prob)

    names = list(FUNCTIONS)
    label = names[rng.integers(len(names))]
    arity = FUNCTIONS[label][1]
    children = [random_tree(rng, max_depth - 1, method, const_prob)
                for _ in range(arity)]
    return Node(label, children)


def ramped_half_and_half(rng, n, min_depth=2, max_depth=5, const_prob=0.3):
    """Build n trees with maximum structural diversity.

    Cycles depth across min_depth..max_depth and alternates grow/full, so the
    population spans both shallow-and-bushy and deep-and-sparse shapes. Gen 0
    diversity is all evolution has to work with, so it matters more than it looks.

    Depth is the fast index (changes every i) and method the slow one (flips
    once per complete lap of the depth cycle).
    """
    depths = list(range(min_depth, max_depth + 1))
    trees = []
    for i in range(n):
        depth = depths[i % len(depths)]
        method = "full" if (i // len(depths)) % 2 else "grow"
        trees.append(random_tree(rng, depth, method, const_prob))
    return trees
