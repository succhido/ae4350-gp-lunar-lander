"""Variation operators: crossover + the four mutations.

ROLE-MATCHING RULE (project-specific, not in the CS4205 version):
an individual holds two trees with DIFFERENT jobs. tree_main drives the main
engine, tree_lat the lateral ones. Crossover must swap main-with-main and
lat-with-lat. Swapping a lateral subtree into a main-engine tree mixes two
unrelated control laws and destroys both parents' structure.

All four mutations act IN PLACE on the individual they are handed and return
it. The caller (gp.py) only ever hands them a fresh child, never a live parent.
Each clears .fitness, so a mutated individual can never carry a stale score.
"""

import numpy as np

from .tree import FUNCTIONS, VARIABLES, random_terminal, random_tree


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------

def _pick_node(rng, tree):
    """A uniformly random node from this subtree (tree itself included)."""
    nodes = tree.all_nodes()
    return nodes[rng.integers(len(nodes))]


def _pick_node_with_depth(rng, tree):
    """A uniformly random (node, depth-from-root) pair from this tree."""
    pairs = tree.nodes_with_depth()
    return pairs[rng.integers(len(pairs))]


def _pick_tree(rng, individual):
    """One of the individual's two trees, at random."""
    return individual.tree_main if rng.random() < 0.5 else individual.tree_lat


def _swap(a, b):
    """Exchange the CONTENTS of two nodes.

    Nodes hold no parent pointer, so we cannot re-link them from above.
    Swapping label+children in place achieves the same thing: whoever already
    references node `a` now sees b's subtree, and vice versa. It is also its
    own inverse, which is what makes the depth-cap retry below trivial.
    """
    a.label, b.label = b.label, a.label
    a.children, b.children = b.children, a.children


def _graft(node, replacement):
    """Overwrite `node` in place with the contents of `replacement`."""
    node.label, node.children = replacement.label, replacement.children


# --------------------------------------------------------------------------
# Crossover
# --------------------------------------------------------------------------

def subtree_crossover(rng, parent_a, parent_b, max_depth=8, tries=3):
    """Return two children.

    Copy FIRST, then swap -- swapping references out of live parents would
    corrupt the population. For each ROLE independently, pick a random node in
    each child's tree and exchange the subtrees rooted there.

    If a swap pushes either tree past max_depth it is undone and retried; after
    `tries` failures that role is left as the plain copy. The hard depth cap is
    the last line of defence against bloat.
    """
    ca, cb = parent_a.copy(), parent_b.copy()

    for tree_a, tree_b in ((ca.tree_main, cb.tree_main),
                           (ca.tree_lat, cb.tree_lat)):
        for _ in range(tries):
            node_a = _pick_node(rng, tree_a)
            node_b = _pick_node(rng, tree_b)
            _swap(node_a, node_b)
            if tree_a.depth() <= max_depth and tree_b.depth() <= max_depth:
                break
            _swap(node_a, node_b)   # over the cap -- undo and try again

    return ca, cb


# --------------------------------------------------------------------------
# Mutations
# --------------------------------------------------------------------------

def point_mutation(rng, individual, p_node=0.1):
    """With probability p_node per node, replace that node's LABEL with
    another of the same arity.

    function -> different function of equal arity ("+" -> "*", not "+" -> tanh)
    variable -> different variable
    constant -> perturbed by a unit-normal step

    Structure-preserving: the tree shape is untouched, only the labels move.
    """
    for tree in (individual.tree_main, individual.tree_lat):
        for node in tree.all_nodes():
            if rng.random() >= p_node:
                continue
            if isinstance(node.label, float):
                node.label = float(node.label + rng.normal(0.0, 1.0))
            elif node.label in VARIABLES:
                node.label = VARIABLES[rng.integers(len(VARIABLES))]
            else:
                arity = FUNCTIONS[node.label][1]
                same = [name for name, (_, a) in FUNCTIONS.items() if a == arity] # makes list of functions with identical arities
                node.label = same[rng.integers(len(same))]
                

    individual.fitness = None
    return individual


def subtree_mutation(rng, individual, max_depth=5, max_depth_cap=8):
    """Replace a random subtree with a freshly generated random tree.

    The most disruptive operator -- this is the main source of NEW genetic
    material once the population has converged.

    The graft budget is depth-aware. Grafting a height-h tree onto a node at
    depth d gives a tree of depth d + h, so the budget is capped at
    (max_depth_cap - d). Without that, subtree mutation is the one operator
    with no depth ceiling at all: measured at depth 13 against a cap of 8,
    breaching on ~2.7% of calls once crossover has already deepened the trees.
    A budget of 0 is legitimate -- random_tree returns a bare terminal, which
    is the only graft that fits.
    """
    tree = _pick_tree(rng, individual)
    node, node_depth = _pick_node_with_depth(rng, tree)
    budget = min(max_depth, max(0, max_depth_cap - node_depth))
    _graft(node, random_tree(rng, budget, "grow"))

    individual.fitness = None
    return individual


def hoist_mutation(rng, individual):
    """Pick a random subtree, pick a random subtree INSIDE it, and replace the
    outer with the inner.

    Strictly shrinks the tree. Bloat control operator #1.

    Both picks are restricted rather than uniform-over-all-nodes: `outer` must
    have children and `inner` must be a PROPER descendant. Sampling either
    freely makes hoist a no-op most of the time -- a leaf outer has nothing
    inside it, and inner == outer changes nothing -- which quietly drops the
    operator's real rate far below p_hoist and weakens the bloat control the
    ablation study is meant to measure.
    """
    tree = _pick_tree(rng, individual)
    internal = [n for n in tree.all_nodes() if not n.is_terminal()] # if n is not a leaf, add to the list of internal

    if internal:                       # a bare-terminal tree cannot be hoisted
        outer = internal[rng.integers(len(internal))]
        descendants = outer.all_nodes()[1:]     # [0] is outer itself
        _graft(outer, descendants[rng.integers(len(descendants))])

    individual.fitness = None
    return individual


def shrink_mutation(rng, individual):
    """Replace a random subtree with a single random terminal.

    Also strictly shrinking. Bloat control operator #2. Ablating hoist+shrink
    is a nice report result (plan section 6, week 6).
    """
    node = _pick_node(rng, _pick_tree(rng, individual))
    _graft(node, random_terminal(rng))

    individual.fitness = None
    return individual


def mutate(rng, individual, p_point=0.4, p_subtree=0.3, p_hoist=0.15,
           p_shrink=0.15, max_depth=5, max_depth_cap=8):
    """Choose ONE of the four mutations by the given probabilities and apply it.

    Keeping the choice here (rather than applying all four) means the sweep
    over mutation rate in plan 4.3 #4 has a single knob to turn.
    """
    ops = (point_mutation, subtree_mutation, hoist_mutation, shrink_mutation)
    p = np.array([p_point, p_subtree, p_hoist, p_shrink], dtype=float)
    op = ops[int(rng.choice(len(ops), p=p / p.sum()))]

    if op is subtree_mutation:
        return op(rng, individual, max_depth=max_depth,
                  max_depth_cap=max_depth_cap)
    return op(rng, individual)
