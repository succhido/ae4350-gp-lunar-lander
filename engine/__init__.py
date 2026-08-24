"""GP engine for the symbolic lunar-lander controller.

Ported/adapted from the CS4205 Lecture-4 symbolic-regression assignment.
The ONLY conceptually new piece versus that version is the fitness signal:
there it was MSE against a data table, here it is negative mean rollout
return (see ../rollout.py).

Fill order (each builds on the previous):
    tree.py       -> Node, evaluation, GROW/FULL/RHH initialization
    policy.py     -> SymbolicPolicy: pair of trees -> (f_main, f_lat)
    selection.py  -> tournament_selection
    operators.py  -> subtree crossover, point/subtree/hoist/shrink mutation
    gp.py         -> the generational loop that ties it together
"""
