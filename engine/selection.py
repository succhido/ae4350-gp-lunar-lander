"""Selection. Small file, big lever -- k is a sensitivity sweep (plan 4.3 #5)."""


def tournament_selection(rng, population, k=3):
    """Pick k individuals uniformly at random (with replacement), return the
    one with the BEST fitness.

    Sign warning: fitness() in rollout.py is MINIMISED (it returns negative
    mean return). So "best" here means LOWEST .fitness. Getting this backwards
    evolves the worst possible lander, which looks like a broken engine rather
    than a flipped comparison -- check this first when debugging.

    k controls selection pressure: k=2 is nearly random drift, k=10 converges
    fast and loses diversity.
    """
    idx = rng.integers(len(population), size=k)
    return min((population[i] for i in idx), key=lambda ind: ind.fitness)


def elitism(population, n_elites=1):
    """Return copies of the n_elites best individuals.

    Guarantees the best-so-far survives into the next generation, so the
    learning curve can never go DOWN. Cheap insurance against a good policy
    being destroyed by crossover.

    The cached fitness is carried onto the clone by hand. copy() clears it by
    design (a *mutated* child must not inherit a stale value), but an elite is
    structurally identical to its source, so its fitness is still true -- and
    re-flying it would cost a full set of rollouts for a known answer.
    """
    best = sorted(population, key=lambda ind: ind.fitness)[:n_elites]
    elites = []
    for ind in best:
        clone = ind.copy()
        clone.fitness = ind.fitness
        elites.append(clone)
    return elites
