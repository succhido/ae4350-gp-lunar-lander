"""Read-only analysis of sweep results already on disk in ../results/.

tree_io.py    -> load a controller: exact (results/exact_<tag>.json, if
                 extract_exact.py has been run for that tag) or an
                 approximate reconstruction from the printed 3-sig-fig
                 single_<tag>.json string otherwise (protected_div, not raw
                 Python /, either way)
extract_exact.py -> re-run evolve() for specific tags to recover the real,
                 full-precision evolved constants (single_<tag>.json only
                 ever stored a rounded string) -- writes exact_<tag>.json
plot_convergence.py -> best/mean fitness and mean tree size vs. generation
watch_tree.py -> fly an evolved controller in human-render mode
"""
