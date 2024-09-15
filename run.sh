#!/bin/bash

# ANTS: 80            (-m 80)
# NEIGHBOURS: 32      (DEFAULT)
# BACKUP CITIES: 1024 (-k 1024)
# LS PERCENTAGE: 1%   (-l 1)
# ITERATIONS: 1000    (-n 1000)
# TRIES: 1            (-r 1)

./acotsp-AVX2 -i instances/pr2392.tsp -m 100 -k 1024 -l 0 -n 1000 -r 10
