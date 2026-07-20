# Useful commands
## Debug and Audit Performance
```bash
python3 -m cProfile -s tottime -m core.engine
```
**Result** :
```bash
36442387 function calls (36428161 primitive calls) in 16.455 seconds

   Ordered by: internal time

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
```