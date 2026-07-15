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
   548272    4.100    0.000    4.119    0.000 sensors.py:216(_detect_creature)
    34267    2.611    0.000   10.818    0.000 sensors.py:366(perceive)
      393    0.837    0.002    2.319    0.006 renderer.py:144(render)
   274136    0.662    0.000    0.679    0.000 sensors.py:309(_detect_pheromone)
```