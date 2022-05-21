Two repeater detection helpers are provided

1. MESS
    Detect repeaters with MESS.
    1.1 Selection of repeater candidates
        (1) Construct your final MESS catalog
        (2) Clustering with rather tolerant parameters, e.g. CC>=0.8, dt_sp<=0.03s
    1.2 Final detection
        (1) Run MESS with repeater candidates as template
        (2) Clustering with rather strict parameters, e.g. CC>=0.9, dt_sp<=0.01s