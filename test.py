from accelerator import *

q_f = Quadrupole(1.6)
q_d = Quadrupole(-0.8)
d = Drift(1)

lat = Lattice([q_f, d, q_d, d, q_f])
opt = lat.match(
    {q_f: "f", q_d: "f"},
    init="periodic",
    target=[0.5, None, None],
    location=q_d,
    plane="h",
)
