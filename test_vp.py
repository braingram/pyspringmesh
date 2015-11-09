#!/usr/bin/env python

import sys

import springmesh
import springmesh.render.vp

#run = lambda m: springmesh.relax.dynamic.run_n(m, n=10, dt=0.0001)

#mtype = 'triangle'
#mtype = 'grid'
mtype = 'random'
#rtype = 'standard'
rtype = 'dynamic'
size = (100, 100)
tel = 1.5
sel = 1.0
b = 0.1
k = None
n = 5
s = 0.0001
dt = 0.001


if len(sys.argv) > 1:
    args = sys.argv[1:]
    if len(args) % 2 != 0:
        raise Exception(
            "Invalid number of arguments: %s is not even" % len(args))
    i = 0
    while i < len(args):
        name = args[i]
        i += 1
        value = args[i]
        try:
            value = int(value)
        except:
            try:
                value = float(value)
            except:
                if value == 'None':
                    value = None
        i += 1
        locals()[name] = value


if not isinstance(size, (tuple, list)):
    size = (size, size)

for name in 'mtype rtype tel sel b k n s dt'.split():
    print("\t%s : %s" % (name, locals()[name]))


def srun(m):
    springmesh.relax.standard.run_n(m, n=n, s=s)


def drun(m):
    springmesh.relax.dynamic.run_n(m, n=n, dt=dt)


if mtype == 'triangle':
    m = springmesh.gen.triangle_mesh(size, k=k, sel=sel, tel=tel, b=b)
elif mtype == 'random':
    m = springmesh.gen.random_mesh(size[0], size[1], k=k, b=b)
else:
    m = springmesh.gen.grid_mesh(size, k=k, b=b, sel=sel, tel=tel)

if rtype == 'standarad':
    run = srun
else:
    run = drun
springmesh.render.vp.run(m, run=run)
