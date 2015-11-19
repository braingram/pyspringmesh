#!/usr/bin/env python

import time

import numpy
import pycuda.driver
import pylab

import springmesh

size = (1000, 1000)
#size = (200, 200)
#size = (100, 100)
n = 100
bs = (1024, 1, 1)
show = False
n_iters = 10
s = 0.01
delay = 0.5

d = pycuda.driver.Device(0)
(free, total) = pycuda.driver.mem_get_info()
print("size: %s" % (size, ))
print("n: %s" % n)
print("bs: %s" % (bs, ))
print("n_iters: %s" % (n_iters, ))
print("s: %s" % s)
(free, total) = pycuda.driver.mem_get_info()
print("mem(free): %s" % (free * 100 / float(total)))


gm = springmesh.gen.triangle_mesh(size, 0.01, 0.01, sel=0.5)
springmesh.relax.cuda.prepare(gm, bs)
(free, total) = pycuda.driver.mem_get_info()
print("mem(free): %s" % (free * 100 / float(total)))
print("springs[n]: %s" % (len(gm.springs), ))
print("points[n]: %s" % (len(gm.points), ))
i = 0
ts = []
while n_iters > 0:
    n_iters -= 1
    t0 = time.time()
    springmesh.relax.cuda.step_n(gm, n=n, s=s)
    t1 = time.time()
    i += n
    ts.append(t1 - t0)

    print(
        "i: %s, size: %s, n: %s, gpu[%s]: %s" % (
            i, size, n, gm.grid, t1 - t0))
    (free, total) = pycuda.driver.mem_get_info()
    print("mem(free): %s" % (free * 100 / float(total)))

    if not show:
        #time.sleep(delay)
        continue
    pylab.figure(1)
    pylab.clf()
    springmesh.render.mpl.plot(gm)
    pylab.title("gpu: %s" % i)
    pylab.show()
gm = springmesh.relax.cuda.finalize(gm)

avg_t = numpy.mean(ts)
print("Size: %s" % (size, ))
print("Average time per run[%s]: %s" % (n, avg_t))
print("Average time per step: %s" % (avg_t / float(n), ))
