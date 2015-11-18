#!/usr/bin/env python

import time

import pycuda.driver
import pylab

import springmesh

size = (200, 200)
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
springmesh.relax.cuda.prepare(gm, s, bs)
(free, total) = pycuda.driver.mem_get_info()
print("mem(free): %s" % (free * 100 / float(total)))
i = 0
while n_iters > 0:
    n_iters -= 1
    t0 = time.time()
    springmesh.relax.cuda.step_n(gm, n)
    t1 = time.time()
    i += n

    print(
        "i: %s, size: %s, n: %s, gpu[%s]: %s" % (
            i, size, n, gm.grid, t1 - t0))
    (free, total) = pycuda.driver.mem_get_info()
    print("mem(free): %s" % (free * 100 / float(total)))

    if not show:
        time.sleep(delay)
        continue
    pylab.figure(1)
    pylab.clf()
    springmesh.render.mpl.plot(gm)
    pylab.title("gpu: %s" % i)
    pylab.show()
gm = springmesh.relax.cuda.finalize(gm)
