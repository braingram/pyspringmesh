#!/usr/bin/env python

import numpy

from . import base


def triangle_mesh(size=None, k=None, b=None, sel=1.0, tel=1.0):
    if size is None:
        size = (3, 3)
    nx, ny = size
    points = []
    springs = []
    to_i = lambda x, y: x + y * nx
    s32 = numpy.sqrt(3) / 2.
    if k is None:
        to_k = lambda l: 1. / l
    else:
        to_k = lambda l: k
    for yi in xrange(ny):
        xo = (yi % 2) * sel * 0.5
        y = yi * s32 * sel
        for xi in xrange(nx):
            x = xi * sel + xo
            points.append((x, y))
            if xi + 1 != nx:
                springs.append(
                    (to_i(xi, yi), to_i(xi+1, yi), to_k(tel), b, tel))
            if yi + 1 != ny:
                springs.append(
                    (to_i(xi, yi), to_i(xi, yi+1), to_k(tel), b, tel))
                if xo != 0:
                    if xi + 1 != nx:
                        springs.append(
                            (to_i(xi, yi), to_i(xi+1, yi+1),
                                to_k(tel), b, tel))
                else:
                    if xi != 0:
                        springs.append(
                            (to_i(xi, yi), to_i(xi-1, yi+1),
                                to_k(tel), b, tel))
            if yi + 2 < ny:
                if xi == 0 and xo == 0:
                    l = tel * s32 * 2
                    springs.append(
                        (to_i(xi, yi), to_i(xi, yi+2), to_k(l), b, l))
                if xi + 1 == nx and xo != 0:
                    l = tel * s32 * 2
                    springs.append(
                        (to_i(xi, yi), to_i(xi, yi+2), to_k(l), b, l))
    points = numpy.array(points, dtype='f8')
    springs = base.to_springs(springs)
    return base.Mesh(points, springs)


def grid_mesh(size=None, k=1.0, b=1.0):
    if size is None:
        size = (4, 4)
    y, x = numpy.mgrid[:size[1], :size[0]].astype('f8')
    springs = []
    to_i = lambda x, y: x + y * size[0]
    for xi in xrange(size[0]):
        for yi in xrange(size[1]):
            if xi != size[0] - 1:
                springs.append(
                    (to_i(xi, yi), to_i(xi+1, yi), k, b, 0.9))
                #if yi != size[1] - 1:
                #    springs.append(
                #        (to_i(xi, yi), to_i(xi+1, yi+1), k, 0.707))
            if yi != size[1] - 1:
                springs.append(
                    (to_i(xi, yi), to_i(xi, yi+1), k, b, 0.9))
                #if xi != 0:
                #    springs.append(
                #        (to_i(xi, yi), to_i(xi-1, yi+1), k, 0.707))
    points = numpy.vstack((x.flat, y.flat)).T
    springs = base.to_springs(springs)
    return base.Mesh(points, springs)


def random(np, ns, xr=None, yr=None):
    if xr is None:
        xr = [-10., 10.]
    if yr is None:
        yr = [-10., 10.]
    points = numpy.empty((np, 2), dtype='f4')
    points[:, 0] = numpy.random.random(np) * float(xr[1] - xr[0]) + xr[0]
    points[:, 1] = numpy.random.random(np) * float(yr[1] - yr[0]) + yr[0]
    #points = numpy.empty(n, dtype=[('x', 'f4'), ('y', 'f4')])
    #points['x'] = numpy.random.random(n) * float(xr[1] - xr[0]) + xr[0]
    #points['y'] = numpy.random.random(n) * float(yr[1] - yr[0]) + yr[0]
    #springs = numpy.zeros((n, 5), dtype='f4')
    #npoints = len(points)
    #springs[:, 0] = numpy.random.randint(0, npoints, n)
    #springs[:, 1] = numpy.random.randint(0, npoints, n)
    #springs[:, 2] = numpy.random.random(n)
    springs = numpy.zeros(ns, dtype=base.spring_dtype)
    npoints = len(points)
    springs['p0'] = numpy.random.randint(0, npoints, ns)
    springs['p1'] = numpy.random.randint(0, npoints, ns)
    springs['k'] = numpy.random.random(ns)
    springs['b'] = numpy.random.random(ns)
    springs['l'] = (numpy.random.random(ns) * 5) + 1.
    return base.Mesh(points, springs)
