#!/usr/bin/env python

import numpy
import pylab

from .. import relax


def plot(mesh, show_k=False, show_v=False, show_f=False):
    points = mesh.points
    springs = mesh.springs
    #pylab.scatter(points['x'], points['y'], color='b')
    errs, dists, deltas = relax.standard.compute_errors(mesh)
    sv = 45. / numpy.abs(errs).max()
    pylab.scatter(
        points[:, 0], points[:, 1], color='b', s=errs * sv + 50)
    if show_k:
        k_s = 5.
        k_o = 1.
    else:
        k_s = 0.
        k_o = 1.
    for i in xrange(len(springs)):
        spring = springs[i]
        #p0 = points[int(spring['p0'])]
        #p1 = points[int(spring['p1'])]
        p0 = points[int(spring[0])]
        p1 = points[int(spring[1])]
        pylab.plot(
            [p0[0], p1[0]], [p0[1], p1[1]],
            linewidth=spring[2] * k_s + k_o,
            #linewidth=spring['k'] * k_s + k_o,
            color='g')
    if show_f and hasattr(mesh, 'forces'):
        for i in xrange(len(mesh.forces)):
            f = mesh.forces[i]
            s = mesh.springs[i]
            p0 = mesh.points[s['p0']]
            p1 = mesh.points[s['p1']]
            #l = s['l']
            #mp = (p0 + p1) / 2.
            f0 = p0 + f
            f1 = p1 - f
            fn = numpy.linalg.norm(f)
            if fn < 0.001:
                continue
            #nf = l * f / (numpy.linalg.norm(f) * 2.)
            #l0 = mp + nf
            #l1 = mp - nf
            pylab.plot(
                [p0[0], f0[0]], [p0[1], f0[1]],
                linewidth=2.,
                alpha=0.5,
                color='r')
            pylab.plot(
                [p1[0], f1[0]], [p1[1], f1[1]],
                linewidth=2.,
                alpha=0.5,
                color='r')
    if show_v and hasattr(mesh, 'velocities'):
        for i in xrange(len(mesh.velocities)):
            v = mesh.velocities[i]
            p = mesh.points[i]
            pylab.plot(
                [p[0], p[0] + v[0]], [p[1], p[1] + v[1]],
                linewidth=4.,
                alpha=0.5,
                color='orange')
