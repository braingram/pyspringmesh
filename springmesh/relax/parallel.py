#!/usr/bin/env python

import multiprocessing

import numpy


def offset_points(points, springs, forces):
    numpy.add.at(
        points,
        springs['p0'],
        forces * 0.5)
    numpy.add.at(
        points,
        springs['p1'],
        -forces * 0.5)


def compute_errors(points, springs):
    delta = points[springs['p1']] - points[springs['p0']]
    dist = numpy.linalg.norm(delta, axis=1)
    err = dist - springs['l']
    return err, dist, delta


def compute_forces(points, springs, s):
    # energy is |force| * |delta|
    err, dist, delta = compute_errors(points, springs)
    forces = (
        (delta / dist[:, numpy.newaxis]) * err[:, numpy.newaxis] *
        springs['k'][:, numpy.newaxis] * s)
    # mesh.forces = forces
    return forces, err


def _init(sa_):
    global shared_points
    shared_points = sa_


def run_partition(args):
    springs, points_shape, n, s = args
    pa = numpy.ctypeslib.as_array(shared_points.get_obj())
    pa.shape = points_shape
    for i in xrange(int(n)):
        forces, err = compute_forces(pa, springs, s)
        with shared_points.get_lock():
            offset_points(pa, springs, forces)


def splits(a, n):
    l = len(a)
    step = int(numpy.ceil(l / float(n)))
    s = 0
    e = step
    while e < l:
        yield a[s:e]
        s += step
        e += step
    if s < l:
        yield a[s:]


def run_n(mesh, n=1, s=0.01, n_splits=8):
    mesh.memmap()
    shared_points, points_shape = mesh.to_shared()
    springs = mesh.springs  # split these per job
    sps = splits(springs, n_splits)
    p = multiprocessing.Pool(
        n_splits, initializer=_init, initargs=(shared_points, ))

    args = []
    for sp in sps:
        args.append((sp, points_shape, n, s))

    p.map(run_partition, args)
    p.close()
    p.join()
    return mesh
