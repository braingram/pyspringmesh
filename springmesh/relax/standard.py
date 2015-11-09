#!/usr/bin/env python


import numpy


def offset_points(mesh, forces):
    numpy.add.at(
        mesh.points,
        mesh.springs['p0'],
        forces * 0.5)
    numpy.add.at(
        mesh.points,
        mesh.springs['p0'],
        -forces * 0.5)
    #for (i, pi) in enumerate(mesh.springs['p0']):
    #    mesh.points[pi] += forces[i] * 0.5
    #for (i, pi) in enumerate(mesh.springs['p1']):
    #    mesh.points[pi] -= forces[i] * 0.5


def compute_errors(mesh):
    delta = mesh.points[mesh.springs['p1']] - mesh.points[mesh.springs['p0']]
    dist = numpy.linalg.norm(delta, axis=1)
    err = dist - mesh.springs['l']
    mesh.err = err
    mesh.dist = dist
    mesh.delta = delta
    return err, dist, delta


def compute_forces(mesh, s):
    # energy is |force| * |delta|
    err, dist, delta = compute_errors(mesh)
    forces = (
        (delta / dist[:, numpy.newaxis]) * err[:, numpy.newaxis] *
        mesh.springs['k'][:, numpy.newaxis] * s)
    mesh.forces = forces
    return forces, err


def run(mesh, n=1, s=0.01, target_error=1., target_delta=1E-6):
    """Relax a spring mesh n times"""
    last_error = numpy.inf
    reason = None
    i = 0
    while i < n:
        # 1) compute force per spring
        forces, err = compute_forces(mesh, s)
        sum_error = numpy.sum(err)
        # 2) offset points per force
        offset_points(mesh, forces)
        delta_error = last_error - sum_error
        if target_error is None and sum_error < target_error:
            reason = 'target_error reached'
            break
        if target_delta is not None and delta_error < target_delta:
            reason = 'target_delta reached'
            break
        last_error = sum_error
        i += 1
    if reason is None:
        reason = 'iterations finished'
    return sum_error, delta_error, i, reason
