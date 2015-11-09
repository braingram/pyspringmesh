#!/usr/bin/env python


import numpy

from . import standard


def compute_forces(mesh):
    # energy is |force| * |delta|
    err, dist, delta = standard.compute_errors(mesh)
    forces = (
        (delta / dist[:, numpy.newaxis]) * err[:, numpy.newaxis] *
        mesh.springs['k'][:, numpy.newaxis])
    #forces[numpy.abs(err) < 0.1] = 0.
    mesh.forces = forces
    return forces, err


def update_velocities(mesh, dt):
    if not hasattr(mesh, 'velocities'):
        mesh.velocities = numpy.zeros_like(mesh.points)
    numpy.add.at(
        mesh.velocities,
        mesh.springs['p0'],
        mesh.forces -
        mesh.velocities[mesh.springs['p0']] *
        mesh.springs['b'][:, numpy.newaxis] * dt)
    numpy.add.at(
        mesh.velocities,
        mesh.springs['p1'],
        -mesh.forces -
        mesh.velocities[mesh.springs['p1']] *
        mesh.springs['b'][:, numpy.newaxis] * dt)
    #for (i, pi) in enumerate(mesh.springs['p0']):
    #    mesh.velocities[pi] += (
    #        mesh.forces[i] -
    #        mesh.velocities[pi] * mesh.springs['b'][i] * dt)
    #for (i, pi) in enumerate(mesh.springs['p1']):
    #    mesh.velocities[pi] += (
    #        -mesh.forces[i] -
    #        mesh.velocities[pi] * mesh.springs['b'][i] * dt)


def offset_points(mesh, dt):
    update_velocities(mesh, dt)
    mesh.points += mesh.velocities * dt


def run_n(mesh, n=1, dt=0.001):
    for i in xrange(int(n)):
        forces, err = compute_forces(mesh)
        offset_points(mesh, dt)
    return mesh
