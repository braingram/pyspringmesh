#!/usr/bin/env python

import atexit
import multiprocessing.sharedctypes
import os
import shutil
import tempfile

import joblib
import numpy


spring_dtype = [
    ('p0', 'i8'), ('p1', 'i8'), ('k', 'f8'), ('b', 'f8'), ('l', 'f8'),
]


def to_springs(arr):
    return numpy.array(arr, dtype=spring_dtype)


class Mesh(object):
    _memmap_dirs = []

    def __init__(self, points, springs):
        if not isinstance(points, numpy.ndarray):
            self.points = numpy.array(points)
        else:
            self.points = points
        if not isinstance(springs, numpy.ndarray):
            self.springs = numpy.array(springs, dtype=spring_dtype)
        elif springs.dtype != spring_dtype:
            self.springs = springs.astype(spring_dtype)
        else:
            self.springs = springs

    def memmap(self):
        if isinstance(self.points, numpy.memmap):
            return
        dn = tempfile.mkdtemp(prefix='springmesh')
        Mesh._memmap_dirs.append(dn)
        pfn = os.path.join(dn, 'mesh_points.npy')
        sfn = os.path.join(dn, 'mesh_springs.npy')
        # dump
        dpfn = joblib.dump(self.points, pfn)[0]
        dsfn = joblib.dump(self.springs, sfn)[0]
        # load
        # TODO free originals?
        self.points = joblib.load(dpfn, 'r+')
        self.springs = joblib.load(dsfn, 'r+')

    def to_shared(self):
        #sa = multiprocessing.sharedctypes.RawArray('d', self.points.size)
        sa = multiprocessing.Array('d', self.points.size)
        s = numpy.frombuffer(sa.get_obj(), 'f8', count=self.points.size)
        s.shape = self.points.shape
        s[:] = self.points[:]
        return sa, self.points.shape
        # a = numpy.ctypeslib.as_array(sa)
        # a.shape = self.points.shape


def remove_memmaps():
    for dn in Mesh._memmap_dirs:
        # remove file and directory
        #os.remove(fn)
        #os.removedirs(dn)
        shutil.rmtree(dn)
        #os.removedirs(os.path.dirname(fn))


atexit.register(remove_memmaps)
