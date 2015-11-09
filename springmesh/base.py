#!/usr/bin/env python


import numpy


spring_dtype = [
    ('p0', 'i4'), ('p1', 'i4'), ('k', 'f8'), ('b', 'f8'), ('l', 'f8'),
]


def to_springs(arr):
    return numpy.array(arr, dtype=spring_dtype)


class Mesh(object):
    def __init__(self, points, springs):
        self.points = points
        self.springs = springs
