#!/usr/bin/env python

import numpy

import pyopencl
import pyopencl.array


context = None
queue = None
cf = None
op = None

mf = pyopencl.mem_flags

# f = (p1 - p0) / |p1 - p0| * (|p1 - p0| - l) * k * s
cf_prg = """
__kernel void compute_forces(
        __global const float2 *pts,
        __global const uint2 *pi,
        __global const float *l,
        __global const float *k,
        __global float2 *f) {

    float2 delta;
    float n;
    unsigned int i = get_global_id(0);

    delta = (pts[pi[i].y] - pts[pi[i].x]);
    n = length(delta);
    f[i] = delta / n * (n - l[i]) * k[i] * 0.01;
}
"""
# TODO replace 0.01 with s

# TODO need to get around no atomic_add float for opencl 
op_prg = """
__kernel void offset_points(
        __global volatile float2 *pts,
        __global const uint2 *pi,
        __global const float2 * f) {
    unsigned int i = get_global_id(0);

    //atomic_add(&(pts[pi[i].x]), f[i]);
    //atomic_sub(&(pts[pi[i].y]), f[i]);
    pts[pi[i].x] += f[i];
    pts[pi[i].y] -= f[i];
}
"""


def setup_context():
    global context
    global queue
    global cf
    global op
    if context is None:
        context = pyopencl.create_some_context()
        queue = pyopencl.CommandQueue(context)
        cf = pyopencl.Program(
            context, cf_prg).build()
        op = pyopencl.Program(
            context, op_prg).build()
    return context, queue, cf, op


def prepare_buffers(mesh):
    c, q, cf, op = setup_context()
    pts = numpy.empty(
        len(mesh.points),
        pyopencl.array.vec.float2)
    pts['x'] = mesh.points[:, 0].astype('f4')
    pts['y'] = mesh.points[:, 1].astype('f4')
    pi = numpy.empty(
        len(mesh.springs),
        pyopencl.array.vec.uint2)
    pi['x'] = mesh.springs['p0'].astype('uint32')
    pi['y'] = mesh.springs['p1'].astype('uint32')
    l = mesh.springs['l'].astype('f4')
    k = mesh.springs['k'].astype('f4')
    cl_pts = pyopencl.Buffer(c, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pts)
    cl_pi = pyopencl.Buffer(c, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pi)
    cl_l = pyopencl.Buffer(c, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=l)
    cl_k = pyopencl.Buffer(c, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k)
    cl_f = pyopencl.Buffer(c, mf.READ_WRITE, pts.nbytes)
    return cl_pts, cl_pi, cl_l, cl_k, cl_f


def run_n(mesh, n=1, s=0.01):
    assert s == 0.01
    c, q, cf, op = setup_context()
    pts, pi, l, k, f = prepare_buffers(mesh)
    shape = mesh.springs.shape
    for i in xrange(int(n)):
        cf.compute_forces(q, shape, None, pts, pi, l, k, f)
        op.offset_points(q, shape, None, pts, pi, f)
    p = numpy.empty(len(mesh.points), pyopencl.array.vec.float2)
    pyopencl.enqueue_copy(q, p, pts)
    mesh.points[:, 0] = p['x']
    mesh.points[:, 1] = p['y']
    return mesh
