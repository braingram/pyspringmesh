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
        __global const int2 *pi,
        __global const float *l,
        __global const float *k,
        __global const float2 *f) {

    __local float2 delta;
    __local float norm;
    __local float2 p0;
    __local float2 p1;

    unsigned int i = get_global_id(0);
    p0 = pts[pi[i].x];
    p1 = pts[pi[i].y];
    delta = (p1 - p0);
    n = length(delta);
    f[i] = delta / n * (n - l[i]) * k[i] * 0.01;
}
"""

# p0 + f
# p1 - f
op_prg = """
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
    # single point
    #pts = mesh.points.astype('f4')
    pts = numpy.empty(
        len(mesh.points),
        pyopencl.array.vec.float2)
    pts['x'] = mesh.points[:, 0]
    pts['y'] = mesh.points[:, 1]
    #pi = mesh.springs[['p0', 'p1']].view(('i8', 2)).astype('uint32')
    pi = numpy.empty(
        len(mesh.springs),
        pyopencl.array.vec.int2)
    pi['x'] = mesh.springs['p0']
    pi['y'] = mesh.springs['p1']
    #p0 = mesh.springs['p0', 'p1'].view('i8')
    #p1 = mesh.springs['p1'].view('i8')
    l = mesh.springs['l'].astype('f4')
    k = mesh.springs['k'].astype('f4')
    cl_pts = pyopencl.Buffer(c, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pts)
    #cl_pinds = pyopencl.Buffer(c, mf.READ_ONLY, hostbuf=pinds)
    #cl_p0 = pyopencl.Buffer(c, mf.READ_ONLY, hostbuf=p0)
    #cl_p1 = pyopencl.Buffer(c, mf.READ_ONLY, hostbuf=p1)
    cl_pi = pyopencl.Buffer(c, mf.READ_ONLY, hostbuf=pi)
    cl_l = pyopencl.Buffer(c, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=l)
    cl_k = pyopencl.Buffer(c, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=k)
    #cl_fx = pyopencl.Buffer(c, mf.READ_WRITE, l.nbytes)
    #cl_fy = pyopencl.Buffer(c, mf.READ_WRITE, l.nbytes)
    cl_f = pyopencl.Buffer(c, mf.READ_WRITE, pts.nbytes)
    #return cl_pts, cl_p0, cl_p1, cl_l, cl_k, cl_fx, cl_fy
    return cl_pts, cl_pi, cl_l, cl_k, cl_f


def run_n(mesh, n=1, s=0.01):
    forces, points = prepare_buffers(mesh)
    for i in xrange(int(n)):
        forces, err = compute_forces(mesh, s)
        offset_points(mesh, forces)
    return mesh
