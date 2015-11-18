#!/usr/bin/env python

import numpy

import pycuda
import pycuda.autoinit
import pycuda.compiler
import pycuda.driver
import pycuda.gpuarray


cf_source = """
#define BLOCK_SIZE %(block_size)d
#define N %(n)d
__global__ void compute_forces(
        float2 *pts,
        uint2 *pi,
        float *l,
        float *k,
        float2 *f) {

    float2 p0;
    float2 p1;
    float2 delta;
    float n;
    // const int i = threadIdx.x;
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < N) {
        p0 = pts[pi[i].x];
        p1 = pts[pi[i].y];
        delta = make_float2(p1.x - p0.x, p1.y - p0.y);
        n = hypotf(delta.x, delta.y);
        f[i] = make_float2(
            delta.x / n * (n - l[i]) * k[i] * %(s)f, 
            delta.y / n * (n - l[i]) * k[i] * %(s)f);
    }
}
"""

op_source = """
#define BLOCK_SIZE %(block_size)d
#define N %(n)d
__global__ void offset_points(
        float2 *pts,
        uint2 *pi,
        float2 *f) {

    // const int i = threadIdx.x;
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < N) {
        atomicAdd(&(pts[pi[i].x].x), f[i].x);
        atomicAdd(&(pts[pi[i].x].y), f[i].y);
        atomicAdd(&(pts[pi[i].y].x), -f[i].x);
        atomicAdd(&(pts[pi[i].y].y), -f[i].y);
    }
}
"""



def compile_functions(n, s, bs):
    assert bs[1] == 1 and bs[2] == 1
    fd = {
        "s": s,
        "block_size": bs[0],
        "n": n,
    }
    cf = pycuda.compiler.SourceModule(
        cf_source % fd).get_function(
            "compute_forces")
    #cf = pycuda.compiler.SourceModule(
    #    cf_source % fd, options=['-use_fast_math']).get_function(
    #        "compute_forces")
    op = pycuda.compiler.SourceModule(
        op_source % fd).get_function("offset_points")
    return cf, op


def prepare_buffers(mesh):
    # copy points up to gpu
    cpts = mesh.points.astype('f32')
    gpts = pycuda.gpuarray.to_gpu(cpts)

    cpi = mesh.springs[['p0', 'p1']].view(('i8', 2)).astype('uint32')
    gpi = pycuda.gpuarray.to_gpu(cpi)

    cl = mesh.springs['l'].astype('f32')
    gl = pycuda.gpuarray.to_gpu(cl)

    ck = mesh.springs['k'].astype('f32')
    gk = pycuda.gpuarray.to_gpu(ck)

    gf = pycuda.gpuarray.empty_like(gpts)
    return gpts, gpi, gl, gk, gf


def prepare(mesh, s, bs):
    s *= 0.5  # prescale
    n = len(mesh.springs)
    cf, op = compile_functions(n, s, bs)
    gpts, gpi, gl, gk, gf = prepare_buffers(mesh)
    mesh.gpts = gpts
    mesh.gpi = gpi
    mesh.gl = gl
    mesh.gk = gk
    mesh.gf = gf
    mesh.cf = cf
    mesh.op = op
    # TODO prepare invocations of functions
    mesh.block = bs
    mesh.grid = (int(numpy.ceil(n / float(bs[0]))), 1)
    return mesh


def step_n(gpu_mesh, n=1):
    for i in xrange(int(n)):
        gpu_mesh.cf(
            gpu_mesh.gpts, gpu_mesh.gpi, gpu_mesh.gl, gpu_mesh.gk, gpu_mesh.gf,
            block=gpu_mesh.block, grid=gpu_mesh.grid)
        gpu_mesh.op(
            gpu_mesh.gpts, gpu_mesh.gpi, gpu_mesh.gf,
            block=gpu_mesh.block, grid=gpu_mesh.grid)


def finalize(gpu_mesh):
    pts = gpu_mesh.gpts.get()
    gpu_mesh.points[:] = pts[:]
    return gpu_mesh


def run_n(mesh, n=1, s=0.01, bs=(128, 1, 1)):
    mesh = prepare(mesh, s=s, bs=bs)
    step_n(mesh, n=n)
    return finalize(mesh)
