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
        const float *pxs,
        const float *pys,
        const uint *p0s,
        const uint *p1s,
        const float *ls,
        const float *ks,
        float *fxs,
        float *fys,
        float s) {

    //float2 p0;
    //float2 p1;
    //float2 delta;
    // const int i = threadIdx.x;

    float dx;
    float dy;
    float n;
    const uint i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    //const uint p0 = p0s[i];
    //const uint p1 = p1s[i];

    if (i < N) {
        dx = pxs[p1s[i]] - pxs[p0s[i]];
        dy = pys[p1s[i]] - pys[p0s[i]];
        n = hypotf(dx, dy);
        fxs[i] = dx / n * (n - ls[i]) * ks[i] * s;
        fys[i] = dy / n * (n - ls[i]) * ks[i] * s;

        //p0 = pts[pi[i].x];
        //p1 = pts[pi[i].y];
        //delta = make_float2(p1.x - p0.x, p1.y - p0.y);
        //n = hypotf(delta.x, delta.y);
        //f[i] = make_float2(
        //    delta.x / n * (n - l[i]) * k[i] * s, 
        //    delta.y / n * (n - l[i]) * k[i] * s);
        //fxs[i] = pxs[p1] - pxs[p0];
        //fys[i] = pys[p1] - pys[p0];
    }
}
"""

op_source = """
#define BLOCK_SIZE %(block_size)d
#define N %(n)d
__global__ void offset_points(
        float *pxs,
        float *pys,
        uint *p0s,
        uint *p1s,
        float *fxs,
        float *fys) {

    // const int i = threadIdx.x;
    const int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    if (i < N) {
        atomicAdd(&(pxs[p0s[i]]), fxs[i]);
        atomicAdd(&(pys[p0s[i]]), fys[i]);
        atomicAdd(&(pxs[p1s[i]]), -fxs[i]);
        atomicAdd(&(pys[p1s[i]]), -fys[i]);
        //atomicAdd(&(pts[pi[i].x].x), f[i].x);
        //atomicAdd(&(pts[pi[i].x].y), f[i].y);
        //atomicAdd(&(pts[pi[i].y].x), -f[i].x);
        //atomicAdd(&(pts[pi[i].y].y), -f[i].y);
    }
}
"""



def compile_functions(n, bs):
    assert bs[1] == 1 and bs[2] == 1
    fd = {
        # "s": s,
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
    pxs = mesh.points[:, 0].astype('f32')
    pys = mesh.points[:, 1].astype('f32')
    gpxs = pycuda.gpuarray.to_gpu(pxs)
    gpys = pycuda.gpuarray.to_gpu(pys)

    #cpi = mesh.springs[['p0', 'p1']].view(('i8', 2)).astype('uint32')
    p0s = mesh.springs['p0'].astype('uint32')
    p1s = mesh.springs['p1'].astype('uint32')
    gp0s = pycuda.gpuarray.to_gpu(p0s)
    gp1s = pycuda.gpuarray.to_gpu(p1s)

    cl = mesh.springs['l'].astype('f32')
    gls = pycuda.gpuarray.to_gpu(cl)

    ck = mesh.springs['k'].astype('f32')
    gks = pycuda.gpuarray.to_gpu(ck)

    #gf = pycuda.gpuarray.empty_like(gpts)
    gfxs = pycuda.gpuarray.empty_like(gls)
    gfys = pycuda.gpuarray.empty_like(gls)
    #return gpts, gpi, gl, gk, gf
    return gpxs, gpys, gp0s, gp1s, gls, gks, gfxs, gfys


def prepare(mesh, bs):
    n = len(mesh.springs)
    cf, op = compile_functions(n, bs)
    #gpts, gpi, gl, gk, gf = prepare_buffers(mesh)
    gpxs, gpys, gp0s, gp1s, gls, gks, gfxs, gfys = prepare_buffers(mesh)
    #mesh.gpts = gpts
    #mesh.gpi = gpi
    #mesh.gl = gl
    #mesh.gk = gk
    #mesh.gf = gf

    mesh.cf = cf
    mesh.op = op

    mesh.gpxs = gpxs
    mesh.gpys = gpys
    mesh.gp0s = gp0s
    mesh.gp1s = gp1s
    mesh.gls = gls
    mesh.gks = gks
    mesh.gfxs = gfxs
    mesh.gfys = gfys
    # TODO prepare invocations of functions
    mesh.block = bs
    mesh.grid = (int(numpy.ceil(n / float(bs[0]))), 1)
    return mesh


def step_n(gpu_mesh, n=1, s=0.01):
    s = numpy.float32(s * 0.5)
    for i in xrange(int(n)):
        gpu_mesh.cf(
            gpu_mesh.gpxs, gpu_mesh.gpys, gpu_mesh.gp0s, gpu_mesh.gp1s,
            gpu_mesh.gls, gpu_mesh.gks, gpu_mesh.gfxs, gpu_mesh.gfys,
            s, block=gpu_mesh.block, grid=gpu_mesh.grid)
        gpu_mesh.op(
            gpu_mesh.gpxs, gpu_mesh.gpys, gpu_mesh.gp0s, gpu_mesh.gp1s,
            gpu_mesh.gfys, block=gpu_mesh.block, grid=gpu_mesh.grid)
        #gpu_mesh.cf(
        #    gpu_mesh.gpts, gpu_mesh.gpi, gpu_mesh.gl, gpu_mesh.gk, gpu_mesh.gf,
        #    block=gpu_mesh.block, grid=gpu_mesh.grid)
        #gpu_mesh.op(
        #    gpu_mesh.gpts, gpu_mesh.gpi, gpu_mesh.gf,
        #    block=gpu_mesh.block, grid=gpu_mesh.grid)


def finalize(gpu_mesh):
    #pts = gpu_mesh.gpts.get()
    #gpu_mesh.points[:] = pts[:]
    xs = gpu_mesh.gpxs.get()
    ys = gpu_mesh.gpys.get()
    gpu_mesh.points[:, 0] = xs
    gpu_mesh.points[:, 1] = ys
    return gpu_mesh


def run_n(mesh, n=1, s=0.01, bs=(128, 1, 1)):
    if not hasattr(mesh, gpxs):
        mesh = prepare(mesh, bs=bs)
    step_n(mesh, n=n, s=s)
    return finalize(mesh)
