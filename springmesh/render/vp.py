#!/usr/bin/env python

import numpy
from vispy import app, scene

#from .. import relax


def plot(mesh, show_k=False, show_v=False, show_f=False):
    points = mesh.points
    springs = mesh.springs

    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'
    view.camera.aspect = 1
    edges = springs[['p0', 'p1']].view(('i8', 2))
    lines = scene.Line(
        pos=points, connect=edges,
        antialias=False,
        method='gl', color='green', parent=view.scene)
    markers = scene.Markers(
        pos=points, face_color='blue', symbol='o', parent=view.scene,
        size=0.5, scaling=True
        )

    view.camera.set_range()

    app.run()


def run(mesh, show_k=False, show_v=False, show_f=False, run=None):
    points = mesh.points
    springs = mesh.springs

    canvas = scene.SceneCanvas(keys='interactive', show=True)
    view = canvas.central_widget.add_view()
    view.camera = 'panzoom'
    view.camera.aspect = 1
    edges = springs[['p0', 'p1']].view(('i8', 2))
    lines = scene.Line(
        pos=points, connect=edges,
        antialias=False,
        method='gl', color='green', parent=view.scene)
    markers = scene.Markers(
        pos=points, face_color='blue', symbol='o', parent=view.scene,
        size=0.5, scaling=True
        )

    view.camera.set_range()

    def update(ev):
        run(mesh)
        if mesh.points.min() == numpy.nan or mesh.points.max() == numpy.nan:
            return False
        markers.set_data(pos=mesh.points, size=0.5, scaling=True)
        lines.set_data(pos=mesh.points)

    if run is not None:
        timer = app.Timer(interval=0, connect=update, start=True)
    app.run()
