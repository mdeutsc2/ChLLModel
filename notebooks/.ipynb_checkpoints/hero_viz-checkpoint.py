import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from mpl_toolkits.mplot3d import axes3d 
import os
from tqdm.auto import tqdm
from numba import jit
import scipy

def plot_quadrants(ax, array, fixed_coord, cmap):
    """For a given 3d *array* plot a plane with *fixed_coord*, using four quadrants."""
    nx, ny, nz = array.shape
    index = {
        'x': (nx // 2, slice(None), slice(None)),
        'y': (slice(None), ny // 2, slice(None)),
        'z': (slice(None), slice(None), nz // 2),
    }[fixed_coord]
    plane_data = array[index]

    n0, n1 = plane_data.shape
    quadrants = [
        plane_data[:n0 // 2, :n1 // 2],
        plane_data[:n0 // 2, n1 // 2:],
        plane_data[n0 // 2:, :n1 // 2],
        plane_data[n0 // 2:, n1 // 2:]
    ]

    min_val = array.min()
    max_val = array.max()

    cmap = plt.get_cmap(cmap)

    for i, quadrant in enumerate(quadrants):
        facecolors = cmap((quadrant - min_val) / (max_val - min_val))
        if fixed_coord == 'x':
            Y, Z = np.mgrid[0:ny // 2, 0:nz // 2]
            X = nx // 2 * np.ones_like(Y)
            Y_offset = (i // 2) * ny // 2
            Z_offset = (i % 2) * nz // 2
            ax.plot_surface(X, Y + Y_offset, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)
        elif fixed_coord == 'y':
            X, Z = np.mgrid[0:nx // 2, 0:nz // 2]
            Y = ny // 2 * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Z_offset = (i % 2) * nz // 2
            ax.plot_surface(X + X_offset, Y, Z + Z_offset, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)
        elif fixed_coord == 'z':
            X, Y = np.mgrid[0:nx // 2, 0:ny // 2]
            Z = nz // 2 * np.ones_like(X)
            X_offset = (i // 2) * nx // 2
            Y_offset = (i % 2) * ny // 2
            ax.plot_surface(X + X_offset, Y + Y_offset, Z, rstride=1, cstride=1,
                            facecolors=facecolors, shade=False)


def figure_3D_array_slices(array, i,cmap=None):
    """Plot a 3d array using three intersecting centered planes."""
    fig = plt.figure(figsize=(6,6),dpi=125)
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect(array.shape)
    plot_quadrants(ax, array, 'x', cmap=cmap)
    plot_quadrants(ax, array, 'y', cmap=cmap)
    plot_quadrants(ax, array, 'z', cmap=cmap)
    ax.view_init(30,-60+10*i)
    return fig, ax


datafiles = ["../data/hero4801/hero4801_data.npz",
            "../data/hero4802/hero4802_data.npz",
            "../data/hero4803/hero4803_data.npz",
            "../data/hero4804/hero4804_data.npz",
            "../data/hero4805/hero4805_data.npz",
            "../data/hero4806/hero4806_data.npz",
            "../data/hero4807/hero4807_data.npz",
            "../data/hero4808/hero4808_data.npz",
            "../data/hero4809/hero4809_data.npz",
            "../data/hero48010/hero48010_data.npz",
            "../data/hero48011/hero48011_data.npz",
            "../data/hero48012/hero48012_data.npz",
            "../data/hero48013/hero48013_data.npz",
            "../data/hero48014/hero48014_data.npz",
            "../data/hero48015/hero48015_data.npz",
            "../data/hero48016/hero48016_data.npz",
            "../data/hero48017/hero48017_data.npz",
            "../data/hero48018/hero48018_data.npz"
            "../data/hero48019/hero48019_data.npz"
            "../data/hero48020/hero48020_data.npz"]
for i in np.arange(1,21):
    fname = datafiles[i-1]
    data = np.load(fname)
    figure_3D_array_slices(data['s'], i,cmap='bwr')
    plt.savefig("hero_viz_"+str(i)+".png")