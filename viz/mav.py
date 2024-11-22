# Create the data.
import numpy as np
from mayavi.mlab import *
import argparse

def plot_all(data):
    shape = data['nx'].shape
    X,Y,Z = np.mgrid[0:shape[0],0:shape[1],0:shape[2]]
    
    u = data['nx']
    v = data['ny']
    w = data['nz']
    vecs = quiver3d(X,Y,Z, u, v, w, scalars=data['s'],line_width=3, scale_factor=1,mode='cylinder',colormap='blue-red')
    vecs.glyph.color_mode = 'color_by_scalar'
    scalarbar(vecs,orientation='vertical',nb_labels=3,title="Chirality")
    show()

def plot_slices(data):
    shape = data['nx'].shape
    X,Y,Z = np.mgrid[0:shape[0],0:shape[1],0:shape[2]]
    
    u = data['nx']
    v = data['ny']
    w = data['nz']
    #vecs = quiver3d(X,Y,Z, u, v, w, scalars=data['s'],line_width=3, scale_factor=1,mode='cylinder',colormap='blue-red')

    src = pipeline.vector_field(X,Y,Z,u, v, w,scalars=data['s'])
    pipeline.vector_cut_plane(src, mask_points=2, line_width=3,mode='cylinder',scale_factor=1,colormap='blue-red')
    scalarbar(src,orientation='vertical',nb_labels=3,title="Chirality")
    show()

def main(args):
    #data = load_data(args.filename)
    data  = np.load(args.filename)
    if args.all:
        plot_all(data)
    elif args.slices:
        plot_slices(data)
    else:
        print("No plotting type specified")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a file with NumPy and Mayavi.")
    parser.add_argument("filename", type=str, help="The input filename to process")
    parser.add_argument('-a', '--all',action='store_true',help="plot all points")
    parser.add_argument('-s', '--slices',action='store_true',help="plot slices")

    # Parse the arguments
    args = parser.parse_args()
    # Call the main function with the provided filename
    main(args)