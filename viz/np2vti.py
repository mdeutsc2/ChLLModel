import vtk
from vtk.util import numpy_support
import numpy as np
import sys,os
np.bool = np.bool_


def convert_npz(fn):
	outfile = ".".join(fn.split(".")[:-1]) + ".vti"
	data = np.load(fn)
	img = vtk.vtkImageData()
	shape = data['nx'].shape
	img.SetDimensions(shape[0],shape[1],shape[2])
	s_data = numpy_support.numpy_to_vtk(num_array = data['s'].flatten(),
					deep=True,
					array_type=vtk.VTK_INT)
	s_data.SetName("s")
	img.GetPointData().AddArray(s_data)

	n = np.stack((data['nx'],data['ny'],data['nz']))
	flat_vector_array = n.reshape(-1,3)
	vtk_vectors = numpy_support.numpy_to_vtk(num_array=flat_vector_array,
					    deep=True,
					    array_type=vtk.VTK_FLOAT)
	vtk_vectors.SetName("n")
	img.GetPointData().AddArray(vtk_vectors)

	writer = vtk.vtkXMLImageDataWriter()
	writer.SetFileName(outfile)
	writer.SetInputData(img)
	writer.Write()
	print(fn," => ",outfile)


infile = sys.argv[1]
if os.path.isdir(infile):
	for root,dirs,files in os.walk(infile):
		for file in files:
			if file.endswith('.npz'):
				file_path = os.path.join(root,file)
				convert_npz(file_path)
elif infile.endswith('.npz'):
	convert_npz(infile)
else:
	print("not npz file or directory!")
	exit()

print("Done!")
