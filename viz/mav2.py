import numpy as np
import argparse
from traits.api import HasTraits, Instance, Array, String, on_trait_change
from traitsui.api import View, Item, HGroup, Group

from tvtk.api import tvtk
from tvtk.pyface.scene import Scene

from mayavi import mlab
from mayavi.core.api import PipelineBase, Source
from mayavi.core.ui.api import SceneEditor, MayaviScene, \
                                MlabSceneModel

################################################################################
# The object implementing the dialog
class VolumeSlicer(HasTraits):
    # The data to plot
    nx = Array()
    ny = Array()
    nz = Array()
    s = Array()

    # The 4 views displayed
    scene3d_sliced = Instance(MlabSceneModel, ())
    scene3d_all = Instance(MlabSceneModel, ())
    scene_x = Instance(MlabSceneModel, ())
    scene_y = Instance(MlabSceneModel, ())
    scene_z = Instance(MlabSceneModel, ())

    # The data source
    data_src3d = Instance(Source)

    # The image plane widgets of the 3D scene
    ipw_3d_x = Instance(PipelineBase)
    ipw_3d_y = Instance(PipelineBase)
    ipw_3d_z = Instance(PipelineBase)

    _axis_names = dict(x=0, y=1, z=2)


    #---------------------------------------------------------------------------
    def __init__(self, **traits):
        super(VolumeSlicer, self).__init__(**traits)
        # Force the creation of the image_plane_widgets:
        self.ipw_3d_x
        self.ipw_3d_y
        self.ipw_3d_z


    #---------------------------------------------------------------------------
    # Default values
    #---------------------------------------------------------------------------
    def _data_src3d_default(self):
        return mlab.pipeline.scalar_field(self.s,
                            figure=self.scene3d_sliced.mayavi_scene)

    def make_ipw_3d(self, axis_name):
        ipw = mlab.pipeline.image_plane_widget(self.data_src3d,
                        figure=self.scene3d_sliced.mayavi_scene,
                        plane_orientation='%s_axes' % axis_name)
        return ipw

    def _ipw_3d_x_default(self):
        return self.make_ipw_3d('x')

    def _ipw_3d_y_default(self):
        return self.make_ipw_3d('y')

    def _ipw_3d_z_default(self):
        return self.make_ipw_3d('z')


    #---------------------------------------------------------------------------
    # Scene activation callbaks
    #---------------------------------------------------------------------------
    @on_trait_change('scene3d_sliced.activated')
    def display_scene3d_sliced(self):
        outline = mlab.pipeline.outline(self.data_src3d,
                        figure=self.scene3d_sliced.mayavi_scene,
                        )
        self.scene3d_sliced.mlab.view(40, 50)
        self.scene3d_sliced.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d_sliced.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()
        

    @on_trait_change('scene3d_all.activated')
    def display_scene3d_all(self):
        outline = mlab.quiver3d(self.nx,self.ny,self.nz,scalars=self.s,
                        figure=self.scene3d_all.mayavi_scene,
                        line_width=3, scale_factor=1,mode='cylinder',colormap='blue-red'
                        )
        outline.glyph.color_mode = 'color_by_scalar'
        self.scene3d_all.mlab.view(40, 50)
        self.scene3d_all.scene.background = (0, 0, 0)
        # Keep the view always pointing up
        self.scene3d_all.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleTerrain()


    def make_side_view(self, axis_name):
        scene = getattr(self, 'scene_%s' % axis_name)

        # To avoid copying the data, we take a reference to the
        # raw VTK dataset, and pass it on to mlab. Mlab will create
        # a Mayavi source from the VTK without copying it.
        # We have to specify the figure so that the data gets
        # added on the figure we are interested in.
        outline = mlab.pipeline.outline(
                            self.data_src3d.mlab_source.dataset,
                            figure=scene.mayavi_scene,
                            )
        ipw = mlab.pipeline.image_plane_widget(
                            outline,
                            plane_orientation='%s_axes' % axis_name)
        setattr(self, 'ipw_%s' % axis_name, ipw)

        # Synchronize positions between the corresponding image plane
        # widgets on different views.
        ipw.ipw.sync_trait('slice_position',
                            getattr(self, 'ipw_3d_%s'% axis_name).ipw)

        # Make left-clicking create a crosshair
        ipw.ipw.left_button_action = 0
        # Add a callback on the image plane widget interaction to
        # move the others
        def move_view(obj, evt):
            position = obj.GetCurrentCursorPosition()
            for other_axis, axis_number in self._axis_names.items():
                if other_axis == axis_name:
                    continue
                ipw3d = getattr(self, 'ipw_3d_%s' % other_axis)
                ipw3d.ipw.slice_position = position[axis_number]

        ipw.ipw.add_observer('InteractionEvent', move_view)
        ipw.ipw.add_observer('StartInteractionEvent', move_view)

        # Center the image plane widget
        ipw.ipw.slice_position = 0.5*self.data.shape[
                    self._axis_names[axis_name]]

        # Position the view for the scene
        views = dict(x=( 0, 90),
                     y=(90, 90),
                     z=( 0,  0),
                     )
        scene.mlab.view(*views[axis_name])
        # 2D interaction: only pan and zoom
        scene.scene.interactor.interactor_style = \
                                 tvtk.InteractorStyleImage()
        scene.scene.background = (0, 0, 0)


    @on_trait_change('scene_x.activated')
    def display_scene_x(self):
        return self.make_side_view('x')

    @on_trait_change('scene_y.activated')
    def display_scene_y(self):
        return self.make_side_view('y')

    @on_trait_change('scene_z.activated')
    def display_scene_z(self):
        return self.make_side_view('z')


    #---------------------------------------------------------------------------
    # The layout of the dialog created
    #---------------------------------------------------------------------------
    view = View(HGroup(
                  Group(
                        Item('scene3d_all',
                            editor=SceneEditor(scene_class=MayaviScene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                  Group(
                       Item('scene3d_sliced',
                            editor=SceneEditor(scene_class=MayaviScene),
                            height=250, width=300),
                       show_labels=False,
                  ),
                ),
                resizable=True,
                title="Default Window",
                )


# Create some data
def main(args):
    data  = np.load(args.filename)
    shape = data['nx'].shape
    X,Y,Z = np.mgrid[0:shape[0],0:shape[1],0:shape[2]]
    
    u = data['nx']
    v = data['ny']
    w = data['nz']
    # data = np.sin(3*x)/x + 0.05*z**2 + np.cos(3*y)
    # print(data.shape)

    m = VolumeSlicer(nx=data['nx'],
                     ny=data['nx'],
                     nz=data['nz'],
                     s=data['s'],dataname=args.filename)
    m.configure_traits()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a file with NumPy and Mayavi.")
    parser.add_argument("filename", type=str, help="The input filename to process")

    # Parse the arguments
    args = parser.parse_args()
    # Call the main function with the provided filename
    main(args)