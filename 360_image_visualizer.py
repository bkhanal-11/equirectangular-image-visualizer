import open3d as o3d
import cv2 as cv
import numpy as np
from argparse import ArgumentParser

class SphericalVisualizer:
    def __init__(self):
        '''
        Creates a visualization for 360 (spherical) image
        from equirectangular image.
        '''
        self.load_view_point()

    def equi_to_spherical(self, image, Z = 3):
        '''
        Converts equirectagular image to 
        360 image which can be visualized.
        
        :type image: numpy array
        :param image: image to be converted into mesh
        '''
        height, width, n_channels = image.shape
        image = image[...,::-1]

        mesh_x, mesh_y, mesh_z = np.meshgrid(np.linspace(0, width-1, width), np.linspace(0, height-1, height), Z)

        theta = mesh_x / width * (2*np.pi)
        phi = mesh_y / height * (np.pi)
        r = 3.5

        mesh_x = r * np.sin(phi) * np.cos(-theta) 
        mesh_y = r * np.sin(phi) * np.sin(-theta) 
        mesh_z = r * np.cos(phi) 

        XYZ = np.stack((mesh_x, mesh_y, mesh_z), axis = -1)
        XYZ = np.squeeze(XYZ)
        XYZ = np.reshape(XYZ, (height * width, -1))
        colors = (np.reshape(image, (width * height, n_channels))) / 256
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(XYZ)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd

    def load_view_point(self):
        '''
        Creates a pinhole camera parameter such that
        the default view is directed toward front face.
        '''
        self.param = o3d.camera.PinholeCameraParameters()
        self.param.intrinsic.set_intrinsics(1920, 1016, 879.88181024498977,
                                            879.88181024498977,959.5,507.5)
        self.param.extrinsic = np.array([[0, 1, 0, 0],
                                         [0, 0, -1, 0],
                                         [-1, 0, 0, 0],
                                         [0, 0, 0, 1]])
        
    def visualization(self, image_path):
        '''
        Creates the visualiation.
        
        :type image_path: string
        :param image_path: Path to input image.
        '''
        self.image360 = cv.imread(image_path)
        self.sph = self.equi_to_spherical(self.image360)
        
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = np.asarray([0, 0, 0])
        ctr = vis.get_view_control()
        
        vis.add_geometry(self.sph)
        ctr.convert_from_pinhole_camera_parameters(self.param)
        
        vis.run()
        vis.destroy_window()

if __name__ == '__main__':
    parser = ArgumentParser('Spherical image visualizer.')
    parser.add_argument('--image', help='Path to 360 image to be visualized.')
    args = parser.parse_args()

    image_path = str(args.image)
    SphericalVisualizer().visualization(image_path)
