import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"
os.environ["ETS_TOOLKIT"] = "null"

import numpy as np
import random
import mayavi

from PIL import Image
from mayavi import mlab
mlab.options.offscreen = True
import time

import pdb

''' class names:
'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
'person', 'bicyclist', 'motorcyclist', 'road', 'parking', 'sidewalk',
'other-ground', 'building', 'fence', 'vegetation', 'trunk', 'terrain',
'pole', 'traffic-sign'
'''
colors = np.array(
	[
		[100, 150, 245, 255],
		[100, 230, 245, 255],
		[30, 60, 150, 255],
		[80, 30, 180, 255],
		[100, 80, 250, 255],
		[255, 30, 30, 255],
		[255, 40, 200, 255],
		[150, 30, 90, 255],
		[255, 0, 255, 255],
		[255, 150, 255, 255],
		[75, 0, 75, 255],
		[175, 0, 75, 255],
		[255, 200, 0, 255],
		[255, 120, 50, 255],
		[0, 175, 0, 255],
		[135, 60, 0, 255],
		[150, 240, 80, 255],
		[255, 240, 150, 255],
		[255, 0, 0, 255],
	]).astype(np.uint8)

def get_grid_coords(dims, resolution):
	"""
	:param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
	:return coords_grid: is the center coords of voxels in the grid
	"""

	g_xx = np.arange(0, dims[0] + 1)
	g_yy = np.arange(0, dims[1] + 1)
	sensor_pose = 10
	g_zz = np.arange(0, dims[2] + 1)

	# Obtaining the grid with coords...
	xx, yy, zz = np.meshgrid(g_xx[:-1], g_yy[:-1], g_zz[:-1])
	coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
	coords_grid = coords_grid.astype(np.float32)

	coords_grid = (coords_grid * resolution) + resolution / 2

	temp = np.copy(coords_grid)
	temp[:, 0] = coords_grid[:, 1]
	temp[:, 1] = coords_grid[:, 0]
	coords_grid = np.copy(temp)

	return coords_grid

def draw(
    voxels,
    vox_origin,
    voxel_size=0.2,
    d=7,  # 7m - determine the size of the mesh representing the camera
    save_name=None,
    save_root=None,
    ):
    
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        [voxels.shape[0], voxels.shape[1], voxels.shape[2]], voxel_size
    )

    # Attach the predicted class to every voxel
    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T

    # Remove empty and unknown voxels
    fov_voxels = grid_coords[
        (grid_coords[:, 3] > 0) & (grid_coords[:, 3] < 255)
    ]

    figure = mlab.figure(size=(2800, 2800), bgcolor=(1, 1, 1))

    # Draw occupied inside FOV voxels
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=1,
        vmax=19,
    )

    infov_colors = colors
    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    plt_plot_fov.module_manager.scalar_lut_manager.lut.table = infov_colors
    
    scene = figure.scene
    scene.camera.position = [-50.907238103376244, -51.31911151935225, 104.75510851395386]
    scene.camera.focal_point = [23.005321731256945, 23.263153155247394, 0.7241134057028675]
    scene.camera.view_angle = 19.199999999999996
    scene.camera.view_up = [0.5286546999662366, 0.465851763212298, 0.7095818084728509]
    scene.camera.clipping_range = [92.25158502285397, 220.40602072417923]
	
    scene.camera.compute_view_plane_normal()
    scene.render()

    #mlab.show()

    save_file = save_name + '.png'
    mlab.savefig(os.path.join(save_root, save_file))
    print('saving to {}'.format(os.path.join(save_root, save_file)))
    mlab.close(all=True)
    # return save_file


def main(file_path):
    pred = np.load(file_path)
    save_root = '/u/home/caoh/projects/MA_Jiachen/ESSC-RM/docs/img'
    occ_pred = pred.reshape(256, 256, 32)
    occ_pred = occ_pred.astype(np.uint16)
    occ_pred[occ_pred==255] = 0
    #print(np.unique(occ_pred))

    vox_origin = np.array([0, -25.6, -2])
    os.makedirs(save_root, exist_ok=True)
    
    save_name = os.path.join(save_root, 'demo')
    occformer_img = draw(occ_pred, vox_origin, voxel_size=0.2, d=7, save_name=save_name, save_root=save_root)

# python /u/home/caoh/projects/MA_Jiachen/ESSC-RM/Tools/vis.py

if __name__ == '__main__':
    file_path = '/u/home/caoh/datasets/SemanticKITTI/dataset/eval_output/CGFormer/Conv_Conv/08/000000.npy'
    main(file_path)