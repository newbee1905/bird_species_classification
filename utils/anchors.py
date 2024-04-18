"""This the ultility module for generating and handling anchors
"""

import numpy as np
from config import INPUT_SIZE

_default_anchors_setting = (
	dict(layer='p3', stride=32, size=48, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
	dict(layer='p4', stride=64, size=96, scale=[2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
	dict(layer='p5', stride=128, size=192, scale=[1, 2 ** (1. / 3.), 2 ** (2. / 3.)], aspect_ratio=[0.667, 1, 1.5]),
)


def generate_default_anchor_maps(anchors_setting=None, input_shape=INPUT_SIZE):

	"""
	generate default anchor

	:param 
		- anchors_setting: all informations of anchors
		- input_shape: shape of input images, e.g. (h, w)

	:return
		- centre_anchors: # anchors * 4 (oy, ox, h, w)
		- edge_anchors: # anchors * 4 (y0, x0, y1, x1)
		- anchor_area: # anchors * 1 (area)
	"""

	if anchors_setting is None:
		anchors_setting = _default_anchors_setting

	centre_anchors = np.zeros((0, 4), dtype=np.float32)
	edge_anchors = np.zeros((0, 4), dtype=np.float32)
	anchor_areas = np.zeros((0,), dtype=np.float32)
	input_shape = np.array(input_shape, dtype=int)

	for anchor_info in anchors_setting:

		stride = anchor_info['stride']
		size = anchor_info['size']
		scales = anchor_info['scale']
		aspect_ratios = anchor_info['aspect_ratio']

		output_map_shape = np.ceil(input_shape.astype(np.float32) / stride)
		output_map_shape = output_map_shape.astype(int)
		output_shape = tuple(output_map_shape) + (4,)

		ostart = stride / 2.
		oy = np.arange(ostart, ostart + stride * output_shape[0], stride)
		oy = oy.reshape(output_shape[0], 1)
		ox = np.arange(ostart, ostart + stride * output_shape[1], stride)
		ox = ox.reshape(1, output_shape[1])

		# Generating template for anchor map
		# Containing location of the centre point of the anchor
		#	while setting the width and height to 0
		centre_anchor_map_template = np.zeros(output_shape, dtype=np.float32)
		centre_anchor_map_template[:, :, 0] = oy
		centre_anchor_map_template[:, :, 1] = ox

		for scale in scales:
			for aspect_ratio in aspect_ratios:

				centre_anchor_map = centre_anchor_map_template.copy()

				width = size * scale / float(aspect_ratio) ** 0.5
				height = size * scale * float(aspect_ratio) ** 0.5

				centre_anchor_map[:, :, 2] = width
				centre_anchor_map[:, :, 3] = height

				centre_anchors_loc = centre_anchor_map[..., :2]
				centre_anchors_size = centre_anchor_map[..., 2:4]

				edge_anchor_map = np.concatenate(
					(centre_anchor_loc - centre_anchor_size / 2.),
					(centre_anchor_loc + centre_anchor_size / 2.),
					axis=-1
				)

				anchor_area_map = weight * height

				centre_anchors = np.concatenate((centre_anchors, centre_anchor_map.reshape(-1, 4)))
				edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
				anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))

	return centre_anchors, edge_anchors, anchor_areas
