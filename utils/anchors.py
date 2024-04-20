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
	:type
		- dict
		- Tuple[int, int]

	:return
		- centre_anchors: # anchors * 4 (oy, ox, h, w)
		- edge_anchors: # anchors * 4 (y0, x0, y1, x1)
		- anchor_area: # anchors * 1 (area)
	
	:rtype Tuple[np.ndarray, np.ndarray, np.ndarray]
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
					((centre_anchors_loc - centre_anchors_size / 2.),
					(centre_anchors_loc + centre_anchors_size / 2.)),
					axis=-1
				)

				anchor_area_map = centre_anchor_map[..., 2] * centre_anchor_map[..., 3]

				centre_anchors = np.concatenate((centre_anchors, centre_anchor_map.reshape(-1, 4)))
				edge_anchors = np.concatenate((edge_anchors, edge_anchor_map.reshape(-1, 4)))
				anchor_areas = np.concatenate((anchor_areas, anchor_area_map.reshape(-1)))

	return centre_anchors, edge_anchors, anchor_areas

def hard_nms(cdds, topn=10, iou_thresh=0.25):
	"""
	Hard Non-Maximum Suppression (NMS) algorithm to select the top scoring bounding boxes while suppressing
	highly overlapping boxes.

	:param
		- cdds: Detected bounding boxes with confidence scores, shape N * 5 (confidence, y0, x0, y1, x1)
		- topn: Maximum number of bounding boxes to retain
		- iou_thresh: Intersection over Union (IoU) threshold for overlap suppression
	
	:type
		- cdds: np.ndarray
		- topn: int
		- iou_thresh: float

	:return: Selected bounding boxes after hard NMS
	:rtype: np.ndarray
	"""

	if not (type(cdds).__module__ == 'numpy' and len(cdds.shape) == 2 and cdds.shape[1] >= 5):
		raise TypeError('edge_box_map should be N * 5+ ndarray')

	cdds = cdds.copy()
	indices = np.argsort(cdds[:, 0])
	cdds = cdds[indices]
	cdd_results = []

	res = cdds

	while res.any():
		cdd = res[-1]

		cdd_results.append(cdd)
		if len(cdd_results) == topn:
			return np.array(cdd_results)

		res = res[:-1]

		start_max = np.maximum(res[:, 1:3], cdd[1:3])
		end_min = np.minimum(res[:, 3:5], cdd[3:5])
		lengths = end_min - start_max

		intersec_map = lengths[:, 0] * lengths[:, 1]
		intersec_map[np.logical_or(lengths[:, 0] < 0, lengths[:, 1] < 0)] = 0
		iou_map_cur = intersec_map / ((res[:, 3] - res[:, 1]) * (res[:, 4] - res[:, 2]) + (cdd[3] - cdd[1]) * (cdd[4] - cdd[2]) - intersec_map)

		res = res[iou_map_cur < iou_thresh]

	return np.array(cdd_results)
