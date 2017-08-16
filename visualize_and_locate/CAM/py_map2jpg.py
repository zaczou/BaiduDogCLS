import numpy as np
import matplotlib.pyplot as plt

def py_map2jpg(imgmap, rang, colorMap):
	if rang is None:
		rang = [np.min(imgmap), np.max(imgmap)]

	heatmap_x = np.round(imgmap*255).astype(np.uint8)

	cmap = plt.get_cmap('jet')
	rgba_img=cmap(imgmap)
	return np.delete(rgba_img,3,2)