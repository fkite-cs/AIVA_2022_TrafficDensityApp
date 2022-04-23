import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from numba import jit

import pdb

@jit(nopython=True, parallel=True)
def kde_quartic(d, h):
    dn = d / h
    P = (15 / 16) * (1 - dn ** 2) ** 2
    return P

@jit(nopython=True, parallel=True)
def calculate_kde(_xc, x, _yc, y, h):
    _d = np.sqrt((_xc - x)**2 + (_yc - y)**2)
    _kde_value = np.where(_d <= h, kde_quartic(_d, h), 0)
    return _kde_value

class HeatMap():

    def __init__(self) -> None:
        self.res = 1

    def global_heat_map(self, img, crop_list, out_folder, block_size=1000):
        """
            list_img
            return:
                img (5000,5000)
        """
        x = []
        y = []
        hh, ww, _ = img.shape
        y_grid = np.arange(0, img.shape[0])
        x_grid = np.arange(0, img.shape[1])
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        
        grid_size = 1
        h = 70
        xc = x_mesh + (grid_size / 2)
        yc = y_mesh + (grid_size / 2)
        xc = xc.astype(np.float32)
        yc = yc.astype(np.float32)
        for crop in crop_list:
            for vehicle in crop.vehicles_list:
                gy, gx = vehicle.get_global_coordinates()
                _y = int(gy + vehicle.dy/2)
                _x = int(gx + vehicle.dx/2)
                x.append(_x)
                y.append(_y)

        x = np.array(x)
        y = np.array(y)
        hh, w, c = img.shape
        step = int(hh/block_size)
        idxs = np.arange(0, hh, step, dtype=np.int32)

        intensity = np.zeros(tuple(xc.shape), dtype=np.float32)
        len_idxs = len(idxs)
        for i in range(len(idxs)):
            print(f"{i+1}/{len_idxs}")
            ia, ib = idxs[i], step+idxs[i]
            if i < len_idxs - 1:
                _xc = np.expand_dims(xc[ia:ib], axis=2)
                _yc = np.expand_dims(yc[ia:ib], axis=2)
            else:
                _xc = np.expand_dims(xc[ia:], axis=2)
                _yc = np.expand_dims(yc[ia:], axis=2)
            _xc = np.repeat(_xc, x.size, axis=2)
            _yc = np.repeat(_yc, y.size, axis=2)
            _kde_value = calculate_kde(_xc, x, _yc, y, h)
            if i < len_idxs - 1:
                intensity[ia:ib] = np.add.reduce(_kde_value, axis=2)
            else:
                intensity[ia:] = np.add.reduce(_kde_value, axis=2)
        
        # intensity = np.ma.masked_array(intensity, intensity < 0.01*intensity.max())
        fig, ax = plt.subplots(1, 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cb = ax.pcolormesh(x_mesh, y_mesh, intensity, alpha=0.25, cmap="OrRd")
        fig.colorbar(cb)
        plt.axis("off")
        path = os.path.join(out_folder, "heatmap.tif")
        plt.savefig(path, dpi=1500)
        plt.close("all")
        return 1
