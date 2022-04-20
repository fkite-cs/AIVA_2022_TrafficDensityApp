import os
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

class HeatMap():

    def __init__(self) -> None:
        self.res = 1

    def kde_quartic(self, d, h):
        dn = d / h
        P = (15 / 16) * (1 - dn ** 2) ** 2
        return P

    def global_heat_map(self, img, crop_list, out_folder):
        """
            list_img
            return:
                img (5000,5000)
        """
        
        x = []
        y = []
        
        x_grid = np.arange(0, img.shape[0])
        y_grid = np.arange(0, img.shape[0])
        x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
        
        grid_size = 1
        h = 30
        xc = x_mesh + (grid_size / 2)
        yc = y_mesh + (grid_size / 2)
        
        for crop in crop_list:
            for vehicle in crop.vehicles_list:
                gy, gx = vehicle.get_global_coordinates()
                y = int(gy + vehicle.dy/2)
                x = int(gx + vehicle.dx/2)
                
                x.append(x)
                y.append(y)
                
        intensity_list = []
        for j in range(len(xc)):
            intensity_row = []
            for k in range(len(xc[0])):
                kde_value_list = []
                for i in range(len(x)):
                    # CALCULATE DISTANCE
                    d = math.sqrt((xc[j][k] - x[i]) ** 2 + (yc[j][k] - y[i]) ** 2)
                    if d <= h:
                        p = self.kde_quartic(d, h)
                    else:
                        p = 0
                    kde_value_list.append(p)
                # SUM ALL INTENSITY VALUE
                p_total = sum(kde_value_list)
                intensity_row.append(p_total)
            intensity_list.append(intensity_row)
            
        intensity = np.array(intensity_list)
        intensity = np.ma.masked_array(intensity, intensity < 0.01*intensity.max())
        fig, ax = plt.subplots(1, 1)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        cb = ax.pcolormesh(x_mesh, y_mesh, intensity, alpha=0.5, cmap="inferno")
        fig.colorbar(cb)
        plt.axis("off")
        path = os.path.join(out_folder, "heatmap.png")
        plt.savefig(path)

    def create_mask(self, img):
        """
            return:
                mask: (250,250)
        """
        return len(img)

    def run(self, data):
        """
            data: CropManager
        """
        return 1