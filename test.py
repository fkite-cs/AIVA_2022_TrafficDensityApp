import pdb
import src.detector_base as detector_base 
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS

from types import SimpleNamespace
import os
import numpy as np
import cv2
import math
import time
import numba
from numba import njit, jit, config, threading_layer
# config.THREADING_LAYER = 'threadsafe'

# pip install pyscenic tbb

""" pip3 installl
OSMPythonTools - https://wiki.openstreetmap.org/wiki/OSMPythonTools
osmx - https://automating-gis-processes.github.io/CSC18/lessons/L3/retrieve-osm-data.html
osmnx
"""

# ./run.sh /home/fkite/docker_compartido/ austin1.tif

def test_tiff_image_metada():
    import tifffile
    tif = tifffile.TiffFile("examples/austin1.tif")
    # print(dir(tif))
    print(tif.geotiff_metadata)


def test_osmnx():
    import osmnx as ox
    place_name = "Kamppi, Helsinki, Finland"
    graph = ox.graph_from_place(place_name)
    ox.grap
    fig, ax = ox.plot_graph(graph)
    plt.tight_layout()
    plt.show()
 
def get_metadata_pil():
    # Leer imagen
    imagen = Image.open("examples/austin1.tif")
    
    # Obtener metadatos
    datos_exif = imagen.getexif()
    
    # Diccionario en el cual se guardaran los datos
    exif_dic = {}
    
    # Ciclo para decodificar los datos
    for tag, value in datos_exif.items():
        # Obtener el nombre de la etiqueta etiqueta
        etiqueta = TAGS.get(tag, tag)
        # Obtener el valor de la etiqueta
        valor = datos_exif.get(tag)
        exif_dic[etiqueta] = valor
        # Imprimir los datos
        # print(str(etiqueta) + " : " +str(valor))
        print(str(etiqueta))


def renderize_image(fig_plot, new_shape=(600,600),normalize=True):
    fig_plot.canvas.draw()

    img = cv2.cvtColor(np.asarray(fig_plot.canvas.buffer_rgba()), cv2.COLOR_RGBA2BGR)
    img = cv2.resize(img, new_shape)

    if normalize:
        img = img / 255.0
    return img


@jit(nopython=True, parallel=True)
def kde_quartic(d, h):
    dn = d / h
    P = (15 / 16) * (1 - dn ** 2) ** 2
    return P

@jit(nopython=True, parallel=True)
def last_step(_xc, x, _yc, y, h):
    _d = np.sqrt((_xc - x)**2 + (_yc - y)**2)
    # _d = math.dist(_xc-x, _yc-y)
    # _d = np.linalg.norm(_xc-x, _yc-y)
    _kde_value = np.where(_d <= h, kde_quartic(_d, h), 0)
    return _kde_value

def create_heat_map(img, crop_list, out_folder="example/", block_size=1000):
    # block_size = 100
    # crop = crop_list[1]
    # img = crop.get_crop(img)
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
    # crop = crop_list[1]
        for vehicle in crop.vehicles_list:
            gy, gx = vehicle.get_global_coordinates()
            _y = int(gy + vehicle.dy/2)
            _x = int(gx + vehicle.dx/2)
            # print("local ", vehicle.y, vehicle.x)
            # print("gy, gx ", gy, gx, vehicle.global_hw)
            # _y = vehicle.y +  vehicle.dy/2
            # _x = vehicle.x +  vehicle.dx/2
            
            x.append(_x)
            y.append(_y)
        # break

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
        t0 = time.time()
        if i < len_idxs - 1:
            _xc = np.expand_dims(xc[ia:ib], axis=2)
            _yc = np.expand_dims(yc[ia:ib], axis=2)
        else:
            _xc = np.expand_dims(xc[ia:], axis=2)
            _yc = np.expand_dims(yc[ia:], axis=2)
        t1 = time.time()
        _xc = np.repeat(_xc, x.size, axis=2)
        _yc = np.repeat(_yc, y.size, axis=2)
        t2 = time.time()
        _kde_value = last_step(_xc, x, _yc, y, h)
        t3 = time.time()
        if i < len_idxs - 1:
            intensity[ia:ib] = np.add.reduce(_kde_value, axis=2)
        else:
            intensity[ia:] = np.add.reduce(_kde_value, axis=2)
        t4 = time.time()
        print("t1 - t0 ", t1 - t0)
        print("t2 - t1 ", t2 - t1)
        print("t3 - t2 ", t3 - t2)
        print("t4 - t3 ", t4 - t3)
        print("total ", t4 - t0)
    
    intensity = np.ma.masked_array(intensity, intensity < 0.01*intensity.max())
    
    fig, ax = plt.subplots(1, 1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cb = ax.pcolormesh(x_mesh, y_mesh, intensity, alpha=0.25, cmap="OrRd")
    fig.colorbar(cb)
    plt.axis("off")
    path = os.path.join(out_folder, "crop_2.tif")
    plt.savefig(path, dpi=1500)


    """
        t1 - t0  0.0007436275482177734
        t2 - t1  0.21412968635559082
        t3 - t2  0.24992895126342773
        t4 - t3  0.02015519142150879
        total  0.4849574565887451

    """

def create_heat_map2_optimized(img, out_folder="example/"):
    x = np.array([20, 50, 70, 90, 110, 130, 150, 170])
    y = np.array([20, 50, 70, 90, 110, 130, 150, 170])
    
    y_grid = np.arange(0, img.shape[0])
    x_grid = np.arange(0, img.shape[1])
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    
    grid_size = 1
    h = 30
    xc = x_mesh + (grid_size / 2)
    yc = y_mesh + (grid_size / 2)
    xc = xc.astype(np.float16)
    yc = yc.astype(np.float16)

    xc = np.expand_dims(xc, axis=2)
    xc = np.repeat(xc, len(x), axis=2)

    yc = np.expand_dims(yc, axis=2)
    yc = np.repeat(yc, len(y), axis=2)

    d = np.sqrt((xc - x)**2 + (yc - y)**2)
    kde_value = np.where(d <= h, kde_quartic(d, h), 0)
    intensity = kde_value.sum(axis=2)
    
    intensity = np.ma.masked_array(intensity, intensity < 0.01*intensity.max())
    fig, ax = plt.subplots(1, 1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cb = ax.pcolormesh(x_mesh, y_mesh, intensity, alpha=0.5, cmap="inferno")
    fig.colorbar(cb)
    plt.axis("off")
    path = os.path.join(out_folder, "heatmap_op.png")
    plt.savefig(path)

def create_heat_map3_optimized(img, out_folder="example/", block_size=10):
    x = np.array([20, 50, 70, 90, 110, 130, 150, 170])
    y = np.array([20, 50, 70, 90, 110, 130, 150, 170])
    
    y_grid = np.arange(0, img.shape[0])
    x_grid = np.arange(0, img.shape[1])
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    
    grid_size = 1
    h = 30
    xc = x_mesh + (grid_size / 2)
    yc = y_mesh + (grid_size / 2)
    xc = xc.astype(np.float16)
    yc = yc.astype(np.float16)

    hh, w, c = img.shape
    step = int(hh/block_size)
    idxs = np.arange(0, hh, step, dtype=np.int32)

    intensity = np.zeros(tuple(xc.shape), dtype=np.float16)
    len_idxs = len(idxs)
    for i in range(len(idxs)):
        ia, ib = idxs[i], step+idxs[i]
        if i < len_idxs - 1:
            _xc = np.expand_dims(xc[ia:ib], axis=2)
            _yc = np.expand_dims(yc[ia:ib], axis=2)
        else:
            _xc = np.expand_dims(xc[ia:], axis=2)
            _yc = np.expand_dims(yc[ia:], axis=2)
        _xc = np.repeat(_xc, len(x), axis=2)
        _yc = np.repeat(_yc, len(y), axis=2)

        _d = np.sqrt((_xc - x)**2 + (_yc - y)**2)
        _kde_value = np.where(_d <= h, kde_quartic(_d, h), 0)
        if i < len_idxs - 1:
            intensity[ia:ib] = _kde_value.sum(axis=2)
        else:
            intensity[ia:] = _kde_value.sum(axis=2)
    
    intensity = np.ma.masked_array(intensity, intensity < 0.01*intensity.max())
    fig, ax = plt.subplots(1, 1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cb = ax.pcolormesh(x_mesh, y_mesh, intensity, alpha=0.5, cmap="inferno")
    fig.colorbar(cb)
    plt.axis("off")
    path = os.path.join(out_folder, "heatmap_op.png")
    plt.savefig(path)

def create_heat_map4_optimized(img, out_folder="example/", block_size=10):
    hh, ww, _ = img.shape
    # x = np.arange(20,hh,80)
    # y = np.arange(20,hh,80)
    x = np.random.randint(0, ww, 10)
    y = np.random.randint(0, hh, 10)
    print("x ", x)
    print("y ", y)
    
    y_grid = np.arange(0, img.shape[0])
    x_grid = np.arange(0, img.shape[1])
    x_mesh, y_mesh = np.meshgrid(x_grid, y_grid)
    
    grid_size = 1
    h = 150
    xc = x_mesh + (grid_size / 2)
    yc = y_mesh + (grid_size / 2)
    xc = xc.astype(np.float32)
    yc = yc.astype(np.float32)

    hh, w, c = img.shape
    step = int(hh/block_size)
    idxs = np.arange(0, hh, step, dtype=np.int32)

    intensity = np.zeros(tuple(xc.shape), dtype=np.float32)
    len_idxs = len(idxs)
    for i in range(len(idxs)):
        # print(f"{i+1}/{len_idxs}")
        ia, ib = idxs[i], step+idxs[i]
        # t0 = time.time()
        if i < len_idxs - 1:
            _xc = np.expand_dims(xc[ia:ib], axis=2)
            _yc = np.expand_dims(yc[ia:ib], axis=2)
        else:
            _xc = np.expand_dims(xc[ia:], axis=2)
            _yc = np.expand_dims(yc[ia:], axis=2)
        # t1 = time.time()
        _xc = np.repeat(_xc, x.size, axis=2)
        _yc = np.repeat(_yc, y.size, axis=2)
        # t2 = time.time()
        _kde_value = last_step(_xc, x, _yc, y, h)
        # t3 = time.time()
        if i < len_idxs - 1:
            intensity[ia:ib] = np.add.reduce(_kde_value, axis=2)
        else:
            intensity[ia:] = np.add.reduce(_kde_value, axis=2)
        # t4 = time.time()
        # print("t1 - t0 ", t1 - t0)
        # print("t2 - t1 ", t2 - t1)
        # print("t3 - t2 ", t3 - t2)
        # print("t4 - t3 ", t4 - t3)
        # print("total ", t4 - t0)
    
    intensity_color = np.copy(intensity)
    print("intensity_color ", intensity_color.min(), intensity_color.max())
    intensity = np.ma.masked_array(intensity, intensity < 0.01*intensity.max())
    plt.figure()
    fig, ax = plt.subplots(1, 1)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cb = ax.pcolormesh(x_mesh, y_mesh, intensity, alpha=0.5, cmap="inferno")
    fig.colorbar(cb)
    plt.axis("off")
    # path = os.path.join(out_folder, "heatmap_op_2.png")
    # plt.savefig(path)
    # int_map = renderize_image(fig, tuple(intensity.shape), normalize=False)
    # intensity_color = cv2.normalize(img,  norm, 0, 255, cv.NORM_MINMAX
    intensity_color = (intensity_color - intensity_color.min()) / (intensity_color.max() - intensity_color.min())
    intensity_color = intensity_color*255
    intensity_color = intensity_color.astype(np.uint8)
    intensity_color = np.expand_dims(intensity_color, axis=2)
    h, w, _ = intensity_color.shape
    z = np.zeros((h,w,2))
    intensity_color = np.concatenate((intensity_color, z), axis=2)
    intensity_color = intensity_color.astype(np.uint8)
    # cv2.imshow("aa ", intensity_color)
    # cv2.waitKey(0)
    print("intensity_color ", intensity_color.min(), intensity_color.max())
    img = cv2.addWeighted(img, 0.6, intensity_color, 0.4, 0.0)
    cv2.imwrite("prueba2.tif", img)

if __name__ == "__main__":

    """model_type = "yolo"
    config = {
        "weights_path": "best.pt",
        "device": "cpu"
    }

    config = SimpleNamespace(**config)

    m = detector_base.DetectorBase(model_type, config)
    pdb.set_trace()"""
    #test_tiff_image_metada()
    # get_metadata_pil()
    # test_osmnx()

    # img = cv2.imread("gatito.jpeg")
    # print("img ", img.shape)
    # p0 = time.time()
    # create_heat_map2(img, out_folder="./")
    # p1 = time.time()
    # print("timne og ", p1-p0) # 4.67

    # img = cv2.imread("gatito.jpeg")
    # h, w, _ = tuple(img.shape)
    # img = cv2.resize(img, (h*3, w*3))
    # print("img ", img.shape)
    # p0 = time.time()
    # create_heat_map4_optimized(img, out_folder="./") # 5.83 -> 0.44 -> 0.38
    # p1 = time.time()
    # print("timne optimized ", p1-p0)

    img = cv2.imread("examples/austin1.tif")
    data = np.load("data.npy", allow_pickle=True)
    print("data ", len(data))
    p0 = time.time()
    create_heat_map(img, data, out_folder="./")
    p1 = time.time()
    print("timne optimized ", p1-p0)