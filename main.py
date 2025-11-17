## This file should load in data to a numpy array, instantiate our model object, and call curve fitting functions 

from model import *
import numpy as np

# Path to your data file
file_path = "data/W25000_L1000"
idvg_path = file_path + "_idvg.data"
idvd_path = file_path + "_idvd.data"

idvg_data = np.genfromtxt(idvg_path, delimiter=None, comments='#', skip_header=1)
idvd_data = np.genfromtxt(idvd_path, delimiter=None, comments='#', skip_header=1)

filename = file_path.split("/")[-1]
width = float(filename.split("_")[0][1:]) * 1e-9
length = float(filename.split("_")[1][1:]) * 1e-9

# print(width)

mosfet = EKV_Model(idvg_data, idvd_data, width, length)
mosfet.fit_all()
mosfet.plot(reference=True, model=True)
# mosfet.plot_kappa()

