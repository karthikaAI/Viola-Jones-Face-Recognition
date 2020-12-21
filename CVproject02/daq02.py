import os
import cv2
import numpy as np
import glob
from PIL import Image

os.getcwd()
os.chdir('D:\CVproject02')

def in_array():
    Fimg = glob.glob("D:\\CVproject02\\Face16\\*.bmp")
    NFimg = glob.glob("D:\\CVproject02\\Nonface16\\*.bmp")
    p=len(Fimg[600:700])
    q=len(NFimg[700:800])
    data_x=Fimg[600:700]+ NFimg[700:800]    
    x = np.array([np.array(Image.open(fname)) for fname in data_x])
    return x,p,q

