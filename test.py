import torch
from ATT_FUNC import SpectralResidual,get_cv2_func
from PIL import Image
import numpy as np

import matplotlib.pyplot as plt

def show_img(data):
    data = data.astype(np.uint8)
    plt.figure()  #
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data)
    # plt.savefig(save_path, bbox_inches='tight')
    # pdf.savefig()
    plt.show()
    plt.close()


cv2_func    = get_cv2_func(0)
I = Image.open('./pic/p2.png')
I_array  = np.array(I).astype(np.uint8)

this_mask   = SpectralResidual(cv2_func, I_array, 0.5)*255
show_img(I_array)
show_img(this_mask)
print(' ')


