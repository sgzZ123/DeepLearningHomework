import torch
import numpy as np
from skimage.color import lab2rgb
from skimage import img_as_ubyte
from skimage import img_as_float


def to_rgb(grayscale_input, ab_input):
    color_image = torch.cat((grayscale_input, ab_input), 1).squeeze(0).numpy()
    color_image = color_image.transpose((1,2,0))
    color_image[:,:,0:1] = color_image[:,:,0:1] * 100
    color_image[:,:,1:3] = color_image[:,:,1:3] *255 -128
    color_image =lab2rgb(color_image.astype(np.float64))
    return color_image

