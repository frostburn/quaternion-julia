import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from pylab import *
from julia import advanced_eval, second_order_eval, basic_eval
from render_image import make_picture_frame
import numpy as np
import quaternion
from util import generate_mesh_slices, threaded_anti_alias
from density import illuminate_and_absorb

if __name__ == '__main__':
    scale = 10
    u_samples = 1<<11
    theta = 0.1
    phi = 0.1
    gamma = 0.2
    beta = -0.1
    anti_aliasing = 4
    max_iter = 20
    zoom = -0.3
    width = 108*scale
    height = 108*scale
    depth = u_samples
    du = 1.0 / u_samples

    np.random.seed(16)
    a = np.quaternion(*np.random.randn(4)) * 0.3
    b = np.quaternion(*np.random.randn(4)) * 0.25
    c = np.quaternion(*np.random.randn(4)) * 0.3

    def source(q):
        val = basic_eval(q, a, b, c, max_iter)
        core = (val == 0)
        illumination = exp(-0.25*val)
        illumination = array([illumination, illumination*0.5, illumination*0.2]) * 90
        illumination[:, core] = 0
        absorption = array([core, 2*core, core]) * 110
        return illumination, absorption

    def generate_subpixel_image(offset_x, offset_y):
        slices = generate_mesh_slices(width, height, depth, -0.1, 0.25, 0, 0.13, zoom, theta, phi, gamma, beta, offset_x, offset_y)
        image = illuminate_and_absorb(slices, source, array([0.5, 0.35, 0.55]), du)
        return image

    image = threaded_anti_alias(generate_subpixel_image, width, height, anti_aliasing)

    imsave("/tmp/out.png", make_picture_frame(image))
