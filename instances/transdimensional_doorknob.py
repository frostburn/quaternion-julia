import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from pylab import *
from julia import advanced_eval, second_order_eval, basic_eval, biaxial_eval, x_root_mandelbrot_eval
from render_image import make_picture_frame
import numpy as np
import quaternion
from util import generate_mesh_slices, threaded_anti_alias, generate_imaginary_mesh_slices
from density import illuminate_and_absorb


def mandel(q, c, max_iter):
    r = np.quaternion(np.cos(np.pi*2/12.0), 0, np.sin(np.pi*2/12.0), 0)
    r /= abs(r)

    c = c + q*0
    d = 1
    escaped = -np.ones(q.shape)
    for i in range(max_iter):
        escaped[np.logical_and(escaped < 0, abs(q) >= 128)] = i
        s = escaped < 0
        if s.any():
            q[s] = q[s]*q[s]*q[s]*q[s] + c[s]
        c = r*c

    exponent = 2
    s = escaped > 0
    escaped[s] = np.log(np.log(abs(q[s]))) / np.log(exponent) - escaped[s] + max_iter - 1 - np.log(np.log(128)) / np.log(exponent)
    escaped[~s] = 0

    return escaped


if __name__ == '__main__':
    scale = 90
    u_samples = 1<<11
    theta = -0.75
    phi = 0.3
    gamma = 0.25
    beta = 0.0
    anti_aliasing = 4
    max_iter = 20
    zoom = -0.7

    width, height = 7*scale, 12*scale
    depth = u_samples
    du = 1.0 / u_samples
    c = np.quaternion(0.45, -0.3, 0.2, 0.5)

    def source(q):
        val = mandel(q, q, max_iter)
        core = (val <= 0)
        illumination = exp(-0.39*val)
        illumination = array([0.8*illumination, 0.5*illumination, illumination*0.99]) * 40
        illumination[:, core] = 0
        absorption = array([2.5*core, 1.5*core, core]) * 40
        return illumination, absorption

    def generate_subpixel_image(offset_x, offset_y):
        slices = generate_mesh_slices(width, height, depth, 0, 0, 0, 0, zoom, theta, phi, gamma, beta, offset_x, offset_y)
        image = illuminate_and_absorb(slices, source, array([0.35, 0.3, 0.4])*0.25, du)
        return image

    image = threaded_anti_alias(generate_subpixel_image, width, height, anti_aliasing)

    imsave("/tmp/transdimensional_doorknob.png", make_picture_frame(image))
