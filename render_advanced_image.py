from pylab import *
from julia import advanced_eval, second_order_eval, basic_eval, biaxial_eval, x_root_mandelbrot_eval
from render_image import make_picture_frame
import numpy as np
import quaternion
from util import generate_mesh_slices, threaded_anti_alias
from density import illuminate_and_absorb

if __name__ == '__main__':
    scale = 10
    u_samples = 1<<11
    theta = -0.5
    phi = 0.2
    gamma = 0.2
    beta = 0.0
    anti_aliasing = 2
    max_iter = 22
    zoom = -0.4
    width = 108*scale
    height = 108*scale
    depth = u_samples
    du = 1.0 / u_samples

    def source(q):
        val = x_root_mandelbrot_eval(q, q, max_iter)
        core = (val == 0)
        illumination = exp(-0.018*val*val)
        illumination = array([0.8*illumination, illumination, illumination*0.4]) * 40
        illumination[:, core] = 0
        absorption = array([2.5*core, 1.5*core, core]) * 40
        return illumination, absorption

    def generate_subpixel_image(offset_x, offset_y):
        slices = generate_mesh_slices(width, height, depth, 0, 0, 0, 0, zoom, theta, phi, gamma, beta, offset_x, offset_y)
        image = illuminate_and_absorb(slices, source, array([0.35, 0.3, 0.4])*0.25, du)
        return image

    image = threaded_anti_alias(generate_subpixel_image, width, height, anti_aliasing)

    imsave("/tmp/out.png", make_picture_frame(image))
