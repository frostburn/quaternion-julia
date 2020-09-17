from pylab import *
from julia import advanced_eval, second_order_eval, basic_eval, biaxial_eval
from render_image import make_picture_frame
import numpy as np
import quaternion
from util import generate_mesh_slices, threaded_anti_alias
from density import illuminate_and_absorb

if __name__ == '__main__':
    scale = 10
    u_samples = 1<<9
    theta = 0.1
    phi = 0.1
    gamma = 0.2
    beta = -0.1
    anti_aliasing = 4
    max_iter = 15
    zoom = -0.5
    width = 108*scale
    height = 108*scale
    depth = u_samples
    du = 1.0 / u_samples

    np.random.seed(17)
    c0 = np.quaternion(*np.random.randn(4)) * 0.28
    c1 = np.quaternion(*np.random.randn(4)) * 0.25

    exponent0 = 3
    exponent1 = 5

    def source(q):
        val = biaxial_eval(q, c0, c1, exponent0, exponent1, max_iter)
        core = (val == 0)
        illumination = exp(-0.3*val)
        illumination = array([illumination, illumination*0.5, illumination*0.2]) * 20
        illumination[:, core] = 0
        absorption = array([core, 2*core, core]) * 80
        return illumination, absorption

    def generate_subpixel_image(offset_x, offset_y):
        slices = generate_mesh_slices(width, height, depth, 0, 0, 0, 0, zoom, theta, phi, gamma, beta, offset_x, offset_y)
        image = illuminate_and_absorb(slices, source, array([0.5, 0.35, 0.55]), du)
        return image

    image = threaded_anti_alias(generate_subpixel_image, width, height, anti_aliasing)

    imsave("/tmp/out.png", make_picture_frame(image))
