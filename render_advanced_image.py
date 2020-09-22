from pylab import *
from julia import advanced_eval, second_order_eval, basic_eval, biaxial_eval, x_root_mandelbrot_eval, tesseract
from render_image import make_picture_frame
import numpy as np
import quaternion
from util import generate_mesh_slices, threaded_anti_alias, generate_imaginary_mesh_slices
from density import illuminate_and_absorb


if __name__ == '__main__':
    scale = 10
    u_samples = 1<<12
    theta = -0.75
    phi = 0.5
    gamma = 0.25
    beta = 0.0
    anti_aliasing = 2
    max_iter = 26
    zoom = -0.6
    width = 108*scale
    height = 108*scale
    depth = u_samples
    du = 1.0 / u_samples
    c = np.quaternion(0.45, -0.3, 0.2, 0.5)*1.15

    def source(q):
        val = tesseract(q, q, max_iter)
        core = (val <= 0)
        illumination = exp(-0.22*maximum(0, val-5))
        illumination = array([0.7*illumination, 0.4*illumination, illumination*0.8]) * 15
        core_glow = exp(-0.01*(val-1)**2)
        illumination += array([-0.05*core_glow, 0.9*core_glow, -core_glow*0.04]) * 35
        aura_glow = exp(-0.2*(val-20)**2)
        illumination += array([0.9*aura_glow, 0.1*aura_glow, aura_glow*0.5]) * 5
        illumination[:, core] = 0
        absorption = array([2.5*core, 1.5*core, core]) * 20
        aura_shroud = exp(-0.1*(val-12)**2)
        absorption += array([0.9*aura_shroud, 0.3*aura_shroud, aura_shroud*0.5]) * 100
        return 0.9*illumination, absorption

    def generate_subpixel_image(offset_x, offset_y):
        slices = generate_mesh_slices(width, height, depth, 0, 0, 0, 0, zoom, theta, phi, gamma, beta, offset_x, offset_y)
        image = illuminate_and_absorb(slices, source, array([0.35, 0.3, 0.4])*0.025, du)
        return image

    image = threaded_anti_alias(generate_subpixel_image, width, height, anti_aliasing)

    imsave("/tmp/out.png", make_picture_frame(image))
