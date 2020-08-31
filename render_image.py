from pylab import *
from julia import julia
import numpy as np
import quaternion

def make_picture_frame(rgb, dither=1.0/256.0):
    if dither:
        rgb = [channel + random(channel.shape)*dither for channel in rgb]
    frame = stack(rgb, axis=-1)
    frame = clip(frame, 0.0, 1.0)
    return frame


if __name__ == '__main__':
    scale = 10
    # Desktop
    width, height = 192*scale, 108*scale
    # Instagram
    width, height = 108*scale, 108*scale

    anti_aliasing = 4
    max_iter = 29

    center = np.quaternion(0.05, 0, 0.1, -0.1)
    zoom = -0.2
    theta = -0.1
    phi = -1.5
    x_scale = 1.0
    u_scale = 1.0
    u_samples = 4*1048
    v_samples = 1
    v_scale = 0.0

    np.random.seed(4)
    a = np.quaternion(*np.random.randn(4)) * 0.2
    b = np.quaternion(*np.random.randn(4)) * 0.4
    c = np.quaternion(*np.random.randn(4)) * 0.2

    def color_map(area, red, green, blue):
        return array([1.05*red, green, 0.95*blue])*2.1
        # return array([1.1*sqrt(red), 0.5*red + 1.1*sqrt(green), green + 3*blue**2])*1.5
        # return exp((-20*array([0.4*green, red, blue])-5*area)*0.7)

    image = julia(
        width, height, center, zoom, theta, phi, a, b, c, max_iter, color_map,
        anti_aliasing=anti_aliasing, x_scale=x_scale,
        u_scale=u_scale, u_samples=u_samples,
        v_scale=v_scale, v_samples=v_samples,
        pseudo_mandelbrot=False, coloring=1, bg_luminance=0.03, attenuation=19,
    )

    imsave("/tmp/out.png", make_picture_frame(image))
