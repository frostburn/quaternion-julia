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
    scale = 1
    # Desktop
    width, height = 192*scale, 108*scale
    # Instagram
    width, height = 108*scale, 108*scale

    anti_aliasing = 4
    max_iter = 19

    center = np.quaternion(-0.2, 0.2, 0, 0)
    zoom = -0.4
    theta = -0.6
    phi = 0.4
    x_scale = 0.8

    np.random.seed(7)
    a = np.quaternion(*np.random.randn(4)) * 0.15
    b = np.quaternion(*np.random.randn(4)) * 0.15
    c = np.quaternion(*np.random.randn(4)) * 0.45

    def color_map(area, red, green, blue):
        return exp((-20*array([red, 0.4*green, blue])-5*area)*2.1)

    image = julia(width, height, center, zoom, theta, phi, a, b, c, max_iter, color_map, anti_aliasing=anti_aliasing, x_scale=x_scale)

    imsave("/tmp/out.png", make_picture_frame(image))
