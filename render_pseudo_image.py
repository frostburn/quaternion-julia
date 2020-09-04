from pylab import *
from _routines import ffi, lib
from render_image import make_picture_frame
from random import sample, seed
import numpy as np
from threading import Thread, Lock

julia_eval = np.frompyfunc(lib.julia_eval, 16, 1)
smooth_julia_eval = np.frompyfunc(lib.smooth_julia_eval, 16, 1)

if __name__ == '__main__':
    scale = 10
    # Desktop
    # width, height = 192*scale, 108*scale
    # Instagram
    width, height = 108*scale, 108*scale

    anti_aliasing = 2
    max_iter = 32

    s = 24
    seed(s)
    np.random.seed(s)

    idx = sample(list(range(15)), 3)

    x = linspace(-1.3, 0.55, width)
    y = linspace(-1.51, 1.51, height)
    z = linspace(-1, 1, 2**14)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    x, y = np.meshgrid(x, y)

    args = list(randn(16)*0.09)
    args[-1] = max_iter

    result = array([0*x, 0*x, 0*x])

    def accumulate_subpixels(offset_x, offset_y):
        global result
        args[idx[0]] = x + offset_x
        args[idx[1]] = y + offset_y

        red = 0*x + 0.09
        green = 0*x + 0.08
        blue = 0*x + 0.07
        for w in z:
            args[idx[2]] = w
            val = smooth_julia_eval(*args).astype('float')
            core = (val == 0)
            red += core * dz * 0.5
            green += core * dz * 0.4
            blue += core * dz * 0.4
            abs_red = 2*exp(-0.1*val)
            abs_green = 2*exp(-(0.1*(val-2))**2)
            abs_blue = 3*exp(-(0.1*(val-5))**2)
            abs_red[core] = 0
            abs_green[core] = 0
            abs_blue[core] = 0
            red *= exp(-dz*abs_red)
            green *= exp(-dz*abs_green)
            blue *= exp(-dz*abs_blue)

        result += array([red, green, blue])

    ts = []
    offsets_x = np.arange(anti_aliasing) / anti_aliasing * dx
    offsets_y = np.arange(anti_aliasing) / anti_aliasing * dy
    for i in offsets_x:
        for j in offsets_y:
            accumulate_subpixels(i, j)

    result /= anti_aliasing**2

    image = result
    imsave("/tmp/out.png", make_picture_frame(image))
