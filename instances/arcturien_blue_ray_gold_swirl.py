import os
import sys
sys.path.insert(0, os.path.abspath('.'))

from pylab import *
from julia import advanced_eval, second_order_eval, basic_eval
from render_image import make_picture_frame
import numpy as np
import quaternion
from threading import Thread, Lock

if __name__ == '__main__':
    scale = 10
    u_samples = 1<<14
    theta = 0.1
    phi = 0.1
    anti_aliasing = 2
    max_iter = 22

    grid_x = linspace(-1.0, 1.0, scale*108)
    grid_y = linspace(-1.2, 1.2, scale*63)
    z = linspace(-1.5, 1.5, u_samples)
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    dz = z[1] - z[0]

    grid_x, grid_y = meshgrid(grid_x, grid_y)

    np.random.seed(7)
    a = np.quaternion(*np.random.randn(4)) * 0.35
    b = np.quaternion(*np.random.randn(4)) * 0.33
    c = np.quaternion(*np.random.randn(4)) * 0.5

    result = array([0*grid_x, 0*grid_x, 0*grid_x])

    lock = Lock()

    def accumulate_subpixels(offset_x, offset_y):
        global result
        x = grid_x + offset_x
        y = grid_y + offset_y

        red = ones(x.shape) * 0.0
        green = ones(x.shape) * 0.0
        blue = ones(x.shape) * 0.0
        for w in z:
            q = (
                x * np.quaternion(cos(theta), 0, sin(theta), 0) +
                y * np.quaternion(0, cos(phi), 0, sin(phi)) +
                w * np.quaternion(-sin(theta), 0, cos(theta), 0) +
                np.quaternion(0, -sin(phi), 0, cos(phi))
            )
            center = y**2 + (cos(theta)*w - sin(theta)*x)**2
            center = 10*exp(-4.3*center)

            val = basic_eval(q, a, b, c, max_iter)
            core = (val == 0)
            lumination = 100*exp(-0.04*val**2)
            lumination[core] = 0
            absorption = 0.02*exp(-0.1*val) + center
            absorption[core] = 10
            red += lumination * dz
            red *= exp(-dz * absorption)

            lumination = 100*exp(-0.045*val**2)
            lumination[core] = 0
            absorption = 0.03*exp(-0.1*val) + center
            absorption[core] = 10
            green += lumination * dz
            green *= exp(-dz * absorption)

            lumination = 40*exp(-val)
            lumination[core] = 0
            absorption = 0.02*exp(-0.2*val) + center
            absorption[core] = 10
            blue += lumination * dz
            blue *= exp(-dz * absorption)

        lock.acquire()
        result += array([red, green, blue])*1
        lock.release()


    ts = []
    offsets_x = np.arange(anti_aliasing) / anti_aliasing * dx
    offsets_y = np.arange(anti_aliasing) / anti_aliasing * dy
    for i in offsets_x:
        for j in offsets_y:
            ts.append(Thread(target=accumulate_subpixels, args=(i, j)))
            ts[-1].start()
    for t in ts:
        t.join()

    result /= anti_aliasing**2

    image = result
    imsave("/tmp/out.png", make_picture_frame(image))
