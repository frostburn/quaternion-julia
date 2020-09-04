from pylab import *
from julia import advanced_eval, second_order_eval, basic_eval
from render_image import make_picture_frame
import numpy as np
import quaternion
from threading import Thread, Lock

if __name__ == '__main__':
    scale = 10
    u_samples = 2**12
    theta = 0.4
    anti_aliasing = 4
    max_iter = 22

    grid_x = linspace(-0.7, 1.5, scale*108)
    grid_y = linspace(-1.6, 1, scale*108)
    z = linspace(-1.5, 1.5, u_samples)
    dx = grid_x[1] - grid_x[0]
    dy = grid_y[1] - grid_y[0]
    dz = z[1] - z[0]

    grid_x, grid_y = meshgrid(grid_x, grid_y)

    np.random.seed(6)
    a = np.quaternion(*np.random.randn(4)) * 0.35
    b = np.quaternion(*np.random.randn(4)) * 0.25
    c = np.quaternion(*np.random.randn(4)) * 0.2

    result = array([0*grid_x, 0*grid_x, 0*grid_x])

    lock = Lock()

    def accumulate_subpixels(offset_x, offset_y):
        global result
        x = grid_x + offset_x
        y = grid_y + offset_y

        red = ones(x.shape) * 10.0
        green = ones(x.shape) * 9.2
        blue = ones(x.shape) * 9.0
        for w in z:
            q = x * np.quaternion(cos(theta), 0, sin(theta), 0) + y * np.quaternion(0, 1, 0, 0) + w * np.quaternion(-sin(theta), 0, cos(theta), 0)
            val = basic_eval(c + q*np.quaternion(0, 0, 0, 0.2)/q, a - 0.2*q*q, b + 0.2*q, q, 23)
            core = (val == 0)
            absorption = 3*exp(-0.25*val)
            absorption[core] = 5
            red += 50*core * dz
            red *= exp(-dz * absorption)

            absorption = 2.5*exp(-0.3*val)
            absorption[core] = 4
            green += 40*core * dz
            green *= exp(-dz * absorption)

            absorption = 2*exp(-0.35*val)
            absorption[core] = 2.5
            blue += 30*core * dz
            blue *= exp(-dz * absorption)

        lock.acquire()
        result += array([red, green, blue])*0.1
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
