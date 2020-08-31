import numpy as np
from numpy import sin, cos
from _routines import ffi, lib
from threading import Thread, Lock


def julia(
    width, height, center, zoom, theta, phi,
    a, b, c, max_iter, color_map,
    anti_aliasing=2, x_scale=1.0, u_scale=1.0, v_scale=0.005, u_samples=1024, v_samples=4,
    pseudo_mandelbrot=False, coloring=1, bg_luminance=0.2, attenuation=1.0):
    lock = Lock()

    num_color_channels = 3
    result = np.zeros((num_color_channels, height, width))

    zoom = 2**-zoom

    u_max = u_scale * zoom
    v_max = v_scale * zoom
    u_delta = -2*u_scale*zoom / u_samples
    v_delta = -2*v_scale*zoom / v_samples
    def accumulate_subpixels(offset_x, offset_y):
        nonlocal result

        x = np.arange(width, dtype='float64') + offset_x
        y = np.arange(height, dtype='float64') + offset_y
        x = x_scale * (2 * x - width) * zoom / height
        y = (2 * y - height) * zoom / height

        x, y = np.meshgrid(x, y)

        qw = center.w + x*cos(theta) + u_max*sin(theta)
        qx = center.x + u_max*cos(theta) - x*sin(theta)
        qy = center.y + y*cos(phi) + v_max*sin(phi)
        qz = center.z + v_max*cos(phi) - y*sin(phi)

        uw = sin(theta) * u_delta
        ux = cos(theta) * u_delta
        uy = 0
        uz = 0

        vw = 0
        vx = 0
        vy = sin(phi) * v_delta
        vz = cos(phi) * v_delta

        area = qw + 0
        red = qx + 0
        green = qy + 0
        blue = qz + 0
        qw_buf = ffi.cast("double*", area.ctypes.data)
        qx_buf = ffi.cast("double*", red.ctypes.data)
        qy_buf = ffi.cast("double*", green.ctypes.data)
        qz_buf = ffi.cast("double*", blue.ctypes.data)

        lib.julia(
            qw_buf, qx_buf, qy_buf, qz_buf, width*height,
            uw, ux, uy, uz, u_samples,
            vw, vx, vy, vz, v_samples,
            a.w, a.x, a.y, a.z,
            b.w, b.x, b.y, b.z,
            c.w, c.x, c.y, c.z,
            max_iter, pseudo_mandelbrot, coloring,
            bg_luminance, attenuation
        )

        subpixel_image = color_map(area, red, green, blue)

        lock.acquire()
        result += subpixel_image
        lock.release()

    ts = []
    offsets = np.arange(anti_aliasing) / anti_aliasing
    for i in offsets:
        for j in offsets:
            ts.append(Thread(target=accumulate_subpixels, args=(i, j)))
            ts[-1].start()
    for t in ts:
        t.join()

    result /= anti_aliasing**2

    return result
