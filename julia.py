import numpy as np
from numpy import sin, cos, log
from _routines import ffi, lib
from threading import Thread, Lock


def basic_eval(q, a, b, c, max_iter, shape=None):
    if shape is None:
        for z in (q, a, b, c):
            if hasattr(z, 'shape'):
                if shape is None:
                    shape = z.shape
                if shape == ():
                    shape = z.shape
    # Upshape and make unique
    z = np.zeros(shape)
    q = q + z
    a = a + z
    b = b + z
    c = c + z

    result = z
    q_buf = ffi.cast("double*", q.ctypes.data)
    a_buf = ffi.cast("double*", a.ctypes.data)
    b_buf = ffi.cast("double*", b.ctypes.data)
    c_buf = ffi.cast("double*", c.ctypes.data)
    result_buf = ffi.cast("double*", result.ctypes.data)

    lib.smooth_julia(
        q_buf, a_buf, b_buf, c_buf,
        result_buf, max_iter,
        result.size
    )

    return result


def advanced_eval(q, a, b, c, d, e, f, g, max_iter, exponent=2, shape=None):
    if shape is None:
        for z in (q, a, b, c, d, e, f, g):
            if hasattr(z, 'shape'):
                if shape is None:
                    shape = z.shape
                if shape == ():
                    shape = z.shape
    bailout = 256
    base = 1 / log(exponent)
    offset = max_iter - 1 - log(log(bailout)) * base
    result = -np.ones(shape)
    # Upshape and make unique
    z = np.zeros(shape)
    q = q + z
    a = a + z
    b = b + z
    c = c + z
    d = d + z
    e = e + z
    f = f + z
    g = g + z
    for i in range(max_iter):
        r = abs(q)
        result[np.logical_and(result < 0, r > bailout)] = i
        s = (r <= bailout)
        z = q[s]
        q[s] = z**exponent + z*a[s] + b[s]*z + c[s] + z*d[s]*z + z*e[s]/z + f[s]*z*z + z*z*g[s]
    inside = (result < 0)
    result[~inside] = log(log(abs(q[~inside]))) - result[~inside] + offset
    result[inside] = 0
    return result


def second_order_eval(q0, q1, a, b, c, d, max_iter, exponent=2, shape=None):
    if shape is None:
        for z in (q0, q1, a, b, c, d):
            if hasattr(z, 'shape'):
                if shape is None:
                    shape = z.shape
                if shape == ():
                    shape = z.shape
    bailout = 256
    base = 1 / log(exponent)
    offset = max_iter - 1 - log(log(bailout)) * base
    result = -np.ones(shape)
    # Upshape and make unique
    z = np.zeros(shape)
    q0 = q0 + z
    q1 = q1 + z
    a = a + z
    b = b + z
    c = c + z
    d = d + z
    for i in range(max_iter):
        r = abs(q0)
        result[np.logical_and(result < 0, r > bailout)] = i
        s = (r <= bailout)
        temp = q0[s] + 0
        q0[s] = q0[s]**exponent + q0[s]*d[s]*q1[s] + q1[s]*a[s] + b[s]*q0[s] + c[s]
        q1[s] = temp
    inside = (result < 0)
    result[~inside] = log(log(abs(q0[~inside]))) - result[~inside] + offset
    result[inside] = 0
    return result


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
