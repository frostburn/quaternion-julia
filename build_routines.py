from cffi import FFI

ffibuilder = FFI()

ffibuilder.cdef(
    """void julia(
        double *in_w_out_area, double *in_x_out_red, double *in_y_out_green, double *in_z_out_blue, int num_samples,
        double uw, double ux, double uy, double uz, int u_samples,
        double vw, double vx, double vy, double vz, int v_samples,
        double aw, double ax, double ay, double az,
        double bw, double bx, double by, double bz,
        double cw, double cx, double cy, double cz,
        int max_iterations, int pseudo_mandelbrot, int coloring,
        double coloring_param_a, double coloring_param_b
    );"""
)

ffibuilder.set_source(
    "_routines",
    """
    int eval(
        double qw, double qx, double qy, double qz,
        double dw, double dx, double dy, double dz,
        double ex, double ey, double ez,
        double cw, double cx, double cy, double cz,
        int max_iterations
    ) {
        int i;
        for (i = 0; i < max_iterations; ++i) {
            if (qw*qw + qx*qx + qy*qy + qz*qz > 16) {
                return i;
            }

            double sw = qw;
            double sx = qx;
            double sy = qy;
            double sz = qz;

            double fw = qw + dw;
            double fx = qx + dx;
            double fy = qy + dy;
            double fz = qz + dz;

            qw = sw*fw - sx*fx - sy*fy - sz*fz + cw;
            qx = sw*fx + sx*fw + sy*ez - sz*ey + cx;
            qy = sw*fy + sy*fw - sx*ez + sz*ex + cy;
            qz = sw*fz + sz*fw + sx*ey - sy*ex + cz;
        }
        return i;
    }

    void julia(
        double *in_w_out_area, double *in_x_out_red, double *in_y_out_green, double *in_z_out_blue, int num_samples,
        double uw, double ux, double uy, double uz, int u_samples,
        double vw, double vx, double vy, double vz, int v_samples,
        double aw, double ax, double ay, double az,
        double bw, double bx, double by, double bz,
        double cw, double cx, double cy, double cz,
        int max_iterations, int pseudo_mandelbrot, int coloring,
        double coloring_param_a, double coloring_param_b
    ) {
        double dw = aw + bw;
        double dx = ax + bx;
        double dy = ay + by;
        double dz = az + bz;

        double ex = ax - bx;
        double ey = ay - by;
        double ez = az - bz;

        double metric = 1.0 / (u_samples * v_samples);  // TODO: Calculate actual area metric of (u, v).
        double attenuator = exp(-coloring_param_b*metric);
        double weak_attenuator = exp(-0.3*coloring_param_b*metric);
        for (int i = 0; i < num_samples; ++i) {
            double area = 0.0;
            double red = 0.0;
            double green = 0.0;
            double blue = 0.0;
            if (coloring == 1) {
                red = coloring_param_a;
                green = coloring_param_a;
                blue = coloring_param_a;
            }
            for (int j = 0; j < u_samples; ++j) {
                for (int k = 0; k < v_samples; ++k) {
                    int iterations;
                    if (pseudo_mandelbrot) {
                        iterations = eval(
                            cw, cx, cy, cz,
                            dw, dx, dy, dz,
                            ex, ey, ez,
                            in_w_out_area[i] + uw*j + vw*k,
                            in_x_out_red[i] + ux*j + vx*k,
                            in_y_out_green[i] + uy*j + vy*k,
                            in_z_out_blue[i] + uz*j + vz*k,
                            max_iterations
                        );
                    } else {
                        iterations = eval(
                            in_w_out_area[i] + uw*j + vw*k,
                            in_x_out_red[i] + ux*j + vx*k,
                            in_y_out_green[i] + uy*j + vy*k,
                            in_z_out_blue[i] + uz*j + vz*k,
                            dw, dx, dy, dz,
                            ex, ey, ez,
                            cw, cx, cy, cz,
                            max_iterations
                        );
                    }
                    if (coloring == 0) {
                        if (iterations == max_iterations) {
                            area += metric;
                            double u = j / (double) u_samples;
                            double v = k / (double) v_samples;
                            red += u * metric;
                            blue += v * metric;
                            u = 2*u - 1;
                            v = 2*v - 1;
                            double r = u*u + v*v;
                            if (r < 1) {
                                green += (1.0 - sqrt(r))*metric;
                            }
                        }
                    } else if (coloring == 1) {
                        if (iterations == max_iterations) {
                            area += metric;
                            red += metric;
                            green += metric;
                            blue += metric;
                        } else if (iterations == max_iterations - 1) {
                            red *= weak_attenuator;
                            green *= weak_attenuator;
                            blue *= attenuator;
                        } else if (iterations == max_iterations - 2) {
                            red *= weak_attenuator;
                            green *= attenuator;
                            blue *= weak_attenuator;
                        } else if (iterations == max_iterations - 3) {
                            red *= attenuator;
                            green *= weak_attenuator;
                            blue *= weak_attenuator;
                        } else if (iterations == max_iterations -4) {
                            red *= weak_attenuator;
                            green *= weak_attenuator;
                            blue *= weak_attenuator;
                        }
                    }
                }
            }
            in_w_out_area[i] = area;
            in_x_out_red[i] = red;
            in_y_out_green[i] = green;
            in_z_out_blue[i] = blue;
        }
    }
    """
)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
