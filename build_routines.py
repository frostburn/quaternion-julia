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
        int max_iterations
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
        int max_iterations
    ) {
        double dw = aw + bw;
        double dx = ax + bx;
        double dy = ay + by;
        double dz = az + bz;

        double ex = ax - bx;
        double ey = ay - by;
        double ez = az - bz;

        double metric = 1.0 / (u_samples * v_samples);  // TODO: Calculate actual area metric of (u, v).
        for (int i = 0; i < num_samples; ++i) {
            double area = 0.0;
            double red = 0.0;
            double green = 0.0;
            double blue = 0.0;
            for (int j = 0; j < u_samples; ++j) {
                for (int k = 0; k < v_samples; ++k) {
                    int iterations = eval(
                        in_w_out_area[i] + uw*j + vw*k,
                        in_x_out_red[i] + ux*j + vx*k,
                        in_y_out_green[i] + uy*j + vy*k,
                        in_z_out_blue[i] + uz*j + vz*k,
                        dw, dx, dy, dz,
                        ex, ey, ez,
                        cw, cx, cy, cz,
                        max_iterations
                    );
                    if (iterations == max_iterations) {
                        area += 1.0;
                        double u = j / (double) u_samples;
                        double v = k / (double) v_samples;
                        red += u;
                        blue += v;
                        u = 2*u - 1;
                        v = 2*v - 1;
                        double r = u*u + v*v;
                        if (r < 1) {
                            green += 1.0 - sqrt(r);
                        }
                    }
                }
            }
            in_w_out_area[i] = area * metric;
            in_x_out_red[i] = red * metric;
            in_y_out_green[i] = green * metric;
            in_z_out_blue[i] = blue * metric;
        }
    }
    """
)

if __name__ == '__main__':
    ffibuilder.compile(verbose=True)
