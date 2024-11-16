#include "fastkpm.h"
#include "iostream_util.h"
#include <cassert>

using fkpm::Vec;
typedef double flt;
typedef std::complex<flt> cx_flt;

template <typename T>
double exact_energy(arma::Mat<T> const& H, double kT, double mu) {
    return fkpm::electronic_grand_energy(arma::real(arma::eig_gen(H)), kT, mu);
}

int main(int argc, char **argv) {
    fkpm::RNG rng(0);
    std::normal_distribution<flt> normal;

    int w = 48;
    int h = 48;

    double t = 1.0;
    double mu = 0 * t;
    double V0 = 1.3 * t;
    double V1 = 0 * t;
    double E_imp = 5 * t;
    double kT = 0.001 * t;

    double Delta_init_0 = 1 * t;
    double Delta_init_1 = 0 * t;

    double blend = 0.5;

    int n_sites = w*h;
    int n_rows = 2*n_sites;

    int n_colors = 16*16;
    n_colors = std::min(n_colors, n_sites);

    int M = 1000;
    int Mq = 4*M;

    auto idx = [&](int b, int x, int y) {
        return ((y)*w + x)*2 + b;
    };

    auto build_matrix = [&]() {
        fkpm::SpMatElems<cx_flt> elems(n_rows, n_rows, 1);

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {

                cx_flt os = {-mu, 0.0};
                if (x == 0 && y == 0) {
                    os += E_imp;
                }

                // on-site term
                Vec<Vec<cx_flt>> vs = { {os, {Delta_init_0, 0.0}}, {{Delta_init_0, 0.0}, -os} };
                for (int bi = 0; bi < 2; bi++) {
                    for (int bj = 0; bj < 2; bj++) {
                        int i = idx(bi, x, y);
                        int j = idx(bj, x, y);
                        elems.add(i, j, &vs[bi][bj]);
                    }
                }

                // nearest neighbor
                vs = { {{-t, 0.0}, {Delta_init_1, 0.0}},
                       {{Delta_init_1, 0.0}, {+t, 0.0}} };
                static const Vec<int> x_off = { 1, 0, -1, 0 };
                static const Vec<int> y_off = { 0, 1, 0, -1 };
                for (int nn = 0; nn < 4; nn++) {
                    int xp = (x + x_off[nn] + w) % w;
                    int yp = (y + y_off[nn] + h) % h;
                    for (int bi = 0; bi < 2; bi++) {
                        for (int bj = 0; bj < 2; bj++) {
                            int i = idx(bi,  x,  y);
                            int j = idx(bj, xp, yp);
                            elems.add(i, j, &vs[bi][bj]);
                        }
                    }
                }
            }
        }

        return fkpm::SpMatBsr<cx_flt>(elems);
    };

    auto mix_delta = [&](fkpm::SpMatBsr<cx_flt> const& D, fkpm::SpMatBsr<cx_flt>& H) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {

                // on site
                int ui = idx(0, x, y);
                int vi = idx(1, x, y);

                cx_flt old_delta = *H(ui, vi);
                cx_flt new_delta = - V0 * *D(ui, vi);
                cx_flt delta = blend * old_delta + (1 - blend) * new_delta;
                *H(ui, vi) = delta;
                *H(vi, ui) = conj(delta);

                // nearest neighbor
                static const Vec<int> x_off = { 1, 0, -1, 0 };
                static const Vec<int> y_off = { 0, 1, 0, -1 };
                for (int nn = 0; nn < 4; nn++) {
                    int xp = (x + x_off[nn] + w) % w;
                    int yp = (y + y_off[nn] + h) % h;

                    int ui = idx(0, x, y);
                    // int uj = idx(0, xp, yp);
                    // int vi = idx(1, x, y);
                    int vj = idx(1, xp, yp);

                    cx_flt old_delta = *H(ui, vj);
                    cx_flt new_delta = - V1 * *D(ui, vj);
                    cx_flt delta = blend * old_delta + (1 - blend) * new_delta;
                    *H(ui, vj) = delta;
                    *H(vj, ui) = conj(delta);
                }
            }
        }
    };

    auto H = build_matrix();


//    auto gamma = fkpm::moment_transform(moments, Mq);
//    Vec<double> x, rho, irho;
//    fkpm::density_function(gamma, es, x, rho);
//    fkpm::integrated_density_function(gamma, es, x, irho);
//    for (int i = 0; i < x.size(); i++) {
//        cout << x[i] << " " << rho[i] << " " << irho[i] << "\n";
//    }

//    cout << arma::real(H.to_arma_dense()) << "\n";
//    cout << arma::sort(arma::real(arma::eig_gen(H.to_arma_dense()))) << "\n";
//    return 0;

    using std::placeholders::_1;
    auto g = std::bind(fkpm::fermi_energy, _1, kT, 0);
    auto f = std::bind(fkpm::fermi_density, _1, kT, 0);

//    flt extra = 0.1;
//    flt tolerance = 1e-2;
//    auto es = fkpm::energy_scale(H, extra, tolerance);
    fkpm::EnergyScale es = {-10, 10};

    auto g_c = expansion_coefficients(M, Mq, g, es);
    auto f_c = expansion_coefficients(M, Mq, f, es);
    auto engine = fkpm::mk_engine<cx_flt>();
    auto D = H;

    int c_len = int(sqrt(n_colors));
    assert(c_len*c_len == n_colors);
    assert(w % c_len == 0 && h % c_len == 0);
    Vec<int> groups(n_rows);
    for (int i = 0; i < n_sites; i++) {
        int x = i % w;
        int y = i / h;
        int cx = x % c_len;
        int cy = y % c_len;
        groups[2*i+0] = 2*(cy*c_len + cx) + 0;
        groups[2*i+1] = 2*(cy*c_len + cx) + 1;
    }

//    engine->set_R_identity(n_rows);
//    engine->set_R_uncorrelated(n_rows, n_colors*2, rng);
    engine->set_R_correlated(groups, rng);

    for (int iter = 0; iter < 30; iter++) {
        engine->set_H(H, es);
        auto moments = engine->moments(M);
        engine->autodiff_matrix(g_c, D);
        // engine->stoch_matrix(f_c, D);

        cx_flt delta_mom1 = {0.0, 0.0};
        cx_flt delta_mom2 = {0.0, 0.0};
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                auto d = *H(idx(0, x, y), idx(1, x, y));
                delta_mom1 += d;
                delta_mom2 += d * conj(d);
            }
        }
        cx_flt mean_delta   = delta_mom1 / double(w*h);
        cx_flt var_delta = delta_mom2 / double(w*h) - mean_delta * conj(mean_delta);
        cout << "Iter  = " << iter << "\n";
        cout << "Delta = " << mean_delta << " +- " << sqrt(var_delta) << "\n";

        mix_delta(D, H);
    }


    cout << "# x Delta\n";
    for (int x = 0; x < w; x++) {
        auto d = *H(idx(0, x, 0), idx(1, x, 0));
        cout << x << " " << d.real() << "\n";
    }

    return 0;
}
