#include <iomanip>
#include <cassert>
#include "iostream_util.h"
#include "fastkpm.h"
#include "kondo.h"


using fkpm::Vec;
using fkpm::cx_float;
using fkpm::cx_double;


inline int pos_cubic(int x, int y, int z, int lx) { return x + lx * (y + lx * z); }
inline int pos_square(int x, int y, int lx) { int temp = x + lx * y; assert(temp>=0 && temp<lx*lx); return temp; }



void testConductivity1() {
    std::cout << std::endl << "test1: Hall conductivity on square lattice." << std::endl;
    auto engine = fkpm::mk_engine<cx_double>();
    int lx = 100;
    int n  = lx * lx;
    int s  = 4;
    int M  = 40;
    int Mq = 2*M;

    double hopping     = 1.0;
    double B_over_phi0 = 0.05;
    double kT          = 0.001 * hopping;
    auto kernel        = fkpm::jackson_kernel(M);
    fkpm::SpMatElems<cx_double> H_elems(n, n, 1);
    fkpm::SpMatElems<cx_double> j1_elems(n, n, 1); // longitudinal (x direction)
    fkpm::SpMatElems<cx_double> j2_elems(n, n, 1); // transverse   (y direction)
    for (int xi = 0; xi < lx; xi++) {
        for (int yi = 0; yi < lx; yi++) {
            cx_double hop_xpos = -hopping * cx_double(cos(2.0*fkpm::Pi*B_over_phi0*yi),
                                                      -sin(2.0*fkpm::Pi*B_over_phi0*yi));
            cx_double hop_xneg = -hopping * cx_double(cos(2.0*fkpm::Pi*B_over_phi0*yi),
                                                      sin(2.0*fkpm::Pi*B_over_phi0*yi));
            cx_double hop_ypos = {-hopping, 0.0};
            cx_double hop_yneg = {-hopping, 0.0};

            cx_double crt_xpos =  hop_xpos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_xneg = -hop_xneg * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_ypos =  hop_ypos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_yneg = -hop_yneg * cx_double(0.0,1.0/sqrt(n));
            int pos_i = pos_square(xi, yi, lx);
            int pos_j;

            Vec<int> xyj = {(lx+xi-1)%lx, yi};
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xpos);
            j1_elems.add(pos_i, pos_j, &crt_xpos);

            xyj[0] = (lx+xi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xneg);
            j1_elems.add(pos_i, pos_j, &crt_xneg);

            xyj[0] = xi;
            xyj[1] = (lx+yi-1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_ypos);
            j2_elems.add(pos_i, pos_j, &crt_ypos);

            xyj[1] = (lx+yi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_yneg);
            j2_elems.add(pos_i, pos_j, &crt_yneg);
        }
    }
    for (int i = 0; i < n; i++) {
        cx_double val(0.0, 0.0);
        H_elems.add(i, i, &val);
    }
    fkpm::SpMatBsr<cx_double> j1_BSR(j1_elems);
    fkpm::SpMatBsr<cx_double> j2_BSR(j2_elems);
    j1_elems.clear();
    j2_elems.clear();
    fkpm::SpMatBsr<cx_double> H(H_elems);
    H_elems.clear();

    auto es = engine->energy_scale(H, 0.1);
    engine->set_H(H, es);

    fkpm::RNG rng(0);
    engine->set_R_uncorrelated(n, s, rng);

    auto mu_xy = engine->moments2_v1(M, j2_BSR, j1_BSR);
    std::cout << "mu[7,7]  = " << mu_xy[7][7] << "\n   [expect (0.00242202,-0.00292942)]" << std::endl;
    std::cout << "mu[3,5]  = " << mu_xy[3][5] << "\n   [expect (-0.0013152,-0.235975)]" << std::endl;
    std::cout << "mu[5,3]  = " << mu_xy[5][3] << "\n   [expect (-0.00104142,0.233898)]" << std::endl;
    std::cout << "mu[9,17] = " << mu_xy[9][17] << "\n   [expect (0.00215957,0.148249)]" << std::endl;
    std::cout << "mu[17,9] = " << mu_xy[17][9] << "\n   [expect (-0.000153006,-0.14291)]" << std::endl;
    auto cmn = fkpm::electrical_conductivity_coefficients(M, Mq, kT, -3.5, 0.0, es, kernel);
    std::cout << "cmn[8,5] = " << cmn[8][5] << "\n   [expect (3465.91,-9.31494)]" << std::endl;
    std::cout << "cmn[13,20]=" << cmn[13][20] << "\n   [expect (928.206,15.9604)]" << std::endl;
    std::cout << "sigma_{xy}(mu = -3.5) = " << std::real(fkpm::moment_product(cmn, mu_xy))
              << "\n                [expect 1.52646]" << std::endl;

    cmn = fkpm::electrical_conductivity_coefficients(M, Mq, kT, -2.8, 0.0, es, kernel);
    std::cout << "sigma_{xy}(mu = -2.8) = " << std::real(fkpm::moment_product(cmn, mu_xy))
              << "\n                [expect 2.44527]" << std::endl;

    H.clear();
    j1_BSR.clear();
    j2_BSR.clear();
}


// triangular lattice
void testConductivity2() {
    std::cout << std::endl << "test2: conductivity on triangular lattice." << std::endl;
    int w = 100, h = 100;
    auto m = SimpleModel::mk_triangular(w, h);
    m->J = 5.0 * sqrt(3.0);
    m->t1 = -1;
    int M = 40;
    int Mq = 2*M;
    int n_colors = 16;
    auto kernel = fkpm::jackson_kernel(M);
    auto engine = fkpm::mk_engine<cx_flt>();

    m->set_spins("allout", toml_from_str(""), m->spin);
    m->set_hamiltonian(m->spin);

    auto es = engine->energy_scale(m->H, 0.1);
    engine->set_H(m->H, es);

    fkpm::RNG rng(0);
    engine->set_R_uncorrelated(m->H.n_rows, 2*n_colors, rng);

    auto jx = m->electric_current_operator(m->spin, {1,0,0});
    auto jy = m->electric_current_operator(m->spin, {0,1,0});
    auto mu_xx = engine->moments2_v1(M, jx, jx);
    auto mu_xy = engine->moments2_v1(M, jx, jy);
    std::cout << "muxx[3,5]  = " << mu_xx[3][5] << "\n     [expect (-0.664472,0.00428363)]" << std::endl;
    std::cout << "muxx[10,10]= " << mu_xx[10][10] << "\n     [expect (-1.4545,-0.00293068)]" << std::endl;
    std::cout << "muxy[12,2] = " << mu_xy[12][2] << "\n     [expect (-0.000104344,-0.089929)]" << std::endl;
    std::cout << "muxy[25,14]= " << mu_xy[25][14] << "\n     [expect (-0.00318346,-0.0581466)]" << std::endl;

    auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), -10.5, 0.0, es, kernel);
    std::cout << "cmn[30,30] = " << cmn[30][30] << "\n     [expect (-1.54812,0)]" << std::endl;
    std::cout << "sigma_{xx}(mu = -10.5) = " << std::real(fkpm::moment_product(cmn, mu_xx))
              << "\n                 [expect 1.76712]" << std::endl;
    cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), -9.0, 0.0, es, kernel);
    std::cout << "cmn[9,9]   = " << cmn[9][9] << "\n     [expect (-315.933,0)]" << std::endl;
    std::cout << "sigma_{xy}(mu = -9)    = " << std::real(fkpm::moment_product(cmn, mu_xy))
              << "\n                 [expect -1.50499]" << std::endl;
    cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), 0.0, 0.0, es, kernel);
    std::cout << "sigma_{xx}(mu = 0)     = " << std::real(fkpm::moment_product(cmn, mu_xx))
              << "\n                 [expect -0.00053196]" << std::endl;
    std::cout << "sigma_{xy}(mu = 0)     = " << std::real(fkpm::moment_product(cmn, mu_xy))
              << "\n                 [expect -6.9186e-05]" << std::endl;
    cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), 9.0, 0.0, es, kernel);
    std::cout << "sigma_{xy}(mu = 9)     = " << std::real(fkpm::moment_product(cmn, mu_xy))
              << "\n                 [expect 1.69197]" << std::endl;
}


void test_AndersonModel() {
    int lx = 32;
    int n  = lx * lx * lx;
    int s  = 1;
    int num_replica = 10;
    int M  = 100;
    int Mq = 2*M;

    double hopping = 1.0;
    double W       = 12.0 * hopping;
    double kT      = 1.0 * hopping;
    double mu      =  0.0 * hopping;
    auto kernel    = fkpm::jackson_kernel(M);

    Vec<double> omega;
    for (double omega_elem = 0.2; omega_elem < 15.0; omega_elem += 0.2) omega.push_back(omega_elem);

    // need universal energy scale for all the replicas
    fkpm::EnergyScale es {-1.8*(2.0*hopping+W/2.0), 1.8*(2.0*hopping+W/2.0)};
    auto engine = fkpm::mk_engine<cx_double>();
    std::cout << es << std::endl;

    fkpm::RNG rng(0);
    std::uniform_real_distribution<double> dist_uniform(-W/2.0, W/2.0);

    //engine->set_R_identity(n);
    engine->set_R_uncorrelated(n, s, rng);
    //    Vec<int> groups(n);
    //    for (int i = 0; i < n; i++)
    //        groups[i] = i%s;
    //    engine->set_R_correlated(groups, rng);


    // Build Anderson model (Weisse paper, Eq 111)
    fkpm::SpMatElems<cx_double> elems_base(n, n, 1);
    fkpm::SpMatElems<cx_double> j1_elems(n, n, 1); // longitudinal (x direction)
    fkpm::SpMatElems<cx_double> j2_elems(n, n, 1); // transverse   (y direction)
    for (int xi = 0; xi < lx; xi++) {
        for (int yi = 0; yi < lx; yi++) {
            for (int zi = 0; zi < lx; zi++) {
                int pos_i = pos_cubic(xi, yi, zi, lx);
                int pos_j;
                cx_double v0 = {-hopping, 0.0};
                cx_double v1(0.0, hopping/sqrt(n));
                cx_double v2(0.0, -hopping/sqrt(n));

                Vec<int> xyzj = {(lx+xi-1)%lx, yi, zi};
                pos_j = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);      // +x
                elems_base.add(pos_i, pos_j, &v0);
                j1_elems.add(pos_i, pos_j, &v1);

                xyzj[0] = (lx+xi+1)%lx;
                pos_j   = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);    // -x
                elems_base.add(pos_i, pos_j, &v0);
                j1_elems.add(pos_i, pos_j, &v2);

                xyzj[0] = xi;
                xyzj[1] = (lx+yi-1)%lx;
                pos_j   = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);    // +y
                elems_base.add(pos_i, pos_j, &v0);
                j2_elems.add(pos_i, pos_j, &v1);

                xyzj[1] = (lx+yi+1)%lx;
                pos_j   = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);    // -y
                elems_base.add(pos_i, pos_j, &v0);
                j2_elems.add(pos_i, pos_j, &v2);

                xyzj[1] = yi;
                xyzj[2] = (lx+zi-1)%lx;
                pos_j   = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);    // +z
                elems_base.add(pos_i, pos_j, &v0);

                xyzj[2] = (lx+zi+1)%lx;
                pos_j   = pos_cubic(xyzj[0], xyzj[1], xyzj[2], lx);    // -z
                elems_base.add(pos_i, pos_j, &v0);
            }
        }
    }
    fkpm::SpMatBsr<cx_double> j1_BSR(j1_elems);
    fkpm::SpMatBsr<cx_double> j2_BSR(j2_elems);
    j1_elems.clear();
    j2_elems.clear();

    Vec<Vec<cx_double>> gamma_jxy(Mq);
    Vec<double> optical;
    for (int i = 0; i < Mq; i++) {
        gamma_jxy[i].resize(Mq, cx_double(0.0,0.0));
    }
    optical.resize(omega.size(),0.0);

    cout << "calculating moments2... " << std::flush;
    fkpm::timer[0].reset();
    for (int replica = 0; replica < num_replica; replica++) {
        std::cout << std::endl << "replica #" << replica << std::endl;
        auto elems = elems_base;
        for (int xi = 0; xi < lx; xi++) {
            for (int yi = 0; yi < lx; yi++) {
                for (int zi = 0; zi < lx; zi++) {
                    int pos_i = pos_cubic(xi, yi, zi, lx);
                    cx_double epsiloni = {dist_uniform(rng), 0.0};
                    elems.add(pos_i, pos_i, &epsiloni);
                }
            }
        }
        fkpm::SpMatBsr<cx_double> H(elems);
        elems.clear();
        //auto H_dense = H.to_arma_dense();
        engine->set_H(H, es);

        Vec<Vec<cx_double>> mu_longitudinal = engine->moments2_v1(M, j1_BSR, j1_BSR);
        //Vec<Vec<cx_double>> mu_transverse   = engine->moments_tensor(M, j1_BSR, j2_BSR);

        auto temp = fkpm::moment_transform(mu_longitudinal, Mq, kernel);
        for (int i = 0; i < Mq; i++) {
            for (int j = 0; j < Mq; j++) {
                gamma_jxy[i][j] += temp[i][j];
            }
        }
        for (int i = 0; i < omega.size(); i++) {
            auto cmn    = fkpm::electrical_conductivity_coefficients(M, Mq, kT, mu, omega[i], es, kernel);
            optical[i] += std::real(fkpm::moment_product(cmn, mu_longitudinal));
        }

    }
    for (int i = 0; i < Mq; i++) {             // average over number of replicas
        for (int j = 0; j < Mq; j++) gamma_jxy[i][j] /= num_replica;
    }
    for (int i = 0; i < omega.size(); i++) optical[i] /= num_replica;
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";

    Vec<double> x, y;
    Vec<Vec<cx_double>> jxy;
    fkpm::density_function(gamma_jxy, es, x, y, jxy);
    std::ofstream fout1("jxy.dat");
    fout1 << std::scientific << std::right;
    //fout << std::setw(20) << "x" << std::setw(20) << "y" << std::setw(20) << "jxy" << std::endl;
    for (int i = 5; i < Mq-5; i++) {
        for (int j = 5; j < Mq-5; j++) {
            fout1 << std::setw(20) << x[i] << std::setw(20) << y[j] << std::setw(20) << std::real(jxy[i][j]) << std::endl;
        }
    }
    fout1.close();

    std::ofstream fout2("sigma.dat", std::ios::out | std::ios::app);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "omega" << std::setw(20) << "sigma" << std::endl;
    for (int i = 0; i < omega.size(); i++) {
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << omega[i]
        << std::setw(20) << optical[i] << std::endl;
    }
    fout2.close();

    j1_BSR.clear();
    j2_BSR.clear();
    std::cout << "done!" << std::endl;

}

void test_Hall_SquareLattice() {
    auto engine = fkpm::mk_engine<cx_double>();

    int lx = 100;
    int n  = lx * lx;
    int s  = 10;
    int M  = 200;
    int Mq = 2*M;

    double hopping     = 1.0;
    double B_over_phi0 = 0.05;
    double kT          = 0.001 * hopping;
    auto kernel        = fkpm::jackson_kernel(M);
    fkpm::SpMatElems<cx_double> H_elems(n, n, 1);
    fkpm::SpMatElems<cx_double> j1_elems(n, n, 1); // longitudinal (x direction)
    fkpm::SpMatElems<cx_double> j2_elems(n, n, 1); // transverse   (y direction)
    for (int xi = 0; xi < lx; xi++) {
        for (int yi = 0; yi < lx; yi++) {
            cx_double hop_xpos = -hopping * cx_double(cos(2.0*fkpm::Pi*B_over_phi0*yi),
                                                      -sin(2.0*fkpm::Pi*B_over_phi0*yi));
            cx_double hop_xneg = -hopping * cx_double(cos(2.0*fkpm::Pi*B_over_phi0*yi),
                                                      sin(2.0*fkpm::Pi*B_over_phi0*yi));
            cx_double hop_ypos = {-hopping, 0.0};
            cx_double hop_yneg = {-hopping, 0.0};

            cx_double crt_xpos = -hop_xpos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_xneg =  hop_xneg * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_ypos = -hop_ypos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_yneg =  hop_yneg * cx_double(0.0,1.0/sqrt(n));
            int pos_i = pos_square(xi, yi, lx);
            int pos_j;

            Vec<int> xyj = {(lx+xi-1)%lx, yi};
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xpos);
            j1_elems.add(pos_i, pos_j, &crt_xpos);

            xyj[0] = (lx+xi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xneg);
            j1_elems.add(pos_i, pos_j, &crt_xneg);

            xyj[0] = xi;
            xyj[1] = (lx+yi-1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_ypos);
            j2_elems.add(pos_i, pos_j, &crt_ypos);

            xyj[1] = (lx+yi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_yneg);
            j2_elems.add(pos_i, pos_j, &crt_yneg);
        }
    }
    for (int i = 0; i < n; i++) {
        cx_double val(0.0, 0.0);
        H_elems.add(i, i, &val);
    }
    fkpm::SpMatBsr<cx_double> j1_BSR(j1_elems);
    fkpm::SpMatBsr<cx_double> j2_BSR(j2_elems);
    j1_elems.clear();
    j2_elems.clear();
    fkpm::SpMatBsr<cx_double> H(H_elems);
    H_elems.clear();

    auto es = engine->energy_scale(H, 0.1);
    engine->set_H(H, es);

    fkpm::RNG rng(0);
    engine->set_R_uncorrelated(n, s, rng);

    std::cout << "calculating dos..." << std::endl;
    auto mu_dos = engine->moments(M);
    auto gamma_dos = fkpm::moment_transform(mu_dos, Mq);
    Vec<double> x, rho;
    fkpm::density_function(gamma_dos, es, x, rho);

    std::ofstream fout1("dos_hall.dat", std::ios::out | std::ios::app);
    fout1 << std::scientific << std::right;
    fout1 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "mu" << std::setw(20) << "dos" << std::endl;
    for (int i = 0; i < x.size(); i++) {
        fout1 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << x[i]
        << std::setw(20) << rho[i] << std::endl;
    }
    fout1.close();

    std::cout << "calculating transverse conductivity..." << std::endl;
    auto mu_xy = engine->moments2_v1(M, j2_BSR, j1_BSR, 20);
    std::cout << "moments obtained." << std::endl;

    Vec<double> mu_list;
    double mu_step = (es.hi-es.lo) / Mq / 2.0;
    for (double mu_elem = es.lo; mu_elem < es.hi; mu_elem += mu_step) mu_list.push_back(mu_elem);
    arma::Col<double> sigma_xy;
    sigma_xy.zeros(mu_list.size());

    for (int i = 0; i < mu_list.size(); i++) {
        auto cmn = fkpm::electrical_conductivity_coefficients(M, Mq, kT, mu_list[i], 0.0, es, kernel);
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
    }

    std::ofstream fout2("sigma_hall.dat", std::ios::out | std::ios::app);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "mu" << std::setw(20) << "sigma_xy" << std::endl;
    for (int i = 0; i < mu_list.size(); i++) {
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << mu_list[i]
        << std::setw(20) << sigma_xy(i) << std::endl;
    }
    fout2.close();

    j1_BSR.clear();
    j2_BSR.clear();
    std::cout << "done!" << std::endl;

}

// square lattice, each triangle threaded by quarter flux.
// for spinless electron, at half filling, sigma_{xy}=1
void test_PRL101_156402_v0() {
    auto engine = fkpm::mk_engine<cx_double>();

    int lx = 100;
    int n  = lx * lx;
    int s  = 10;
    int M  = 200;
    int Mq = 2*M;

    double hopping1 = 1.0;
    double hopping2 = 0.8;
    double flux     = 0.25;
    double kT       = 0.001 * hopping1;
    auto kernel     = fkpm::jackson_kernel(M);
    fkpm::SpMatElems<cx_double> H_elems(n, n, 1);
    fkpm::SpMatElems<cx_double> j1_elems(n, n, 1); // longitudinal (x direction)
    fkpm::SpMatElems<cx_double> j2_elems(n, n, 1); // transverse   (y direction)
    for (int xi = 0; xi < lx; xi++) {
        for (int yi = 0; yi < lx; yi++) {
            cx_double hop_xpos = {-hopping1, 0.0};
            cx_double hop_xneg = {-hopping1, 0.0};
            cx_double hop_ypos = -hopping1 * cx_double(cos(4.0*fkpm::Pi*flux*xi), sin(4.0*fkpm::Pi*flux*xi));
            cx_double hop_yneg = -hopping1 * cx_double(cos(4.0*fkpm::Pi*flux*xi),-sin(4.0*fkpm::Pi*flux*xi));
            cx_double hop_xpyp = -hopping2 * cx_double(cos(2.0*fkpm::Pi*flux*(2.0*xi+1.0)), sin(2.0*fkpm::Pi*flux*(2.0*xi+1.0)));
            cx_double hop_xpyn = -hopping2 * cx_double(cos(2.0*fkpm::Pi*flux*(2.0*xi-1.0)),-sin(2.0*fkpm::Pi*flux*(2.0*xi-1.0)));

            cx_double crt_xpos = -hop_xpos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_xneg =  hop_xneg * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_ypos = -hop_ypos * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_yneg =  hop_yneg * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_xpyp = -hop_xpyp * cx_double(0.0,1.0/sqrt(n));
            cx_double crt_xpyn =  hop_xpyn * cx_double(0.0,1.0/sqrt(n));
            int pos_i = pos_square(xi, yi, lx);
            int pos_j;

            Vec<int> xyj = {(lx+xi-1)%lx, yi};
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xpos);
            j1_elems.add(pos_i, pos_j, &crt_xpos);

            xyj[0] = (lx+xi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xneg);
            j1_elems.add(pos_i, pos_j, &crt_xneg);

            xyj[0] = xi;
            xyj[1] = (lx+yi-1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_ypos);
            j2_elems.add(pos_i, pos_j, &crt_ypos);

            xyj[1] = (lx+yi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_yneg);
            j2_elems.add(pos_i, pos_j, &crt_yneg);

            xyj[0] = (lx+xi-1)%lx;
            xyj[1] = (lx+yi-1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xpyp);
            j1_elems.add(pos_i, pos_j, &crt_xpyp);
            j2_elems.add(pos_i, pos_j, &crt_xpyp);

            xyj[0] = (lx+xi+1)%lx;
            xyj[1] = (lx+yi+1)%lx;
            pos_j = pos_square(xyj[0], xyj[1], lx);
            H_elems.add(pos_i, pos_j, &hop_xpyn);
            j1_elems.add(pos_i, pos_j, &crt_xpyn);
            j2_elems.add(pos_i, pos_j, &crt_xpyn);
        }
    }
    for (int i = 0; i < n; i++) {
        cx_double val(0.0, 0.0);
        H_elems.add(i, i, &val);
    }
    fkpm::SpMatBsr<cx_double> j1_BSR(j1_elems);
    fkpm::SpMatBsr<cx_double> j2_BSR(j2_elems);
    j1_elems.clear();
    j2_elems.clear();
    fkpm::SpMatBsr<cx_double> H(H_elems);
    H_elems.clear();

    auto es = engine->energy_scale(H, 0.1);
    //fkpm::EnergyScale es {-5.0, 5.0};
    engine->set_H(H, es);

    fkpm::RNG rng(0);
    engine->set_R_uncorrelated(n, s, rng);

    std::cout << "calculating dos..." << std::endl;
    auto mu_dos = engine->moments(M);
    auto gamma_dos = fkpm::moment_transform(mu_dos, Mq);
    Vec<double> x, rho;
    fkpm::density_function(gamma_dos, es, x, rho);

    std::ofstream fout1("dos_Martin_v0.dat", std::ios::out | std::ios::app);
    fout1 << std::scientific << std::right;
    fout1 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "mu" << std::setw(20) << "dos" << std::endl;
    for (int i = 0; i < x.size(); i++) {
        fout1 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << x[i]
        << std::setw(20) << rho[i] << std::endl;
    }
    fout1.close();

    std::cout << "calculating transverse conductivity..." << std::endl;
    auto mu_xy = engine->moments2_v1(M, j2_BSR, j1_BSR, 20);
    std::cout << "moments obtained." << std::endl;

    Vec<double> mu_list;
    double mu_step = (es.hi-es.lo) / Mq / 2.0;
    for (double mu_elem = es.lo; mu_elem < es.hi; mu_elem += mu_step) mu_list.push_back(mu_elem);
    arma::Col<double> sigma_xy;
    sigma_xy.zeros(mu_list.size());

    for (int i = 0; i < mu_list.size(); i++) {
        auto cmn = fkpm::electrical_conductivity_coefficients(M, Mq, kT, mu_list[i], 0.0, es, kernel);
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
    }

    std::ofstream fout2("sigma_Martin_v0.dat", std::ios::out | std::ios::app);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "mu" << std::setw(20) << "sigma_xy" << std::endl;
    for (int i = 0; i < mu_list.size(); i++) {
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << mu_list[i]
        << std::setw(20) << sigma_xy(i) << std::endl;
    }
    fout2.close();

    j1_BSR.clear();
    j2_BSR.clear();
    std::cout << "done!" << std::endl;

}


// bench mark Ivar Martin and Cristian's PRL 101, 156402 (2008)
// triangular lattice, chiral spin ordering
// S_1 = ( 1, 1, 1)
// S_2 = ( 1,-1,-1)
// S_3 = (-1, 1,-1)
// S_4 = (-1,-1, 1)
void test_PRL101_156402_v1() {
    auto engine = fkpm::mk_engine<cx_double>();

    int lx = 100;
    int n  = lx * lx;
    int s  = 40;
    int M  = 200;
    int Mq = 2*M;

    double hopping = 1.0;
    double Jkondo  = 5.0;
    double kT      = 0.001 * hopping;
    double volume  = std::sqrt(3.0)/2.0;
    auto kernel    = fkpm::jackson_kernel(M);

    // Build current operators and the hopping part of H on triangular lattice
    fkpm::SpMatElems<cx_double> H_elems(2*n, 2*n, 1);
    fkpm::SpMatElems<cx_double> j1_elems(2*n, 2*n, 1); // longitudinal (x direction)
    fkpm::SpMatElems<cx_double> j2_elems(2*n, 2*n, 1); // transverse   (y direction)
    for (int xi = 0; xi < lx; xi++) {
        cx_double v0 = {-hopping, 0.0};
        cx_double v1(0.0,  hopping/sqrt(n)/sqrt(volume));
        cx_double v2(0.0, -hopping/sqrt(n)/sqrt(volume));
        cx_double v3(0.0,  0.5*hopping/sqrt(n)/sqrt(volume));
        cx_double v4(0.0, -0.5*hopping/sqrt(n)/sqrt(volume));
        cx_double v5(0.0,  0.5*sqrt(3.0)*hopping/sqrt(n)/sqrt(volume));
        cx_double v6(0.0, -0.5*sqrt(3.0)*hopping/sqrt(n)/sqrt(volume));
        for (int yi = 0; yi < lx; yi++) {
            int pos_i = pos_square(xi, yi, lx);
            int pos_j;

            Vec<int> xyj = {(lx+xi-1)%lx, yi};
            pos_j = pos_square(xyj[0], xyj[1], lx);      // +x
            H_elems.add(pos_i, pos_j, &v0);              // spin up
            j1_elems.add(pos_i, pos_j, &v1);
            H_elems.add(pos_i + n, pos_j + n, &v0);      // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v1);

            xyj[0] = (lx+xi+1)%lx;
            pos_j  = pos_square(xyj[0], xyj[1], lx);    // -x
            H_elems.add(pos_i, pos_j, &v0);             // spin up
            j1_elems.add(pos_i, pos_j, &v2);
            H_elems.add(pos_i + n, pos_j + n, &v0);     // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v2);

            xyj[0] = xi;
            xyj[1] = (lx+yi-1)%lx;
            pos_j  = pos_square(xyj[0], xyj[1], lx);     // angle pi/3
            H_elems.add(pos_i, pos_j, &v0);              // spin up
            j1_elems.add(pos_i, pos_j, &v3);
            j2_elems.add(pos_i, pos_j, &v5);
            H_elems.add(pos_i + n, pos_j + n, &v0);      // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v3);
            j2_elems.add(pos_i + n, pos_j + n, &v5);

            xyj[1] = (lx+yi+1)%lx;
            pos_j  = pos_square(xyj[0], xyj[1], lx);     // angle 4pi/3
            H_elems.add(pos_i, pos_j, &v0);              // spin up
            j1_elems.add(pos_i, pos_j, &v4);
            j2_elems.add(pos_i, pos_j, &v6);
            H_elems.add(pos_i + n, pos_j + n, &v0);      // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v4);
            j2_elems.add(pos_i + n, pos_j + n, &v6);

            xyj[0] = (lx+xi-1)%lx;
            xyj[1] = (lx+yi+1)%lx;
            pos_j  = pos_square(xyj[0], xyj[1], lx);     // angle 2pi/3
            H_elems.add(pos_i, pos_j, &v0);              // spin up
            j1_elems.add(pos_i, pos_j, &v3);
            j2_elems.add(pos_i, pos_j, &v6);
            H_elems.add(pos_i + n, pos_j + n, &v0);      // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v3);
            j2_elems.add(pos_i + n, pos_j + n, &v6);

            xyj[0] = (lx+xi+1)%lx;
            xyj[1] = (lx+yi-1)%lx;
            pos_j  = pos_square(xyj[0], xyj[1], lx);     // angle 5pi/3
            H_elems.add(pos_i, pos_j, &v0);              // spin up
            j1_elems.add(pos_i, pos_j, &v4);
            j2_elems.add(pos_i, pos_j, &v5);
            H_elems.add(pos_i + n, pos_j + n, &v0);      // spin down
            j1_elems.add(pos_i + n, pos_j + n, &v4);
            j2_elems.add(pos_i + n, pos_j + n, &v5);
        }
    }
    fkpm::SpMatBsr<cx_double> j1_BSR(j1_elems);
    fkpm::SpMatBsr<cx_double> j2_BSR(j2_elems);
    j1_elems.clear();
    j2_elems.clear();

    //complete the hamiltonian with the Kondo terms
    for (int xi = 0; xi < lx; xi++) {
        for (int yi = 0; yi < lx; yi++) {
            cx_double v1, v2, v3, v4;
            if (xi % 2 == 0 && yi % 2 == 0) {         // sublattice 1
                v1 = cx_double(-Jkondo, Jkondo);
                v2 = cx_double(-Jkondo,-Jkondo);
                v3 = cx_double(-Jkondo, 0.0);
                v4 = cx_double( Jkondo, 0.0);
            } else if (xi % 2 == 1 && yi % 2 == 0) {  // sublattice 2
                v1 = cx_double(-Jkondo,-Jkondo);
                v2 = cx_double(-Jkondo, Jkondo);
                v3 = cx_double( Jkondo, 0.0);
                v4 = cx_double(-Jkondo, 0.0);
            } else if (xi % 2 == 0 && yi % 2 == 1) {  // sublattice 3
                v1 = cx_double( Jkondo, Jkondo);
                v2 = cx_double( Jkondo,-Jkondo);
                v3 = cx_double( Jkondo, 0.0);
                v4 = cx_double(-Jkondo, 0.0);
            } else {                                  // sublattice 4
                v1 = cx_double( Jkondo,-Jkondo);
                v2 = cx_double( Jkondo, Jkondo);
                v3 = cx_double(-Jkondo, 0.0);
                v4 = cx_double( Jkondo, 0.0);
            }
            int pos_up = pos_square(xi, yi, lx);
            int pos_dn = pos_up + n;
            H_elems.add(pos_up, pos_dn, &v1);
            H_elems.add(pos_dn, pos_up, &v2);
            H_elems.add(pos_up, pos_up, &v3);
            H_elems.add(pos_dn, pos_dn, &v4);
        }
    }
    fkpm::SpMatBsr<cx_double> H(H_elems);
    H_elems.clear();

    auto es = engine->energy_scale(H, 0.1);
    engine->set_H(H, es);

    fkpm::RNG rng(0);
    //engine->set_R_identity(2*n);
    engine->set_R_uncorrelated(2*n, s, rng);
    engine->R2 = engine->R;

    std::cout << "calculating dos..." << std::endl;
    auto mu_dos = engine->moments(M);
    auto gamma_dos = fkpm::moment_transform(mu_dos, Mq);
    Vec<double> x, rho;
    fkpm::density_function(gamma_dos, es, x, rho);

    std::ofstream fout1("dos_Martin_v1.dat", std::ios::out | std::ios::app);
    fout1 << std::scientific << std::right;
    fout1 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "mu" << std::setw(20) << "dos" << std::endl;
    for (int i = 0; i < x.size(); i++) {
        fout1 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << x[i]
        << std::setw(20) << rho[i] << std::endl;
    }
    fout1.close();

    std::cout << "calculating transverse conductivity..." << std::endl;
    //auto mu_xx = engine->moments2_v1(M, j1_BSR, j1_BSR);
    auto mu_xy = engine->moments2_v1(M, j1_BSR, j2_BSR);

    //    auto gamma_jxx = fkpm::moment_transform(mu_xx, Mq, kernel);
    //    Vec<Vec<cx_double>> jxx;
    //    Vec<double> x, y;
    //    fkpm::density_function(gamma_jxx, es, x, y, jxx);
    //    std::ofstream fout1("jxy_prl101_156402.dat");
    //    fout1 << std::scientific << std::right;
    //    for (int i = 5; i < Mq-5; i++) {
    //        for (int j = 5; j < Mq-5; j++) {
    //            fout1 << std::setw(20) << x[i] << std::setw(20) << y[j]
    //            << std::setw(20) << std::real(jxx[i][j]) << std::endl;
    //        }
    //    }
    //    fout1.close();

    Vec<double> mu_list;
    double mu_step = (es.hi-es.lo) / Mq / 2.0;
    for (double mu_elem = es.lo; mu_elem < es.hi; mu_elem += mu_step) mu_list.push_back(mu_elem);
    //mu_list.push_back(0.0);
    arma::Col<double> sigma_xx;
    arma::Col<double> sigma_xy;
    sigma_xx.zeros(mu_list.size());
    sigma_xy.zeros(mu_list.size());
    // comment: slow in the loop below, probably something stupid here
    for (int i = 0; i < mu_list.size(); i++) {
        //std::cout << "mu=" << mu_list[i] << std::endl;
        auto cmn = fkpm::electrical_conductivity_coefficients(M, Mq, kT, mu_list[i], 0.0, es, kernel);
        //        sigma_xx(i) = std::real(fkpm::moment_product(cmn, mu_xx));
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
        //std::cout <<  "mu = " << mu_list[i] << ", sigma_xx = " << sigma_xx(i) << std::endl;
    }


    std::ofstream fout2("sigma_Martin_v1.dat", std::ios::out | std::ios::app);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20)
    << "mu" << std::setw(20) << "sigma_xx" << std::setw(20) << "sigma_xy" << std::endl;
    for (int i = 0; i < mu_list.size(); i++) {
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/kT << std::setw(20) << mu_list[i]
        << std::setw(20) << sigma_xx(i) << std::setw(20) << sigma_xy(i) << std::endl;
    }
    fout2.close();

    j1_BSR.clear();
    j2_BSR.clear();
    std::cout << "done!" << std::endl;
}




void testKondo1_cubic() {//cubic
    auto engine = fkpm::mk_engine<cx_flt>();
    if (engine == nullptr) std::exit(EXIT_FAILURE);


    int w = 6, h = 6, h_z = 6;
    int int_mu, total_mu=100;
    double mu =0.0, del_mu=12.4/total_mu, mu_start=-6.2;

    std::stringstream fname;

    auto m = SimpleModel::mk_cubic(w, h, h_z);
    m->J = 0.2;
    m->t1 = -1;
    m->t3 = 0.5;
    m->s1 = 0.0;
    m->set_spins("ferro", toml_from_str(""), m->spin);
    //m->lattice->set_spins("meron", nullptr, m->spin);
    m->spin[0] = vec3(1, 1, 1).normalized();

    fname << "mu_n_cubic_test_L_" << w << "_t1_" <<  m->t1 << "_t2_" <<  m->t2 << "_t3_" <<  m->t3 << "_J_" <<  m->J << "_01.txt";

    m->set_hamiltonian(m->spin);
    int n = m->H.n_rows;

    std::ofstream dump_file(fname.str(), std::ios::trunc);


    for (int_mu=0; int_mu<total_mu; int_mu++) {
        mu = del_mu*int_mu + mu_start;

        Vec<double> gamma;
        Vec<double> moments;

        arma::vec eigs = arma::conv_to<arma::vec>::from(arma::eig_gen(m->H.to_arma_dense()));
        //double E1 = electronic_grand_energy(eigs, m->kT(), mu) / m->n_sites;

        auto es = engine->energy_scale(m->H, 0.1);
        int M = 2000;
        int Mq = 4*M;
        using std::placeholders::_1;
        auto g_c = expansion_coefficients(M, Mq, std::bind(fkpm::fermi_energy, _1, m->kT(), mu), es);
        auto f_c = expansion_coefficients(M, Mq, std::bind(fkpm::fermi_density, _1, m->kT(), mu), es);
        engine->set_H(m->H, es);
        engine->set_R_identity(n);



        //double E2 = moment_product(g_c, engine->moments(M)) / m->n_sites;

        //cout << "H: " << *m->H(0, 0) << " " << *m->H(1, 0) << "\n  [(-0.288675,0)  (-0.288675,-0.288675)]\n";
        //cout << "   " << *m->H(0, 1) << " " << *m->H(1, 1) << "\n  [(-0.288675,0.288675) (0.288675,0)]\n\n";

        engine->autodiff_matrix(g_c, m->D);
        //cout << "D: " << *m->D(0, 0) << " " << *m->D(1, 0) << "\n  [(0.481926,0) (0.0507966,0.0507966)]\n";
        //cout << "   " << *m->D(0, 1) << " " << *m->D(1, 1) << "\n  [(0.0507966,-0.0507966) (0.450972,0)]\n\n";

        Vec<vec3>& force = m->dyn_stor[0];
        m->set_forces(m->D, m->spin, force);

        //cout << std::setprecision(9);
        //cout << "grand energy " <<  E1 << " " << E2 << "\n            [-1.98657216 -1.98657194]\n";
        //cout << "force " << force[0] << "\n     [<x=0.0507965542, y=0.0507965542, z=-0.384523162>]\n\n";

        moments = engine->moments(M);
        gamma = fkpm::moment_transform(moments, Mq);

        //fprintf(fp1, "%10f, %10lf\n", mu, g.get_unwrap<double>("ensemble.filling"));
        //printf("%10lf, %10f\n", mu, g.get_unwrap<double>("ensemble.filling"));
        double filling =  mu_to_filling(gamma, es, m->kT(), mu);
        dump_file << mu << ", " << filling << endl;
        cout  << mu << ", " << filling << endl;
    }
}

arma::cx_mat transformU(int lx) {
    int n = lx * lx;
    arma::cx_mat ret(n,n);
    ret.zeros();
    for (int kx = 0; kx < lx; kx++) {
        for (int ky = 0; ky < lx; ky++) {
            int pos_k = pos_square(kx, ky, lx);
            for (int ix = 0; ix < lx; ix++) {
                for (int iy = 0; iy < lx; iy++) {
                    int pos_i = pos_square(ix, iy, lx);
                    ret(pos_k,pos_i) = std::exp(cx_double(0.0, 2.0 * (kx*ix + ky*iy) * Pi / n)) / std::sqrt(n);
                }
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << "i,j=" << i << "," << j << ", ret=" << ret(i,j) << std::endl;
        }
    }
    arma::cx_mat test(n,n);
    test = arma::trans(ret);
    test = ret * test;
    //    for (int i = 0; i < n; i++) {
    //        if (std::abs(test(i,i)-1.0)>1e-9) {
    //            std::cout << "test(" << i << "," << i << ")=" << test(i,i) << std::endl;
    //        }
    //        for (int j = 0; j < n; j++) {
    //            if (i != j && std::abs(test(i,j))>1e-9) {
    //                std::cout << "test(" << i << "," << j << ")=" << test(i,j) << std::endl;
    //            }
    //        }
    //    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << "i,j=" << i << "," << j << ", test=" << test(i,j) << std::endl;
        }
    }
    return ret;
}


// triangular lattice
void testKondo6() {
    auto engine = fkpm::mk_engine<cx_flt>();

    int w = 100, h = 100;
    auto m = SimpleModel::mk_triangular(w, h);
    m->J = 5.0 * sqrt(3.0);
    m->t1 = -1;
    int M = 200;
    int Mq = M;
    int n_colors = 12;
    auto kernel = fkpm::jackson_kernel(M);


    m->set_spins("allout", toml_from_str(""), m->spin);
    m->set_hamiltonian(m->spin);

    auto es = engine->energy_scale(m->H, 0.1);
    engine->set_H(m->H, es);

    fkpm::RNG rng(0);
    //engine->set_R_correlated(m->groups(n_colors), rng);
    engine->set_R_uncorrelated(m->H.n_rows, 2*n_colors, rng);
    engine->R2 = engine->R;

    //    auto u_fourier = transformU(4);

    auto jx = m->electric_current_operator(m->spin, {1,0,0});
    auto jy = m->electric_current_operator(m->spin, {0,1,0});

    cout << "calculating moments2... " << std::flush;
    fkpm::timer[0].reset();
    auto mu_xy = engine->moments2_v1(M, jx, jy);
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";

    cout << "T=" << m->kT();

    cout << "calculating xy conductivities... " << std::flush;
    std::ofstream fout2("test.dat", std::ios::out /* | std::ios::app */);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20) << "mu" << std::setw(20) << "sigma_xy" << std::endl;
    arma::Col<double> sigma_xy(Mq);
    sigma_xy.zeros();
    for (int i = 0; i < Mq; i++) {
        double mu = es.lo + i * (es.hi-es.lo) / Mq;
        auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), mu, 0.0, es, kernel);
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/m->kT() << std::setw(20) << mu << std::setw(20) << sigma_xy(i) << std::endl;
    }
    fout2.close();
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";

    //    auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), -10.0, 0.0, es, kernel);
    //    std::ofstream fout6("cmn_mu_m10_RE.dat", std::ios::out /* | std::ios::app */);
    //    std::ofstream fout7("cmn_mu_m10_IM.dat", std::ios::out /* | std::ios::app */);
    //    for (int m1 = 0; m1 < M; m1++) {
    //        for (int m2 = 0; m2 < M; m2++) {
    //            fout6 << std::setw(20) << std::real(cmn[m1][m2]);
    //            fout7 << std::setw(20) << std::imag(cmn[m1][m2]);
    //        }
    //        fout6 << endl;
    //        fout7 << endl;
    //    }
    //    fout6.close();
    //    fout7.close();
    //
    //    cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), -9.0, 0.0, es, kernel);
    //    std::ofstream fout8("cmn_mu_m9_RE.dat", std::ios::out /* | std::ios::app */);
    //    std::ofstream fout9("cmn_mu_m9_IM.dat", std::ios::out /* | std::ios::app */);
    //    for (int m1 = 0; m1 < M; m1++) {
    //        for (int m2 = 0; m2 < M; m2++) {
    //            fout8 << std::setw(20) << std::real(cmn[m1][m2]);
    //            fout9 << std::setw(20) << std::imag(cmn[m1][m2]);
    //        }
    //        fout8 << endl;
    //        fout9 << endl;
    //    }
    //    fout8.close();
    //    fout9.close();
    //
    //    cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), -7.5, 0.0, es, kernel);
    //    std::ofstream fout10("cmn_mu_m7d5_RE.dat", std::ios::out /* | std::ios::app */);
    //    std::ofstream fout11("cmn_mu_m7d5_IM.dat", std::ios::out /* | std::ios::app */);
    //    for (int m1 = 0; m1 < M; m1++) {
    //        for (int m2 = 0; m2 < M; m2++) {
    //            fout10 << std::setw(20) << std::real(cmn[m1][m2]);
    //            fout11 << std::setw(20) << std::imag(cmn[m1][m2]);
    //        }
    //        fout10 << endl;
    //        fout11 << endl;
    //    }
    //    fout10.close();
    //    fout11.close();
    //
    //    auto dmn = electrical_conductivity_coefficients(M, Mq, m->kT(), std::sqrt(2.0), 0.0, es, kernel);
    //    std::ofstream fout16("cmn_mu_sqrt2_RE.dat", std::ios::out /* | std::ios::app */);
    //    std::ofstream fout17("cmn_mu_sqrt2_IM.dat", std::ios::out /* | std::ios::app */);
    //    for (int m1 = 0; m1 < M; m1++) {
    //        for (int m2 = 0; m2 < M; m2++) {
    //            fout16 << std::setw(20) << std::real(dmn[m1][m2]);
    //            fout17 << std::setw(20) << std::imag(dmn[m1][m2]);
    //        }
    //        fout16 << endl;
    //        fout17 << endl;
    //    }
    //    fout10.close();
    //    fout11.close();
    //
    //    auto f = std::bind(fkpm::fermi_density, _1, m->kT(), -9.0);
    //    auto f_c = expansion_coefficients(M, Mq, f, es);

    //    std::ofstream fout13("c_mu_m9_RE.dat", std::ios::out /* | std::ios::app */);
    //    for (int m = 0; m < M; m++) {
    //        fout13 << std::setw(20)<< m << std::setw(20) << f_c[m] << std::endl;
    //    }
    //    fout13.close();
    //
    //    f = std::bind(fkpm::fermi_density, _1, m->kT(), -7.5);
    //    f_c = expansion_coefficients(M, Mq, f, es);
    //    std::ofstream fout14("c_mu_m7d5_RE.dat", std::ios::out /* | std::ios::app */);
    //    for (int m = 0; m < M; m++) {
    //        fout14 << std::setw(20)<< m << std::setw(20) << f_c[m] << std::endl;
    //    }
    //    fout14.close();
    //
    //    f = std::bind(fkpm::fermi_density, _1, m->kT(), -10.0);
    //    f_c = expansion_coefficients(M, Mq, f, es);
    //    std::ofstream fout15("c_mu_m10_RE.dat", std::ios::out /* | std::ios::app */);
    //    for (int m = 0; m < M; m++) {
    //        fout15 << std::setw(20)<< m << std::setw(20) << f_c[m] << std::endl;
    //    }
    //    fout15.close();

}

void testKondo7() {
    auto engine = fkpm::mk_engine<cx_flt>();

    int w = 32, h = 32;
    auto m = SimpleModel::mk_kagome(w, h);
    m->J = 15.0 * sqrt(3.0);
    m->t1 = -3.5;
    int M = 300;
    int Mq = 2*M;
    int Lc = 4;
    int n_colors = 3 * Lc * Lc;
    auto kernel = fkpm::jackson_kernel(M);

    //m->set_spins("allout", mk_toml(""), m->spin);
    double dz = 0.55;
    for(int i=0; i<m->n_sites; i++) {
        int v = i % 3;
        vec3 s(0, 0, 0);
        if(v == 0) s = vec3(-0.5 * sqrt(3.), -0.5, dz);
        else if(v == 1) s = vec3(0, 1, dz);
        else s = vec3(+0.5 * sqrt(3.), -0.5, dz);

        m->spin[i] = s.normalized();
    }

    m->set_hamiltonian(m->spin);

    auto es = engine->energy_scale(m->H, 0.1);

    engine->set_H(m->H, es);

    fkpm::RNG rng(0);
    //engine->set_R_correlated(m->groups(n_colors), rng);
    engine->set_R_uncorrelated(m->H.n_rows, 2*n_colors, rng);
    engine->R2 = engine->R;

    auto moments = engine->moments(M);
    std::cout << "time to calculate moments : " << fkpm::timer[0].measure() << "\n";
    auto gamma = fkpm::moment_transform(moments, Mq);

    std::ofstream fs("dos.dat");
    Vec<double> x, rho, irho;
    fkpm::density_function(gamma, es, x, rho);
    fkpm::integrated_density_function(gamma, es, x, irho);
    for (int i = 0; i < x.size(); i++) {
        fs << x[i] << " " << rho[i] / m->H.n_rows << " " << irho[i] / m->H.n_rows << "\n";
    }
    fs.close();

    //    auto u_fourier = transformU(4);

    //double area = 4. * w*h*sqrt(3.0)/2.0;
    auto jx = m->electric_current_operator(m->spin, {1,0,0});
    auto jy = m->electric_current_operator(m->spin, {0,1,0});
    //jx.scale(1/sqrt(area));
    //jy.scale(1/sqrt(area));

    cout << "calculating moments2... " << std::flush;
    fkpm::timer[0].reset();
    auto mu_xy = engine->moments2_v1(M, jx, jy, 3);
    auto mu_xx = engine->moments2_v1(M, jx, jx, 3);
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";

    cout << "calculating xy conductivities... " << std::flush;
    std::ofstream fout2("test.dat", std::ios::out /* | std::ios::app */);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#M" << std::setw(20) << "beta" << std::setw(20) << "mu" << std::setw(20) << "sigma_xy" << std::endl;
    double n_mus = 2*Mq;
    arma::Col<double> sigma_xy(n_mus);
    arma::Col<double> sigma_xx(n_mus);
    sigma_xy.zeros();
    sigma_xx.zeros();
    for (int i = 0; i < n_mus; i++) {
        double mu = es.lo + i * (es.hi-es.lo) / n_mus;
        auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), mu, 0.0, es, kernel);
        sigma_xy(i) = std::real(fkpm::moment_product(cmn, mu_xy));
        sigma_xx(i) = std::real(fkpm::moment_product(cmn, mu_xx));
        fout2 << std::setw(20) << M << std::setw(20) << 1.0/m->kT() << std::setw(20) << mu << std::setw(20) << sigma_xy(i) << '\t' << sigma_xx(i) << std::endl;
    }
    fout2.close();
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";
}


int main(int argc, char **argv) {
    testConductivity1();
    testConductivity2();
    //test_AndersonModel();
    //test_PRL101_156402_v0();
    //test_PRL101_156402_v1();
    //test_Hall_SquareLattice();
    //    testKondo1_cubic();
    //    testKondo6();
    //    testKondo7();
}
