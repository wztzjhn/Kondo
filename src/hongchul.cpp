#include <iomanip>
#include <cassert>
#include "iostream_util.h"
#include "fastkpm.h"
#include "wannier.h"
using fkpm::Vec;
using fkpm::cx_float;
using fkpm::cx_double;


template <typename T>
double exact_energy(arma::Mat<T> const& H, double kT, double mu) {
    return fkpm::electronic_grand_energy(arma::real(arma::eig_gen(H)), kT, mu);
}

void testExpansionCoeffs() {
    int M = 10;
    Vec<int> Mqs { 1*M, 10*M, 100*M, 1000*M };
    fkpm::EnergyScale es { -1, +1 };
    for (int Mq : Mqs) {
        auto f = [](double e) { return e < 0.12 ? e : 0; };
        auto c = expansion_coefficients(M, Mq, f, es);
        std::cout << "quad. points=" << Mq << ", c=" << c << std::endl;
        Vec<double> c_exact { -0.31601, 0.4801, -0.185462, -0.000775683, 0.0255405, 0.000683006,
            -0.00536911, -0.000313801, 0.000758494, 0.0000425209 };
        std::cout << "mathematica c = " << c_exact << std::endl;
    }
}

void testMat() {
    int n = 20;
    
    fkpm::RNG rng(0);
    std::uniform_int_distribution<uint32_t> uniform(0,n-1);
    fkpm::SpMatElems<double> elems(n, n, 1);
    for (int k = 1; k < 400; k++) {
        int i = uniform(rng);
        int j = uniform(rng);
        double v = k;
        elems.add(i, j, &v);
    }
    fkpm::SpMatBsr<double> H(elems);
    arma::sp_mat Ha = H.to_arma().st();
    
    for (int p = 0; p < Ha.n_nonzero; p++) {
        assert(Ha.row_indices[p] == H.col_idx[p]);
    }
    cout << "Column indices match.\n";
    for (int j = 0; j <= Ha.n_cols; j++) {
        assert(Ha.col_ptrs[j] == H.row_ptr[j]);
    }
    cout << "Row pointers match.\n";
}

template <typename T>
void testKPM1() {
    int n = 4;
    fkpm::SpMatElems<T> elems(4, 4, 1);
    Vec<T> v {5, -5, 0, 0};
    elems.add(0, 0, &v[0]);
    elems.add(1, 1, &v[1]);
    elems.add(2, 2, &v[2]);
    elems.add(3, 3, &v[3]);
    fkpm::SpMatBsr<T> H(elems);
    auto g = [](double x) { return x*x; };
    auto f = [](double x) { return 2*x; }; // dg/dx
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    auto engine = fkpm::mk_engine<T>();
    engine->set_R_identity(n);
    engine->set_H(H, es);
    
    int M = 1000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, g, es);
    auto f_c = expansion_coefficients(M, Mq, f, es);
    
    auto mu = engine->moments(M);
    
    double E1 = fkpm::moment_product(g_c, mu);
    std::cout << std::setprecision(12);
    cout << "energy (v1) " << E1 << " expected 50.000432961 for M=1000\n";
    
    auto gamma = fkpm::moment_transform(mu, Mq);
    double E2 = fkpm::density_product(gamma, g, es);
    cout << "energy (v2) " << E2 << endl;
    
    auto D = H;
    engine->autodiff_matrix(g_c, D);

    cout << "derivative <";
    for (int i = 0; i < 4; i++)
        cout << *D(i, i) << " ";
    cout << ">\n";
    cout << "expected   <9.99980319956 -9.99980319958 8.42069190159e-14 8.42069190159e-14 >\n";
}

void testKPM2() {
    int n = 100;
    int s = n/4;
    double noise = 0.2;
    fkpm::RNG rng(0);
    std::normal_distribution<double> normal;
    
    // Build noisy tri-diagonal matrix
    fkpm::SpMatElems<cx_double> elems(n, n, 1);
    for (int i = 0; i < n; i++) {
        auto x = 1.0 + noise * cx_double(normal(rng), normal(rng));
        int j = (i-1+n)%n;
        cx_double v1 = 0, v2 = x, v3 = conj(x);
        elems.add(i, i, &v1);
        elems.add(i, j, &v2);
        elems.add(j, i, &v3);
    }
    fkpm::SpMatBsr<cx_double> H(elems);
    auto H_dense = H.to_arma_dense();
    
    double kT = 0.0;
    double mu = 0.2;
    
    using std::placeholders::_1;
    auto g = std::bind(fkpm::fermi_energy, _1, kT, mu);
    auto f = std::bind(fkpm::fermi_density, _1, kT, mu);
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    int M = 2000;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, g, es);
    auto f_c = expansion_coefficients(M, Mq, f, es);
    auto engine = fkpm::mk_engine<cx_double>();
    engine->set_H(H, es);
    auto D = H;
    
    double E1 = exact_energy(H_dense, kT, mu);
    double eps = 1e-6;
    int i=0, j=1;
    
    arma::sp_cx_mat dH(n, n);
    dH(i, j) = eps;
    dH(j, i) = eps;
    double dE_dH_1 = (exact_energy(H_dense+dH, kT, mu)-exact_energy(H_dense-dH, kT, mu)) / (2*eps);
    
    engine->set_R_identity(n);
    double E2 = fkpm::moment_product(g_c, engine->moments(M));
    engine->stoch_matrix(f_c, D);
    auto dE_dH_2 = *D(i, j) + *D(j, i);
    
    engine->set_R_uncorrelated(n, s, rng);
    double E3 = fkpm::moment_product(g_c, engine->moments(M));
    engine->autodiff_matrix(g_c, D);
    auto dE_dH_3 = *D(i, j) + *D(j, i);
    
    Vec<int> groups(n);
    for (int i = 0; i < n; i++)
        groups[i] = i%s;
    engine->set_R_correlated(groups, rng);
    double E4 = fkpm::moment_product(g_c, engine->moments(M));
    engine->stoch_matrix(f_c, D);
    auto dE_dH_4 = (*D(i, j) + *D(j, i));
    
    engine->moments(M);
    engine->autodiff_matrix(g_c, D);
    auto dE_dH_5 = (*D(i, j) + *D(j, i));
    
    cout << std::setprecision(15);
    cout << "Exact energy            " << E1 << endl;
    cout << "Det. KPM energy         " << E2 << endl;
    cout << "Stoch. energy (uncorr.) " << E3 << endl;
    cout << "Stoch. energy (corr.)   " << E4 << endl << endl;
    
    cout << "Exact deriv.            " << dE_dH_1 << endl;
    cout << "Det. KPM deriv.         " << dE_dH_2 << endl;
    cout << "Stoch. deriv. (uncorr.) " << dE_dH_3 << endl;
    cout << "Stoch. deriv. (corr.)   " << dE_dH_4 << endl;
    cout << "Autodif. deriv. (corr.) " << dE_dH_5 << endl;
}

void testKPM3() {
    int n = 100;
    double noise = 0.2;
    fkpm::RNG rng(0);
    std::normal_distribution<double> normal;
    
    // Build noisy tri-diagonal matrix
    fkpm::SpMatElems<cx_double> elems(n, n, 1);
    for (int i = 0; i < n; i++) {
        auto x = 1.0 + noise * cx_double(normal(rng), normal(rng));
        int j = (i-1+n)%n;
        cx_double v1 = 0, v2 = x, v3 = conj(x);
        elems.add(i, i, &v1);
        elems.add(i, j, &v2);
        elems.add(j, i, &v3);
    }
    fkpm::SpMatBsr<cx_double> H(elems);
    
    double mu = 0.2;
    using std::placeholders::_1;
    auto g = std::bind(fkpm::fermi_energy, _1, 0, mu);
    auto f = std::bind(fkpm::fermi_density, _1, 0, mu);
    
    double extra = 0.1;
    double tolerance = 1e-2;
    auto es = fkpm::energy_scale(H, extra, tolerance);
    int M = 500;
    int Mq = 4*M;
    auto g_c = expansion_coefficients(M, Mq, g, es);
    auto f_c = expansion_coefficients(M, Mq, f, es);
    auto engine = fkpm::mk_engine<cx_double>();
    engine->set_R_uncorrelated(n, 1, rng);
    
    double eps = 1e-5;
    int i=0, j=1;
    
    auto finite_diff = [&](cx_double eps) {
        auto Hp = H;
        *Hp(i, j) += eps;
        *Hp(j, i) += conj(eps);
        engine->set_H(Hp, es);
        double Ep = fkpm::moment_product(g_c, engine->moments(M));
        auto Hm = H;
        *Hm(i, j) -= eps;
        *Hm(j, i) -= conj(eps);
        engine->set_H(Hm, es);
        double Em = fkpm::moment_product(g_c, engine->moments(M));
        return 0.5 * (Ep - Em) / (2.0*eps);
    };
    
    cx_double dE_dH = finite_diff(eps) - finite_diff(eps*cx_double(0, 1));
    
    engine->set_H(H, es);
    auto D1 = H;
    engine->moments(M);
    engine->autodiff_matrix(g_c, D1);
    
    std::cout << std::setprecision(10);
    cout << dE_dH << endl;
    cout << *D1(i, j) << endl;
}

#include <boost/algorithm/string.hpp>


void test_readfile(int argc, char **argv) {
//    std::string filename = argv[1];
    std::string filename = "/Users/chhcl/Code/DATA/SrVO3.dat";
    std::ifstream f_params(filename);
    std::string line;
  
    //Head line
    std::getline(f_params, line);
    
    
    std::vector<std::string> strs;
    
    //number of orbitals
    std::getline(f_params, line);
    boost::trim_if(line, boost::is_any_of("\t\n "));// remove the white spaces
    boost::split(strs, line, boost::is_any_of("\t\n "), boost::token_compress_on);
    std::cout << strs.size() << std::endl;
    int num_orbitals=std::stof(strs[0]);
   
    
    //number of k-points
    std::getline(f_params, line);
    boost::trim_if(line, boost::is_any_of("\t\n "));
    boost::split(strs, line, boost::is_any_of("\t\n "), boost::token_compress_on);
    std::cout << strs.size() << std::endl;
    int num_kpts=std::stof(strs[0]);
    
    int weight_kpts[num_kpts];
    int kpts_i=0;
    
    int temp_kpts=0;
    
    while (std::getline(f_params, line)) {
       // std::vector<std::string> strs;
        boost::trim_if(line, boost::is_any_of("\t\n "));
        boost::split(strs, line, boost::is_any_of("\t\n "), boost::token_compress_on);
        std::cout << "size="<<strs.size() << std::endl;
        
        if (temp_kpts<num_kpts){
            // store the kpoint-weiht informations
            temp_kpts+=strs.size();
            for (int i = 0; i < strs.size(); i++) {
                double x = std::stof(strs[i]);
                std::cout << x << std::endl;
                weight_kpts[kpts_i++]=x; //consider error
            }
        }
        else{
            //store the Hopping parameters
            for (int i = 0; i < strs.size(); i++) {
                double x = std::stof(strs[i]);
                std::cout << x << std::endl;
            }
        }
    }
    std::cout<<"num_orbitals="<<num_orbitals<<endl;
    std::cout<<"num_kpts="<<num_kpts<<endl;
    std::cout<<"temp_kpts="<<temp_kpts<<endl;
    std::cout<<"kpts_i="<<kpts_i<<endl;

}

int main(int argc, char **argv) {
    // testExpansionCoeffs();
//    testMat();
//    testKPM1<cx_float>();
//    testKPM1<cx_double>();
//    testKPM2();
//    testKPM3();
    
    test_readfile(argc, argv);
}
