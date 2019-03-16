#include "kondo.h"
#include "iostream_util.h"
#include <cassert>


Vec<int> Model::colors_to_groups(Vec<int> const& colors) {
    Vec<int> groups(colors.size()*n_orbs);
    for (int i = 0; i < colors.size(); i++) {
        for (int o = 0; o < n_orbs; o++) {
            groups[i*n_orbs+o] = n_orbs*colors[i] + o;
        }
    }
    return groups;
}

Model::Model(int n_sites, int n_orbs):
    n_sites(n_sites),
    n_orbs(n_orbs),
    H_elems(n_sites*n_orbs, n_sites*n_orbs, 1),
    D_elems(n_sites*n_orbs, n_sites*n_orbs, 1)
{
    spin.assign(n_sites, {0, 0, 0});
    dyn_stor[0].assign(n_sites, {0, 0, 0});
    dyn_stor[1].assign(n_sites, {0, 0, 0});
    dyn_stor[2].assign(n_sites, {0, 0, 0});
    dyn_stor[3].assign(n_sites, {0, 0, 0});
    dyn_stor[4].assign(n_sites, {0, 0, 0});
}

void Model::set_spins_random(fkpm::RNG &rng, Vec<vec3> &spin) {
    static std::normal_distribution<double> dist;
    for (int i = 0; i < spin.size(); i++) {
        spin[i] = vec3(dist(rng), dist(rng), dist(rng)).normalized();
    }
}

double Model::kT() {
    return kT_init * exp(- time*kT_decay);
}

void Model::set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force) {
    for (auto& f : force) {
        f = {0,0,0};
    }
    
    // Site-local forces
    for (int i = 0; i < n_sites; i++) {
        if (spin_exist.empty() || spin_exist[i]) {
            force[i]   += - 2 * s0 * spin[i];
            force[i]   += zeeman;
            force[i].z += 2 * easy_z * spin[i].z;
        }
    }
    
    // Super-exchange forces
    Vec<int> js;
    Vec<double> ss = {s1, s2, s3};
    for (int rank = 0; rank < ss.size(); rank++) {
        if (ss[rank] != 0.0) {
            for (int i = 0; i < n_sites; i++) {
                set_neighbors(rank, i, js);
                for (int j : js) {
                    if (spin_exist.empty() || (spin_exist[i] && spin_exist[j])) {
                        force[i] += - ss[rank] * spin[j];
                    }
                }
            }
        }
    }
    
    // Spin transfer torque
    if (current.norm2() > 0) {
        if (! spin_exist.empty()) {
            std::cout << "dilute spin not implemented yet!" << std::endl;
            assert(false);
        }
        for (int i = 0; i < n_sites; i++) {
            // -- Build matrix of nearest neighbor displacements
            int nn_rank = 0;
            set_neighbors(nn_rank, i, js);
            arma::mat X(js.size(), 3);
            for (int idx = 0; idx < js.size(); idx++) {
                vec3 dr = displacement(js[idx], i);
                X.row(idx) = arma::rowvec{dr.x, dr.y, dr.z};
            }
            
            // -- Calculate directional derivative stencil
            arma::mat gram = X.t()*X;
            // The Gram matrix is positive semi-definite. If any diagonal component is zero, this indicates
            // that all displacement vectors are perpendicular to that cartesian unit vector. To avoid a
            // singularity in the Gram inverse, we create non-zero diagonal elements as needed. In effect, we
            // are assuming that each gradient component is zero unless there is evidence to the contrary.
            for (int dir = 0; dir < 3; dir++) {
                if (gram(dir, dir) == 0.0) gram(dir, dir) = 1.0;
            }
            arma::mat stencil = arma::rowvec{current.x, current.y, current.z} * arma::solve(gram, X.t());
            
            // -- Add spin transfer torque to each force
            for (int idx = 0; idx < js.size(); idx++) {
                vec3 dS = spin[js[idx]] - spin[i];
                X.row(idx) = arma::rowvec{dS.x, dS.y, dS.z};
            }
            arma::rowvec t = stencil * X; // (j dot grad) S
            vec3 torque {t[0], t[1], t[2]};
            force[i] += - spin[i].cross(torque);
        }
    }
}

double Model::energy_classical(Vec<vec3> const& spin) {
    double acc = 0;
    
    // Site-local energy
    for (int i = 0; i < n_sites; i++) {
        if (spin_exist.empty() || spin_exist[i]) {
            acc += s0 * spin[i].norm2();
            acc += - zeeman.dot(spin[i]);
            acc += - easy_z * spin[i].z * spin[i].z;
        }
    }
    
    // Super-exchange energy
    Vec<int> js;
    Vec<double> ss = {s1, s2, s3};
    for (int rank = 0; rank < ss.size(); rank++) {
        if (ss[rank] != 0.0) {
            for (int i = 0; i < n_sites; i++) {
                set_neighbors(rank, i, js);
                for (int j : js) {
                    if (spin_exist.empty() || (spin_exist[i] && spin_exist[j])) {
                        acc += ss[rank] * spin[i].dot(spin[j]) * 0.5; // adjust for double counting
                    }
                }
            }
        }
    }
    
    return acc;
}

void Model::pbc_shear(double& xy, double& xz, double& yz) {
    xy = xz = yz = 0;
}

vec3 Model::displacement(int i, int j) {
    // calculate wrapping vectors
    vec3 dim = dimensions();
    double sxy, sxz, syz;
    pbc_shear(sxy, sxz, syz);
    vec3 vx = vec3{1,     0, 0} * dim.x;
    vec3 vy = vec3{sxy,   1, 0} * dim.y;
    vec3 vz = vec3{sxz, syz, 1} * dim.z;
    
    // row vectors of inverse of [vx, vy, vz] matrix
    vec3 ux = vec3{1, -sxy, -sxz} / dim.x;
    vec3 uy = vec3{0,    1, -syz} / dim.y;
    vec3 uz = vec3{0,    0,    1} / dim.z;
    
    // naive offset
    vec3 dR = position(i) - position(j);
    
    // corrected offset
    dR = (vx * std::remainder(ux.dot(dR), 1) +
          vy * std::remainder(uy.dot(dR), 1) +
          vz * std::remainder(uz.dot(dR), 1));
    
    if (std::abs(dR.dot(vx)) > 0.5*vx.norm2() ||
        std::abs(dR.dot(vy)) > 0.5*vy.norm2() ||
        std::abs(dR.dot(vz)) > 0.5*vz.norm2()) {
        std::cerr << "Detected invalid displacement between sites " << i << " and " << j << "!\n";
        std::cerr << "Try increasing system size or decreasing coupling length scale.\n";
        std::exit(EXIT_FAILURE);
    }
    return dR;
}


SimpleModel::SimpleModel(int n_sites): Model(n_sites, 2) {
}

void SimpleModel::set_hamiltonian(Vec<vec3> const& spin) {
    H_elems.clear();
    D_elems.clear();
    
    // Hund coupling
    cx_flt zero = 0;
    for (int i = 0; i < n_sites; i++) {
        for (int s1 = 0; s1 < 2; s1++) {
            for (int s2 = 0; s2 < 2; s2++) {
                cx_flt v = (spin_exist.empty() || spin_exist[i]) ?
                           -pauli[s1][s2].dot(flt(J)*spin[i] + zeeman) : 0.0;
                H_elems.add(2*i+s1, 2*i+s2, &v);
                D_elems.add(2*i+s1, 2*i+s2, &zero);
            }
        }
    }
    
    // Hopping terms
    Vec<int> js;
    Vec<double> ts = {t1, t2, t3};
    for (int rank = 0; rank < ts.size(); rank++) {
        if (ts[rank] != 0.0) {
            cx_flt t = ts[rank];
            for (int i = 0; i < n_sites; i++) {
                set_neighbors(rank, i, js);
                for (int j : js) {
                    // To handle oscillating current, add phase:
                    // vec3 dim = dimensions();
                    // vec3 delta = displacement(i, j);
                    // flt theta = 2*Pi*(delta.x*current.x/dim.x + delta.y*current.y/dim.y);
                    // theta *= (1 + current_growth*time) * cos(current_freq*time);
                    // cx_flt t = exp(I*theta)*ts[rank];
                    H_elems.add(2*i+0, 2*j+0, &t);
                    H_elems.add(2*i+1, 2*j+1, &t);
                }
            }
        }
    }
    
    H.build(H_elems);
    D.build(D_elems);
}

void SimpleModel::set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force) {
    // Classical forces
    Model::set_forces(D, spin, force);
    
    // Hund-coupling forces
    for (int i = 0; i < n_sites; i++) {
        if (!spin_exist.empty() && !spin_exist[i]) continue;
        Vec3<cx_flt> dE_dS(0, 0, 0);
        for (int s1 = 0; s1 < 2; s1 ++) {
            for (int s2 = 0; s2 < 2; s2 ++) {
                // Apply chain rule: dE/dS = dH_ij/dS D_ji
                // where D_ij = dE/dH_ji is the density matrix
                Vec3<cx_flt> dH_ij_dS = -J * pauli[s1][s2];
                cx_flt D_ji = *D(2*i+s2, 2*i+s1);
                dE_dS += dH_ij_dS * D_ji;
            }
        }
        assert(imag(dE_dS).norm() < 1e-5);
        force[i] += -real(dE_dS);
    }
}


fkpm::SpMatBsr<cx_flt> SimpleModel::electric_current_operator(Vec<vec3> const& spin, vec3 dir) {
    vec3 d = dimensions();
    double sqrt_vol = sqrt(d.x*d.y*d.z);
    
    fkpm::SpMatElems<cx_flt> j_elems(n_sites*n_orbs, n_sites*n_orbs, 1);
    cx_flt zero = 0;
    for (int i = 0; i < n_sites*n_orbs; i++) {
        j_elems.add(i, i, &zero);
    }
    Vec<int> js;
    Vec<double> ts = {t1, t2, t3};
    for (int rank = 0; rank < ts.size(); rank++) {
        if (ts[rank] != 0.0) {
            for (int i = 0; i < n_sites; i++) {
                set_neighbors(rank, i, js);
                for (int j : js) {
                    vec3 dR = displacement(i, j);
                    cx_flt v = -I * flt(dir.normalized().dot(dR) * ts[rank] / sqrt_vol);
                    j_elems.add(2*i+0, 2*j+0, &v);
                    j_elems.add(2*i+1, 2*j+1, &v);
                }
            }
        }
    }
    return fkpm::SpMatBsr<cx_flt>(j_elems);
}


class LinearModel: public SimpleModel {
public:
    int w;
    
    LinearModel(int w): SimpleModel(w), w(w) {
    }
    
    vec3 dimensions() {
        return {double(w), 1, 1};
    }
    
    vec3 position(int i) {
        return {double(i), 0, 0};
    }
    
    void set_spins(std::string const& name, const toml_ptr params, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites, {0, 0, 1});
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
            std::exit(EXIT_FAILURE);
        }
    }
    
    void set_neighbors(int rank, int i, Vec<int>& idx) {
        struct Delta {int x;};
        static Vec<Vec<Delta>> deltas {
            { {-1}, {+1} },
            { {-2}, {+2} },
            { {-3}, {+3} },
        };
        assert(0 <= rank && rank < deltas.size());
        auto d = deltas[rank];
        idx.resize(d.size());
        for (int n = 0; n < idx.size(); n++) {
            idx[n] = positive_mod(i + d[n].x, w);
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        if (n_sites % n_colors != 0) {
            std::cerr << "n_colors=" << n_colors << "is not a divisor of lattice size w=" << n_sites << std::endl;
            std::exit(EXIT_FAILURE);
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            colors[i] = i % n_colors;
        }
        return colors_to_groups(colors);
    }
};

std::unique_ptr<SimpleModel> SimpleModel::mk_linear(int w) {
    return std::make_unique<LinearModel>(w);
}


class SquareModel: public SimpleModel {
public:
    int w, h;
    
    SquareModel(int w, int h): SimpleModel(w*h), w(w), h(h) {
    }
    
    vec3 dimensions() {
        return {double(w), double(h), 1};
    }
    
    vec3 position(int i) {
        assert(0 <= i && i < w*h);
        double x = i % w;
        double y = i / w;
        return {x, y, 0};
    }
    
    void set_spins(std::string const& name, const toml_ptr params, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites, {0, 0, 1});
        }
        else if (name == "meron") {
            set_spins_meron(toml_get<double>(params, "a"), toml_get<int64_t>(params, "q"), spin);
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
            std::exit(EXIT_FAILURE);
        }
    }
    
    void set_spins_meron(double a, int q, Vec<vec3>& spin) {
        assert(w % h == 0); // need periodicity in both dimensions
        double b = sqrt(1.0-(a*a));
        double factor, q1[2], q2[2];//k-points
        double q1_phase, q2_phase;
        if ((1.0-(a*a))<1e-4){// in case (1-a*a) = -0.000 at a = 1
            b = 0.0;
            a = 1.;
        }
        factor = 2.*Pi*q/h;
        q1[0] = factor;
        q1[1] = factor;
        q2[0] = factor;
        q2[1] =-factor;
        // printf("meron params: %10lf, %10lf, %10lf, %10lf\n", factor, a, b, 1.0-(a*a));
        for (int i = 0; i < n_sites; i++) {
            int x = i % w;
            int y = i / h;
            q1_phase = q1[0]*x + q1[1]*y;
            q2_phase = q2[0]*x + q2[1]*y;
            spin[i].x =sqrt(a*a + b*b*cos(q2_phase)*cos(q2_phase))*cos(q1_phase);
            spin[i].y =sqrt(a*a + b*b*cos(q2_phase)*cos(q2_phase))*sin(q1_phase);
            spin[i].z =b*sin(q2_phase);
        }
    }
    
    void set_neighbors(int rank, int i, Vec<int>& idx) {
        struct Delta {int x; int y;};
        static Vec<Vec<Delta>> deltas {
            { {1, 0}, {0, 1}, {-1, 0}, {0, -1} },
            { {1, 1}, {-1, 1}, {-1, -1}, {1, -1} },
            { {2, 0}, {0, 2}, {-2, 0}, {0, -2} },
        };
        assert(0 <= rank && rank < deltas.size());
        auto d = deltas[rank];
        
        int x = i % w;
        int y = i / w;
        idx.resize(d.size());
        for (int n = 0; n < idx.size(); n++) {
            int xp = positive_mod(x+d[n].x, w);
            int yp = positive_mod(y+d[n].y, h);
            idx[n] = xp + yp*w;
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        int c_len = int(sqrt(n_colors));
        if (c_len*c_len != n_colors) {
            std::cerr << "n_colors=" << n_colors << " is not a perfect square\n";
            std::exit(EXIT_FAILURE);
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            std::exit(EXIT_FAILURE);
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            int x = i % w;
            int y = i / h;
            int cx = x % c_len;
            int cy = y % c_len;
            colors[i] = cy*c_len + cx;
        }
        return colors_to_groups(colors);
    }
};
std::unique_ptr<SimpleModel> SimpleModel::mk_square(int w, int h) {
    return std::make_unique<SquareModel>(w, h);
}


// TODO: change this to +1 for consistency with Kagome
#define TRIANGULAR_SHEAR_DIRECTION (-1)

class TriangularModel: public SimpleModel {
public:
    int w, h;
    
    TriangularModel(int w, int h): SimpleModel(w*h), w(w), h(h) {
    }
    
    vec3 dimensions() {
        return {double(w), 0.5*sqrt(3.0)*h, 1};
    }
    
    vec3 position(int i) {
        assert(0 <= i && i < w*h);
        double x = i % w;
        double y = i / w;
        double a = 1.0;                // horizontal distance between columns
        double b = 0.5*sqrt(3.0)*a;    // vertical distance between rows
        return {a*x + TRIANGULAR_SHEAR_DIRECTION * 0.5*a*y, b*y, 0};
    }
    
    void pbc_shear(double& xy, double& xz, double& yz) {
        xy = TRIANGULAR_SHEAR_DIRECTION * 1.0/sqrt(3.0);
        xz = yz = 0.0;
    }
    
    
    void set_spins_3q(Vec<vec3> b, Vec<vec3>& spin) {
        for (int i = 0; i < n_sites; i++) {
            int x = i%w;
            int y = i/w;
            if (x%2 == 0 && y%2 == 0) {
                spin[i] =  b[0] + b[1] + b[2];
            } else if (x%2 == 1 && y%2 == 0) {
                spin[i] =  b[0] - b[1] - b[2];
            } else if (x%2 == 0 && y%2 == 1) {
                spin[i] = -b[0] + b[1] - b[2];
            } else if (x%2 == 1 && y%2 == 1) {
                spin[i] = -b[0] - b[1] + b[2];
            }
        }
    }
    
    void set_spins_hexagonal_vortices() {
        vec3 Q1 { 4.0*Pi/7.0, -8.0*Pi/(7.0*sqrt(3.0)), 0.0};
        vec3 Q2 {-6.0*Pi/7.0, -2.0*Pi/(7.0*sqrt(3.0)), 0.0};
        vec3 Q3 { 2.0*Pi/7.0, 10.0*Pi/(7.0*sqrt(3.0)), 0.0};
        
        double d1 = 0.60162300894888405;
        double d2 = 0.45768802314662158;
        vec3 Delta1 = 1.0 * vec3{ d1,                                d2, 1e-6};
        vec3 Delta2 = 0.5 * vec3{-d1 + sqrt(3.0)*d2, -sqrt(3.0)*d1 - d2, 1e-6};
        vec3 Delta3 = 0.5 * vec3{-d1 - sqrt(3.0)*d2,  sqrt(3.0)*d1 - d2, 1e-6};
        
        for (int i = 0; i < n_sites; i++) {
            vec3 x = position(i);
            spin[i] = Delta1 * sin(Q1.dot(x)) + Delta2 * sin(Q2.dot(x)) + Delta3 * sin(Q3.dot(x));
        }
    }
    
    void set_spins_collinear_stripes() {
        vec3 Q {Pi, 0, 0};
        
        for (int i = 0; i < n_sites; i++) {
            vec3 x = position(i);
            spin[i] = {0, 0, sin(Q.dot(x))+cos(Q.dot(x))};
        }
    }
    
    void set_spins(std::string const& name, const toml_ptr params, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites, {0, 0, 1});
        } else if (name == "allout") {
            double c = 1.0/sqrt(3.0);
            set_spins_3q({{c, 0, 0}, {0, c, 0}, {0, 0, c}}, spin);
        } else if (name == "3q_collinear") {
            set_spins_3q({{0, 0, 1}, {0, 0, 1}, {0, 0, 1}}, spin);
        } else if (name == "hexagonal_vortices") {
            set_spins_hexagonal_vortices();
        } else if (name == "collinear_stripes") {
            set_spins_collinear_stripes();
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
            std::exit(EXIT_FAILURE);
        }
    }
    
    void set_neighbors(int rank, int i, Vec<int>& idx) {
        struct Delta {int x; int y;};
        static Vec<Vec<Delta>> deltas {
#if TRIANGULAR_SHEAR_DIRECTION == -1
            // . . C 1 B
            // . 2 c b 0
            // D d * a A
            // 3 e f 5 .
            // E 4 F . .
            { {1, 0}, {1, 1}, {0,  1}, {-1,  0}, {-1, -1}, {0, -1} }, // a b c d e f
            { {2, 1}, {1, 2}, {-1, 1}, {-2, -1}, {-1, -2}, {1, -1} }, // 0 1 2 3 4 5
            { {2, 0}, {2, 2}, {0,  2}, {-2,  0}, {-2, -2}, {0, -2} }, // A B C D E F
#else
            // C 1 B . .
            // 2 c b 0 .
            // D d * a A
            // . 3 e f 5
            // . . E 4 F
            { {1, 0}, {0,  1}, {-1, 1}, {-1,  0}, {0, -1}, {1, -1} }, // a b c d e f
            { {1, 1}, {-1, 2}, {-2, 1}, {-1, -1}, {1, -2}, {2, -1} }, // 0 1 2 3 4 5
            { {2, 0}, {0,  2}, {-2, 2}, {-2,  0}, {0, -2}, {2, -2} }, // A B C D E F
#endif
        };
        assert(0 <= rank && rank < deltas.size());
        auto d = deltas[rank];
        
        int x = i % w;
        int y = i / w;
        idx.resize(d.size());
        for (int n = 0; n < idx.size(); n++) {
            int xp = positive_mod(x+d[n].x, w);
            int yp = positive_mod(y+d[n].y, h);
            idx[n] = xp + yp*w;
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        int c_len = int(sqrt(n_colors));
        if (c_len*c_len != n_colors) {
            std::cerr << "n_colors=" << n_colors << " is not a perfect square\n";
            std::exit(EXIT_FAILURE);
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            std::exit(EXIT_FAILURE);
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            int x = i % w;
            int y = i / h;
            int cx = x % c_len;
            int cy = y % c_len;
            colors[i] = cy*c_len + cx;
        }
        return colors_to_groups(colors);
    }
};
std::unique_ptr<SimpleModel> SimpleModel::mk_triangular(int w, int h) {
    return std::make_unique<TriangularModel>(w, h);
}


class KagomeModel: public SimpleModel {
public:
    int w, h;
    
    KagomeModel(int w, int h): SimpleModel(3*w*h), w(w), h(h) {
    }
    
    vec3 dimensions() {
        return {2.0*w, sqrt(3.0)*h, 1};
    }
    
    //
    //         1         1         1
    //        /D\       /E\       /F\
    //   --- 2 - 0 --- 2 - 0 --- 2 - 0
    //   \ /       \ /       \ /
    //    1         1         1
    //   /A\       /B\       /C\
    //  2 - 0 --- 2 - 0 --- 2 - 0 ---
    //        \ /       \ /       \ /
    //
    vec3 position(int i) {
        int v = i%3;
        int x = (i/3)%w;
        int y = i/(3*w);
        double a = 2.0;                       // horizontal distance between letters (A <-> B)
        double b = 0.5*sqrt(3)*a;             // vertical distance between letters   (A <-> D)
        double r = 1 / (2*sqrt(3))*a;         // distance between letter and number  (A <-> 0)
        double theta = -Pi/6 + (2*Pi/3)*v;    // angle from letter to number
        return {a*x + 0.5*a*y + r*cos(theta), b*y + r*sin(theta), 0};
    }
    
    void pbc_shear(double& xy, double& xz, double& yz) {
        xy = 1.0/sqrt(3.0);
        xz = yz = 0.0;
    }
    
    void set_spins_3q(Vec<Vec<vec3>> b, Vec<vec3>& spin) {
        for (int i = 0; i < n_sites; i++) {
            int v = i%3;
            int x = (i/3)%w;
            int y = i/(3*w);

            vec3 s(0, 0, 0);
            if (x%2 == 0 && y%2 == 0) {
                s =  b[0][v] + b[1][v] + b[2][v];
            } else if (x%2 == 1 && y%2 == 0) {
                s =  b[0][v] - b[1][v] - b[2][v];
            } else if (x%2 == 0 && y%2 == 1) {
                s = -b[0][v] + b[1][v] - b[2][v];
            } else if (x%2 == 1 && y%2 == 1) {
                s = -b[0][v] - b[1][v] + b[2][v];
            }
            spin[i] = s.normalized();
        }
    }
    
    void set_spins(std::string const& name, const toml_ptr params, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites, {0, 0, 1});
        } else if (name == "ncp1") {
            // Chiral phase, 3/12, 7/12 fillings
            // Orthogonal sublattice vectors
            //        v0          v1          v2
            // b_i
            set_spins_3q({
                { {0, 0,  0}, {-1, 0, 0}, {0,  0, 0} },
                { {0, 0, -1}, { 0, 0, 0}, {0,  0, 0} },
                { {0, 0,  0}, { 0, 0, 0}, {0, -1, 0} }
            }, spin);
        } else if (name == "ncp2") {
            // Zero chirality on triangles, 5/12 filling
            set_spins_3q({
                { {-1, 0,  0}, {0,  0, 0}, {1, 0, 0} },
                { { 0, 0,  0}, {0, -1, 0}, {0, 1, 0} },
                { { 0, 0, -1}, {0,  0, 1}, {0, 0, 0} }
            }, spin);
        } else if (name == "ncp3") {
            // Vortex crystal, 1/12, 8/12 fillings
            set_spins_3q({
                { {1, 0, 0}, {0, 0, 0}, {1, 0, 0} },
                { {0, 0, 0}, {0, 1, 0}, {0, 1, 0} },
                { {0, 0, 1}, {0, 0, 1}, {0, 0, 0} }
            }, spin);
        } else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
            std::exit(EXIT_FAILURE);
        }
    }
    
    void set_neighbors(int rank, int k, Vec<int>& idx) {
        if (rank != 0) {
            std::cerr << "Only nearest neighbors supported on kagome lattice\n";
            std::exit(EXIT_FAILURE);
        }
        idx.resize(4);
        int v = k%3;
        int x = (k/3)%w;
        int y = k/(3*w);
        
        auto coord2idx = [&](int v, int x, int y) -> int {
            int xp = (x%w+w)%w;
            int yp = (y%h+h)%h;
            return v + xp*(3) + yp*(3*w);
        };
        
        if (v == 0) {
            //     1
            //    /D\       /E
            //   2 - 0 --- 2 -
            //         \ /
            //          1
            //         /B\.
            idx[0] = coord2idx(2, x+1, y+0);
            idx[1] = coord2idx(1, x+0, y+0);
            idx[2] = coord2idx(2, x+0, y+0);
            idx[3] = coord2idx(1, x+1, y-1);
        } else if (v == 1) {
            //     D\       /E
            //     - 0 --- 2 -
            //         \ /
            //          1
            //         /B\
            //     - 2 --- 0 -
            idx[0] = coord2idx(2, x+0, y+1);
            idx[1] = coord2idx(0, x-1, y+1);
            idx[2] = coord2idx(2, x+0, y+0);
            idx[3] = coord2idx(0, x+0, y+0);
        } else if (v == 2) {
            //              1
            //    D\       /E\
            //    - 0 --- 2 - 0 -
            //        \ /
            //         1
            //        /B\.
            idx[0] = coord2idx(0, x+0, y+0);
            idx[1] = coord2idx(1, x+0, y+0);
            idx[2] = coord2idx(0, x-1, y+0);
            idx[3] = coord2idx(1, x+0, y-1);
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        if (n_colors%3 != 0) {
            std::cerr << "n_colors=" << n_colors << " is not a multiple of 3\n";
            std::exit(EXIT_FAILURE);
        }
        int c_len = int(sqrt(n_colors/3));
        if (c_len*c_len != n_colors/3) {
            std::cerr << "n_colors/3=" << n_colors/3 << " is not a perfect square\n";
            std::exit(EXIT_FAILURE);
        }
        if (w % c_len != 0 || h % c_len != 0) {
            std::cerr << "sqrt(n_colors/3)=" << c_len << " is not a divisor of lattice size (w,h)=(" << w << "," << h << ")\n";
            std::exit(EXIT_FAILURE);
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            int v = i%3;
            int x = (i/3)%w;
            int y = i/(3*w);
            int cx = x%c_len;
            int cy = y%c_len;
            colors[i] = 3*(cy*c_len+cx) + v;
        }
        return colors_to_groups(colors);
    }
};
std::unique_ptr<SimpleModel> SimpleModel::mk_kagome(int w, int h) {
    return std::make_unique<KagomeModel>(w, h);
}


class CubicModel: public SimpleModel {
public:
    int lx, ly, lz;
    
    CubicModel(int lx, int ly, int lz): SimpleModel(lx*ly*lz), lx(lx), ly(ly), lz(lz) {
    }
    
    vec3 dimensions() {
        return {double(lx), double(ly), double(lz)};
    }
    
    vec3 position(int i) {
        assert(0 <= i && i < lx*ly*lz);
        double x = (i % lx);
        double y = (i / lx)%ly;
        double z = (i / lx)/ ly;
        return {x, y, z};
    }
    
    void set_spins(std::string const& name, const toml_ptr params, Vec<vec3>& spin) {
        if (name == "ferro") {
            spin.assign(n_sites, vec3{0, 0, 1});
        }
        else if (name == "antiferro") {
            assert(lx%2 == 0 && ly%2 == 0 && lz%2 == 0);
            for (int i = 0; i < n_sites; i++) {
                int x = i % lx;
                int y = (i/lx) % ly;
                int z = i/(lx*ly);
                if ((x+y+z)%2==0) {
                    spin[i] = {0.0, 0.0, 1.0};
                } else {
                    spin[i] = {0.0, 0.0, -1.0};
                }
            }
        }
        else {
            std::cerr << "Unknown configuration type `" << name << "`\n";
            std::exit(EXIT_FAILURE);
        }
    }
    
    void set_neighbors(int rank, int i, Vec<int>& idx) {
        struct Delta {int x; int y; int z;};
        static Vec<Vec<Delta>> deltas {
            { {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {-1, 0, 0}, {0, -1, 0}, {0, 0, -1} },
            { {1, 1, 0}, {-1, 1, 0}, {-1, -1, 0}, {1, -1, 0}, {1, 0, 1}, {-1, 0, 1}, {-1, 0, -1}, {1, 0, -1}, {0, 1, 1}, {0, -1, 1}, {0, -1, -1}, {0, 1, -1}},
            { {2, 0, 0}, {0, 2, 0}, {0, 0, 2}, {-2, 0, 0}, {0, -2, 0}, {0, 0, -2} },
        };
        assert(0 <= rank && rank < deltas.size());
        auto d = deltas[rank];
        
        int x = (i%lx);
        int y = (i/lx)%ly;
        int z = (i/lx)/ly;
        idx.resize(d.size());
        for (int n = 0; n < idx.size(); n++) {
            int xp = positive_mod(x+d[n].x, lx);
            int yp = positive_mod(y+d[n].y, ly);
            int zp = positive_mod(z+d[n].z, lz);
            idx[n] = (zp*ly + yp)*lx + xp;
        }
    }
    
    Vec<int> groups(int n_colors) {
        n_colors = std::min(n_colors, n_sites);
        int c_len = int(std::cbrt(n_colors));
        if (c_len*c_len*c_len != n_colors) {
            std::cerr << "n_colors=" << n_colors << " is not a perfect cube\n";
            std::exit(EXIT_FAILURE);
        }

        if (lx % c_len != 0 || ly % c_len != 0 || lz % c_len != 0) {
            std::cerr << "cbrt(n_colors)=" << c_len << " is not a divisor of lattice size (lx,ly,lz)=(" << lx << "," << ly << "," << lz << ")\n";
            std::exit(EXIT_FAILURE);
        }
        Vec<int> colors(n_sites);
        for (int i = 0; i < n_sites; i++) {
            int x = i % lx;
            int y = (i/lx) % ly;
            int z = i/(lx*ly);
            int cx = x % c_len;
            int cy = y % c_len;
            int cz = z % c_len;
            colors[i] = cz*c_len*c_len + cy*c_len + cx;
        }
        return colors_to_groups(colors);

    }
};
std::unique_ptr<SimpleModel> SimpleModel::mk_cubic(int w, int h, int h_z) {
    return std::make_unique<CubicModel>(w, h, h_z);
}


