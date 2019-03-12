#include "kondo.h"
#include "iostream_util.h"
#include <cassert>


MostovoyModel::MostovoyModel(int lx, int ly, int lz): Model(lx*ly*lz, 10), lx(lx), ly(ly), lz(lz) {
}

int MostovoyModel::d_idx(int i, int alpha, int sigma) {
    return n_orbs*i + 0 + 2*alpha + sigma;
}

int MostovoyModel::p_idx(int i, int b, int sigma) {
    // b         0      1      2      3      4      5
    // offset (+x/2) (+y/2) (+z/2) (-x/2) (-y/2) (-z/2)
    assert(0 <= b && b < 6);
    if (3 <= b && b < 6) {
        int x = i % lx;
        int y = (i/lx) % ly;
        int z = i/(lx*ly);
        if (b == 3)
            x = (x - 1 + lx) % lx;
        else if (b == 4)
            y = (y - 1 + ly) % ly;
        else if (b == 5)
            z = (z - 1 + lz) % lz;
        i = x + y*lx + z*lx*ly;
        b -= 3;
    }
    return n_orbs*i + 4 + 2*b + sigma;
}


void MostovoyModel::set_hamiltonian(Vec<vec3> const& spin, Vec<bool> const& spin_inactive) {
    assert(spin_inactive.empty());
    H_elems.clear();
    D_elems.clear();
    
    for (int i = 0; i < n_sites; i++) {
        
        // Hund coupling
        cx_flt zero_cx  = 0;
        for (int a = 0; a < 2; a++) {
            for (int s1 = 0; s1 < 2; s1++) {
                for (int s2 = 0; s2 < 2; s2++) {
                    cx_flt v = flt(J) * (pauli[s1][s2].dot(spin[i]) + flt(s1 == s2 ? 1.0 : 0.0));
                    H_elems.add(d_idx(i, a, s1), d_idx(i, a, s2), &v);
                    D_elems.add(d_idx(i, a, s1), d_idx(i, a, s2), &zero_cx);
                }
            }
        }
    
        // On-site p term
        cx_flt delta_cx = delta;
        for (int b = 0; b < 3; b++) {
            for (int s = 0; s < 2; s++) {
                H_elems.add(p_idx(i, b, s), p_idx(i, b, s), &delta_cx);
            }
        }
        
        // d-p hoppings
        static double t_alpha_b[2][3] = {
            {-0.5, -0.5, 1},
            {sqrt(3)/2, -sqrt(3)/2, 0}
        };
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 6; b++) {
                cx_flt t = t_pds * t_alpha_b[a][b % 3];
                for (int s = 0; s < 2; s++) {
                    H_elems.add(d_idx(i, a, s), p_idx(i, b, s), &t);
                    H_elems.add(p_idx(i, b, s), d_idx(i, a, s), &t);
                }
            }
        }
        
        // p-p hopping
        cx_flt t_pp_cx  = t_pp;
        for (int b = 0; b < 6; b++) {
            for (int c = 0; c < 6; c++) {
                if (b % 3 == c % 3) continue;
                for (int s = 0; s < 2; s++) {
                    H_elems.add(p_idx(i, b, s), p_idx(i, c, s), &t_pp_cx);
                }
            }
        }
    }
    
    H.build(H_elems);
    D.build(D_elems);
}

void MostovoyModel::set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force) {
    // Classical forces
    Model::set_forces(D, spin, force);
    
    // Hund-coupling forces
    for (int i = 0; i < n_sites; i++) {
        Vec3<cx_flt> dE_dS(0, 0, 0);
        for (int a = 0; a < 2; a++) {
            for (int s1 = 0; s1 < 2; s1 ++) {
                for (int s2 = 0; s2 < 2; s2 ++) {
                    // Apply chain rule: dE/dS = dH_ij/dS D_ji
                    // where D_ij = dE/dH_ji is the density matrix
                    Vec3<cx_flt> dH_ij_dS = J * pauli[s1][s2];
                    cx_flt D_ji = *D(d_idx(i, a, s2), d_idx(i, a, s1));
                    dE_dS += dH_ij_dS * D_ji;
                }
            }
        }
        assert(imag(dE_dS).norm() < 1e-5);
        force[i] += -real(dE_dS);
    }
}

fkpm::SpMatBsr<cx_flt> MostovoyModel::electric_current_operator(Vec<vec3> const& spin, vec3 dir) {
    std::cerr << "electric_current_operator() not implemented for MostovoyModel\n";
    std::exit(EXIT_FAILURE);
}


void MostovoyModel::set_spins_helical(int qx, int qy, int qz, Vec<vec3>& spin) {
    vec3 q = 2 * Pi * vec3{double(qx)/lx, double(qy)/ly, double(qz)/lz};
    if (qx < 0 || lx/2 < qx ||
        qy < 0 || ly/2 < qy ||
        qz < 0 || lz/2 < qz) {
        std::cerr << "Component of q=" << q << " is out of bounds [0,Pi]\n";
        std::exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < n_sites; i++) {
        vec3 p = position(i);
        spin[i].x = cos(q.dot(p));
        spin[i].y = sin(q.dot(p));
        spin[i].z = 0;
    }
}

void MostovoyModel::set_spins(std::string const& name, const toml_ptr params, Vec<vec3>& spin) {
    if (name == "ferro") {
        spin.assign(n_sites, vec3{0, 0, 1});
    }
    else if (name == "helical") {
        set_spins_helical(toml_get<int64_t>(params, "qx", 0), toml_get<int64_t>(params, "qy", 0), toml_get<int64_t>(params, "qz", 0), spin);
    }
    else {
        std::cerr << "Unknown configuration type `" << name << "`\n";
        std::exit(EXIT_FAILURE);
    }
}


void MostovoyModel::set_neighbors(int rank, int i, Vec<int>& idx) {
    struct Delta {int x; int y; int z;};
    static Vec<Vec<Delta>> deltas {
        { {1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1} },
    };
    assert(0 <= rank && rank < deltas.size());
    auto d = deltas[rank];
    
    int x = i % lx;
    int y = (i/lx) % ly;
    int z = i/(lx*ly);
    
    idx.resize(d.size());
    for (int n = 0; n < idx.size(); n++) {
        int xp = positive_mod(x+d[n].x, lx);
        int yp = positive_mod(y+d[n].y, ly);
        int zp = positive_mod(z+d[n].z, lz);
        idx[n] = (zp*ly + yp)*lx + xp;
    }
}

vec3 MostovoyModel::dimensions() {
    return {double(lx), double(ly), double(lz)};
}

vec3 MostovoyModel::position(int i) {
    double x = i % lx;
    double y = (i/lx) % ly;
    double z = i/(lx*ly);
    return {x, y, z};
}

Vec<int> MostovoyModel::groups(int n_colors) {
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
