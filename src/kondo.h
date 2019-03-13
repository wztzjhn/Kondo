#ifndef __kondo__
#define __kondo__

#include "fastkpm.h"
#include "vec3.h"


typedef float flt;
// typedef double flt;
typedef std::complex<flt> cx_flt;

using fkpm::Vec;
using fkpm::Pi;
const cx_flt I(0, 1);

// {s1, s2} components of pauli matrix vector,
// sigma1     sigma2     sigma3
//  0  1       0 -I       1  0
//  1  0       I  0       0 -1
const Vec3<cx_flt> pauli[2][2] {
    {{0, 0, 1}, {1, -I, 0}},
    {{1, I, 0}, {0, 0, -1}}
};

inline int positive_mod(int i, int n) {
    return (i%n + n) % n;
}


// C++14 feature missing in C++11
namespace std {
    template<typename T, typename ...Args>
    std::unique_ptr<T> make_unique( Args&& ...args ) {
        return std::unique_ptr<T>( new T( std::forward<Args>(args)... ) );
    }
}

// TOML utilities
namespace cpptoml { class table; }
typedef std::shared_ptr<cpptoml::table> toml_ptr;
toml_ptr toml_from_str(std::string const& str);
toml_ptr toml_from_file(std::string const& filename);
template <class T> T toml_get(const toml_ptr table, const std::string& key);
template <class T> T toml_get(const toml_ptr table, const std::string& key, T default_value);


class Model {
protected:
    Vec<int> colors_to_groups(Vec<int> const& colors);

public:
    int n_sites; // Number of classical spins
    int n_orbs;  // Number of rows in Hamilitonian
    double kT_init = 0, kT_decay = 0;
    vec3 zeeman = {0, 0, 0};        // Magnetic field zeeman coupling
    vec3 current = {0, 0, 0};       // Generates spin transfer torque
    double easy_z = 0;              // Easy axis anisotropy (z direction)
    double s0=0, s1=0, s2=0, s3=0;  // Exchange interactions
    fkpm::SpMatElems<cx_flt> H_elems, D_elems;
    fkpm::SpMatBsr<cx_flt> H, D;
    Vec<vec3> spin;
    Vec<bool> spin_exist;  // true if local spin exists on a given site; if empty, all sites occupied
    double time = 0;
    Vec<vec3> dyn_stor[5]; // used by Dynamics to store intermediate data between steps
    
    Model(int n_sites, int n_orbs);
    virtual ~Model() {}

    static void set_spins_random(fkpm::RNG& rng, Vec<vec3>& spin);
    double kT();
    
    virtual void set_hamiltonian(Vec<vec3> const& spin, Vec<bool> const& spin_exist_ = Vec<bool>()) = 0;
    virtual void set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force);
    virtual double energy_classical(Vec<vec3> const& spin);
    virtual fkpm::SpMatBsr<cx_flt> electric_current_operator(Vec<vec3> const& spin, vec3 dir) = 0;
    
    virtual void set_spins(std::string const& name, const toml_ptr params, Vec<vec3>& spin) = 0;
    virtual void set_neighbors(int rank, int k, Vec<int>& idx) = 0;
    virtual vec3 dimensions() = 0;
    virtual vec3 position(int i) = 0;
    virtual void pbc_shear(double& xy, double& xz, double& yz);
    virtual vec3 displacement(int i, int j);
    virtual Vec<int> groups(int n_colors) = 0;
};

class SimpleModel: public Model {
public:
    double J = 0;
    double t1=0, t2=0, t3=0;
    
    SimpleModel(int n_sites);
    
    void set_hamiltonian(Vec<vec3> const& spin, Vec<bool> const& spin_exist_ = Vec<bool>());
    void set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force);
    fkpm::SpMatBsr<cx_flt> electric_current_operator(Vec<vec3> const& spin, vec3 dir);
    
    // instantiations
    static std::unique_ptr<SimpleModel> mk_linear(int w);
    static std::unique_ptr<SimpleModel> mk_square(int w, int h);
    static std::unique_ptr<SimpleModel> mk_triangular(int w, int h);
    static std::unique_ptr<SimpleModel> mk_kagome(int w, int h);
    static std::unique_ptr<SimpleModel> mk_cubic(int lx, int ly, int lz);
};

class MostovoyModel: public Model {
private:
    int d_idx(int i, int alpha, int sigma);
    int p_idx(int i, int b, int sigma);
    
public:
    int lx, ly, lz;
    double J = 0, t_pds = 0, t_pp = 0, delta = 0;
    
    MostovoyModel(int lx, int ly, int lz);
    
    void set_hamiltonian(Vec<vec3> const& spin, Vec<bool> const& spin_exist_ = Vec<bool>());
    void set_forces(fkpm::SpMatBsr<cx_flt> const& D, Vec<vec3> const& spin, Vec<vec3>& force);
    fkpm::SpMatBsr<cx_flt> electric_current_operator(Vec<vec3> const& spin, vec3 dir);
    
    void set_spins_helical(int qx, int qy, int qz, Vec<vec3>& spin);
    void set_spins(std::string const& name, const toml_ptr params, Vec<vec3>& spin);
    void set_neighbors(int rank, int k, Vec<int>& idx);
    vec3 dimensions();
    vec3 position(int i);
    Vec<int> groups(int n_colors);
};

class Dynamics {
public:
    typedef std::function<double(Vec<vec3> const& spin, Vec<vec3>& force)> CalcForce;
    int n_steps = 0;
    double dt;
    
    virtual ~Dynamics() {}
    
    // Overdamped relaxation using Euler integration, with constrained spin magnitude
    static std::unique_ptr<Dynamics> mk_overdamped(double dt);
    
    // Stochastic Landau Lifshitz using Heun-p (Mentink et al., 2010)
    static std::unique_ptr<Dynamics> mk_sll(double alpha, double dt);
    
    // Stochastic Landau Lifshitz using SIB (Mentink et al., 2010)
    // Compared to Heun-p, the SIB dynamics:
    //   - Perfectly conserves spin magnitude
    //   - Is more accurate when alpha is small
    //   - Appears less accurate when alpha is order 1
    static std::unique_ptr<Dynamics> mk_sll_sib(double alpha, double dt);
    
    // Inertial Langevin dynamics using Grønbech-Jensen Farago, velocity explicit method
    //   N. Gr{\o}nbech-Jensen, O. Farago, Mol. Phys. 111, 983--991 (2013)
    // Spin magnitude is *not* conserved
    static std::unique_ptr<Dynamics> mk_gjf(double alpha, double dt);
    
    // Generalized Langevin Spin Dynamics (GLSD) using Heun integration
    //   P.-W. Ma and S. L. Dudarev, Phys. Rev. B 86, 054416 (2012)
    // Spin magnitude is *not* conserved
    static std::unique_ptr<Dynamics> mk_glsd(double alpha, double dt);
    
    // Simple Metropolis update
    static std::unique_ptr<Dynamics> mk_metropolis(const Model& m);
    
    virtual void init(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {}
    virtual void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) = 0;
    virtual double pseudo_kinetic_energy(Model const& m) { return 0; }
};


#endif /* defined(__kondo__) */
