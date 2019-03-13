#include "kondo.h"
#include <cassert>


// project vector p onto plane that is normal to x
vec3 project_tangent(vec3 x, vec3 p) {
    return p - x * (p.dot(x) / x.norm2());
}

vec3 gaussian_vec3(fkpm::RNG& rng) {
    static std::normal_distribution<double> dist;
    return { dist(rng), dist(rng), dist(rng) };
}


class Overdamped: public Dynamics {
public:
    Overdamped(double dt) { this->dt = dt; }
    
    void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& f = m.dyn_stor[0];
        calc_force(m.spin, f);
        for (int i = 0; i < m.n_sites; i++) {
            vec3 beta = sqrt(dt*2*m.kT()) * gaussian_vec3(rng);
            m.spin[i] += project_tangent(m.spin[i], dt*f[i]+beta);
            m.spin[i] = m.spin[i].normalized();
        }
        
        n_steps++;
        m.time = n_steps * dt;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_overdamped(double dt) {
    return std::make_unique<Overdamped>(dt);
}


class SLL: public Dynamics {
public:
    double alpha;
    
    SLL(double alpha, double dt): alpha(alpha) { this->dt = dt; }
    
    void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& s    = m.spin;
        Vec<vec3>& sp   = m.dyn_stor[0];
        Vec<vec3>& spp  = m.dyn_stor[1];
        Vec<vec3>& f    = m.dyn_stor[2];
        Vec<vec3>& fp   = m.dyn_stor[3];
        Vec<vec3>& beta = m.dyn_stor[4];
        
        double D = (alpha / (1 + alpha*alpha)) * m.kT();
        for (int i = 0; i < m.n_sites; i++) {
            beta[i] = sqrt(dt*2*D) * gaussian_vec3(rng);
        }
        
        // one euler step starting from s (with force f), accumulated into sp
        // TODO: scale D terms by 1/|s_i| ?
        // See appendix in Skubic et al., J. Phys.: Condens. Matter 20, 315203 (2008)
        auto accum_euler = [&](Vec<vec3> const& s, Vec<vec3> const& f, double scale, Vec<vec3>& sp) {
            for (int i = 0; i < m.n_sites; i++) {
                vec3 a     = - f[i]    - alpha*s[i].cross(f[i]);
                vec3 sigma = - beta[i] - alpha*s[i].cross(beta[i]);
                sp[i] += scale * s[i].cross(a*dt + sigma);
            }
        };
        
        calc_force(s, f);
        sp = s;
        accum_euler(s, f, 1.0, sp);
        calc_force(sp, fp);
        spp = s;
        accum_euler(s, f,  0.5, spp);
        accum_euler(sp, fp, 0.5, spp);
        
        // copy s = spp, but ensure norm is unchanged (by discretization error)
        for (int i = 0; i < m.n_sites; i++) {
            if (spp[i].norm() == 0.0) {
                assert(s[i].norm() == 0.0);
                // no update to s required
            }
            else {
                s[i] = spp[i].normalized() * s[i].norm();
            }
        }
        
        n_steps++;
        m.time = n_steps * dt;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_sll(double alpha, double dt) {
    return std::make_unique<SLL>(alpha, dt);
}


class SLL_SIB: public Dynamics {
public:
    double alpha;
    
    SLL_SIB(double alpha, double dt): alpha(alpha) { this->dt = dt; }
    
    void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& s    = m.spin;
        Vec<vec3>& sp   = m.dyn_stor[0];
        Vec<vec3>& spp  = m.dyn_stor[1];
        Vec<vec3>& f    = m.dyn_stor[2];
        Vec<vec3>& fp   = m.dyn_stor[3];
        Vec<vec3>& beta = m.dyn_stor[4];
        
        double D = (alpha / (1 + alpha*alpha)) * m.kT();
        for (int i = 0; i < m.n_sites; i++) {
            beta[i] = sqrt(dt*2*D) * gaussian_vec3(rng);
        }
        
        // one semi-implicit step starting with s (with force f[s]), written to sp
        // TODO: scale D terms by 1/|s_i| ?
        // See appendix in Skubic et al., J. Phys.: Condens. Matter 20, 315203 (2008)
        auto implicit_solve = [&](Vec<vec3> const& s, Vec<vec3> const& f, Vec<vec3>& sp) {
            for (int i = 0; i < m.n_sites; i++) {
                vec3 w = 0.5 * (- dt*f[i] - alpha*s[i].cross(dt*f[i]) +
                                - beta[i] - alpha*s[i].cross(beta[i]));
                vec3 rhs = s[i] + s[i].cross(w);
                double denom = 1 + w.norm2();
                sp[i].x = vec3(1+w.x*w.x,   w.x*w.y+w.z, w.x*w.z-w.y).dot(rhs) / denom;
                sp[i].y = vec3(w.x*w.y-w.z, 1+w.y*w.y,   w.y*w.z+w.x).dot(rhs) / denom;
                sp[i].z = vec3(w.x*w.z+w.y, w.y*w.z-w.x, 1+w.z*w.z  ).dot(rhs) / denom;
                // check target equation: sp = s + (s + sp) cross w
                assert((sp[i] - s[i] - (s[i]+sp[i]).cross(w)).norm() < 1e-8);
                // a mathematical consequence is norm preservation
                assert(std::abs(sp[i].norm() - s[i].norm()) < 1e-8);
            }
        };
        
        calc_force(s, f);
        implicit_solve(s, f, sp);
        for (int i = 0; i < m.n_sites; i++) {
            sp[i] = 0.5 * (s[i] + sp[i]);
        }
        calc_force(sp, fp);
        implicit_solve(s, fp, spp);
        s = spp;
        
        n_steps++;
        m.time = n_steps * dt;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_sll_sib(double alpha, double dt) {
    return std::make_unique<SLL_SIB>(alpha, dt);
}


class GJF: public Dynamics {
public:
    double alpha;
    double a, b;
    double mass = 1;
    
    GJF(double alpha, double dt): alpha(alpha) {
        this->dt = dt;
        double denom = 1 + alpha*dt/(2*mass);
        a = (1 - alpha*dt/(2*mass))/denom;
        b = 1 / denom;
    }
    
    void init(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& v = m.dyn_stor[0];
        Vec<vec3>& f1 = m.dyn_stor[1];
        v.assign(m.n_sites, {0,0,0});
        calc_force(m.spin, f1);
    }
    
    void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& s = m.spin;
        Vec<vec3>& v = m.dyn_stor[0];
        Vec<vec3>& f1 = m.dyn_stor[1];
        Vec<vec3>& f2 = m.dyn_stor[2];
        Vec<vec3>& beta = m.dyn_stor[3];
        
        for (int i = 0; i < m.n_sites; i++) {
            beta[i] = sqrt(dt*2*alpha*m.kT()) * gaussian_vec3(rng);
            s[i] += b*dt*v[i] + (b*dt*dt/(2*mass))*f1[i] + (b*dt/(2*mass))*beta[i];
        }
        
        calc_force(s, f2);
        
        for (int i = 0; i < m.n_sites; i++) {
            v[i] = a*v[i] + (dt/(2*mass))*(a*f1[i] + f2[i]) + (b/mass)*beta[i];
            
            // forces will be reused in the next timestep
            f1[i] = f2[i];
        }
        
        n_steps++;
        m.time = n_steps * dt;
    }
    
    double pseudo_kinetic_energy(Model const& m) {
        Vec<vec3> const& v = m.dyn_stor[0];
        double acc = 0;
        for (int i = 0; i < m.n_sites; i++) {
            acc += 0.5 * mass * v[i].norm2();
        }
        return acc;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_gjf(double alpha, double dt) {
    return std::make_unique<GJF>(alpha, dt);
}


class GLSD: public Dynamics {
public:
    double alpha;
    
    GLSD(double alpha, double dt): alpha(alpha) { this->dt = dt; }
    
    void step(CalcForce const& calc_force, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& s    = m.spin;
        Vec<vec3>& sp   = m.dyn_stor[0];
        Vec<vec3>& spp  = m.dyn_stor[1];
        Vec<vec3>& f    = m.dyn_stor[2];
        Vec<vec3>& fp   = m.dyn_stor[3];
        Vec<vec3>& noise = m.dyn_stor[4];
        
        for (int i = 0; i < m.n_sites; i++) {
            noise[i] = sqrt(2*alpha*m.kT()) * gaussian_vec3(rng);
        }
        
        // one euler step starting from s (with force f), accumulated into sp
        auto accum_euler = [&](Vec<vec3> const& s, Vec<vec3> const& f, double scale, Vec<vec3>& sp) {
            for (int i = 0; i < m.n_sites; i++) {
                // magnitude conserving dynamics
                double sp_norm = sp[i].norm();
                sp[i] += scale*dt*s[i].cross(f[i]);
                
                // magnitude can shift by O(dt^2). manually rescale sp[i] for better accuracy
                // TODO: test empirically and justify mathematically?
                if (sp[i].norm() > 0.0) {
                    sp[i] = sp[i].normalized() * sp_norm;
                }
                
                // langevin dynamics
                sp[i] += scale * (dt*alpha*f[i] + sqrt(dt)*noise[i]);
            }
        };
        
        calc_force(s, f);
        sp = s;
        accum_euler(s, f, 1.0, sp);
        calc_force(sp, fp);
        spp = s;
        accum_euler(s, f,  0.5, spp);
        accum_euler(sp, fp, 0.5, spp);
        s = spp;
        
        n_steps++;
        m.time = n_steps * dt;
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_glsd(double alpha, double dt) {
    return std::make_unique<GLSD>(alpha, dt);
}


class Metropolis : public Dynamics {
private:
    std::uniform_real_distribution<double> dist;                                // [0,1]
    
    // site indices which are allowed to be updated
    std::vector<int> allow_update;
public:
    int n_substeps = 0;
    
    Metropolis(const Model& m) {
        assert(m.spin_exist.empty() || m.spin_exist.size() == m.n_sites);
        dist = std::uniform_real_distribution<double>(0.0, 1.0);
        allow_update.clear();
        for (int i = 0; i < m.n_sites; i++) {
            if (m.spin_exist.empty() || m.spin_exist[i]) allow_update.push_back(i);
        }
    }
    
    void step(CalcForce const& calc_Ediff, fkpm::RNG& rng, Model& m) {
        Vec<vec3>& s    = m.spin;
        Vec<vec3>& sp   = m.dyn_stor[0];
        sp = s;
        
        int N = static_cast<int>(allow_update.size());
        assert(N > 0 && N <= m.n_sites);
        std::uniform_int_distribution<int> dist_int(0,N-1);                     // {0,1,2,...,N-1}
        int site = allow_update[dist_int(rng)];
        assert(site >= 0 && site < m.n_sites);
        assert(m.spin_exist.empty() || m.spin_exist[site]);
        
        // generate a random spin in S^2 space, c.f. D. Landau's book
        double zeta1,zeta2;
        double zeta_square = 10.0;
        while (zeta_square >= 1.0) {
            zeta1 = 1.0 - 2.0 * dist(rng);
            zeta2 = 1.0 - 2.0 * dist(rng);
            zeta_square = zeta1 * zeta1 + zeta2 * zeta2;
        }
        double zeta_sqrt = 2.0 * std::sqrt(1.0 - zeta_square);
        vec3 s_new{zeta1 * zeta_sqrt, zeta2 * zeta_sqrt, (1.0 - 2.0 * zeta_square)};
        sp[site] = s_new.normalized() * s[site].norm();
        
        double deltaE = calc_Ediff(s, sp);
        using std::swap;
        if (deltaE <= 0.0) {                                                    // always accept update
            swap(s, sp);
        } else if (dist(rng) < std::exp(-deltaE / m.kT())) {                    // probability: exp(-deltaE/T)
            swap(s, sp);
        }
        
        n_substeps++;
        if (n_substeps >= N) {
            n_substeps = 0;
            n_steps++;
        }
        m.time = static_cast<double>(n_steps);
    }
};
std::unique_ptr<Dynamics> Dynamics::mk_metropolis(const Model& m) {
    return std::make_unique<Metropolis>(m);
}
