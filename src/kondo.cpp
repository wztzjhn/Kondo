#include "kondo.h"
#include "iostream_util.h"

#include <algorithm>
#include <climits>
#include <cstdint>
#include <sstream>
#include <regex>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>


// workaround missing "is_trivially_copyable" in g++ < 5.0
#if __GNUG__ && __GNUC__ < 5
#define IS_TRIVIALLY_COPYABLE(T) __has_trivial_copy(T)
#else
#define IS_TRIVIALLY_COPYABLE(T) std::is_trivially_copyable<T>::value
#endif
static const std::string nibble2hex = "0123456789ABCDEF";
int hex2nibble(char c) {
    for (int i = 0; i < 16; i++) {
        if (c == nibble2hex[i]) return i;
    }
    assert(false);
}
template <typename T, class=typename std::enable_if<IS_TRIVIALLY_COPYABLE(T)>::type>
void serialize_to_hex(T const& data, std::ostream &os) {
    uint8_t *bytes = (uint8_t *)&data;
    for (size_t i = 0; i < sizeof(T)*CHAR_BIT/8; i++) {
        os << nibble2hex[(bytes[i] >> 4) & 0xF];
        os << nibble2hex[bytes[i] & 0xF];
    }
}
template <typename T, class=typename std::enable_if<IS_TRIVIALLY_COPYABLE(T)>::type>
void deserialize_from_hex(std::istream& is, T& data) {
    uint8_t *bytes = (uint8_t *)&data;
    for (int i = 0; i < sizeof(T)*CHAR_BIT/8; i++) {
        char c1, c2;
        is >> c1 >> c2;
        bytes[i] = (hex2nibble(c1) << 4) | hex2nibble(c2);
    }
}

void randomize_spin_exist(Vec<bool> &spin_exist, int num_sites, int num_spins, unsigned seed) {
    assert(num_spins > 0 && num_spins <= num_sites);
    if (num_sites == num_spins) {
        spin_exist.clear();
    } else {
        spin_exist.assign(num_sites, false);
        std::vector<int> sites_active(num_sites);
        for (int i = 0; i < num_sites; i++) sites_active[i] = i;
        std::shuffle(sites_active.begin(), sites_active.end(), std::mt19937(seed));
        std::cout << "Sites for dilute spins: ";
        for (int i = 0; i < num_spins; i++) {
            spin_exist[sites_active[i]] = true;
            std::cout << sites_active[i] << ",";
        }
        std::cout << std::endl;
    }
}

std::unique_ptr<Model> mk_model(const toml_ptr g) {
    std::unique_ptr<Model> ret;
    auto type = toml_get<std::string>(g, "model.type");
    if (type == "simple") {
        auto lattice = toml_get<std::string>(g, "model.lattice");
        std::unique_ptr<SimpleModel> m;
        if (lattice == "linear") {
            m = SimpleModel::mk_linear(toml_get<int64_t>(g, "model.w"));
        } else if (lattice == "square") {
            m = SimpleModel::mk_square(toml_get<int64_t>(g, "model.w"), toml_get<int64_t>(g, "model.h"));
        } else if (lattice == "triangular") {
            m = SimpleModel::mk_triangular(toml_get<int64_t>(g, "model.w"), toml_get<int64_t>(g, "model.h"));
        } else if (lattice == "kagome") {
            m = SimpleModel::mk_kagome(toml_get<int64_t>(g, "model.w"), toml_get<int64_t>(g, "model.h"));
        } else if (lattice == "cubic") {
            m = SimpleModel::mk_cubic(toml_get<int64_t>(g, "model.lx"), toml_get<int64_t>(g, "model.ly"), toml_get<int64_t>(g, "model.lz"));
        } else {
            std::cerr << "Simple model lattice '" << lattice << "' not supported.\n";
            std::exit(EXIT_FAILURE);
        }
        m->J  = toml_get<double>(g, "model.J");
        m->t1 = toml_get<double>(g, "model.t1", 0);
        m->t2 = toml_get<double>(g, "model.t2", 0);
        m->t3 = toml_get<double>(g, "model.t3", 0);
        ret = std::move(m);
    } else if (type == "mostovoy") {
        auto lattice = toml_get<std::string>(g, "model.lattice");
        if (lattice != "cubic") {
            std::cerr << "Mostovoy model requires `lattice = \"cubic\"`\n";
            std::exit(EXIT_FAILURE);
        }
        auto m = std::make_unique<MostovoyModel>(toml_get<int64_t>(g, "model.lx"),
                                                 toml_get<int64_t>(g, "model.ly"),
                                                 toml_get<int64_t>(g, "model.lz"));
        m->J     = toml_get<double>(g, "model.J");
        m->t_pds = toml_get<double>(g, "model.t_pds");
        m->t_pp  = toml_get<double>(g, "model.t_pp");
        m->delta = toml_get<double>(g, "model.delta");
        ret = std::move(m);
    } else {
        std::cerr << "Model type '" << type << "' not supported.\n";
        std::exit(EXIT_FAILURE);
    }
    ret->kT_init  = toml_get<double>(g, "model.kT");
    ret->kT_decay = toml_get<double>(g, "model.kT_decay", 0);
    ret->zeeman   = {toml_get<double>(g, "model.zeeman_x",  0), toml_get<double>(g, "model.zeeman_y",  0), toml_get<double>(g, "model.zeeman_z",  0)};
    ret->current  = {toml_get<double>(g, "model.current_x", 0), toml_get<double>(g, "model.current_y", 0), toml_get<double>(g, "model.current_z", 0)};
    ret->easy_z   = toml_get<double>(g, "model.easy_z", 0);
    ret->s0       = toml_get<double>(g, "model.s0", 0);
    ret->s1       = toml_get<double>(g, "model.s1", 0);
    ret->s2       = toml_get<double>(g, "model.s2", 0);
    ret->s3       = toml_get<double>(g, "model.s3", 0);
    return ret;
}

std::unique_ptr<Dynamics> mk_dynamics(const toml_ptr g) {
    double dt = toml_get<double>(g, "dynamics.dt");
    auto type = toml_get<std::string>(g, "dynamics.type");
    if (type == "overdamped") {
        return Dynamics::mk_overdamped(dt);
    } else if (type == "sll") {
        return Dynamics::mk_sll(toml_get<double>(g, "dynamics.alpha"), dt);
    } else if (type == "sll_sib") {
        return Dynamics::mk_sll_sib(toml_get<double>(g, "dynamics.alpha"), dt);
    } else if (type == "gjf") {
        return Dynamics::mk_gjf(toml_get<double>(g, "dynamics.alpha"), dt);
    } else if (type == "glsd") {
        return Dynamics::mk_glsd(toml_get<double>(g, "dynamics.alpha"), dt);
    }
    
    cerr << "Unsupported dynamics type `" << type << "`!\n";
    std::exit(EXIT_FAILURE);
}

void read_state_from_dump(boost::filesystem::path const& dump_dir, fkpm::RNG &rng, Model &m, Dynamics &dynamics) {
    Vec<boost::filesystem::path> v;
    boost::filesystem::path p(dump_dir);
    for (auto &q : boost::filesystem::directory_iterator(p)) {
        if (std::regex_match(q.path().filename().string(), std::regex("dump[[:digit:]]+\\.json"))) v.push_back(q);
    }
    std::sort(v.begin(), v.end());
    boost::filesystem::ifstream is(v.back());
    boost::property_tree::ptree pt;
    boost::property_tree::read_json (is, pt);
    
    std::cout << "Resuming simulation from " << v.back() << ".\n";
    
    // Read random state
    std::string rng_state = pt.get<std::string>("rng_state");
    std::stringstream ss(rng_state);
    deserialize_from_hex(ss, rng);
    
    // Read spin configuration
    Vec<double> data;
    for (auto& elem : pt.get_child("spin")) {
        assert(elem.first.empty()); // array elements have no names
        data.push_back(std::stod(elem.second.data()));
    }
    assert(data.size() == 3*m.n_sites);
    for (int i = 0; i < m.n_sites; i++) {
        m.spin[i] = {data[3*i+0], data[3*i+1], data[3*i+2]};
    }
    
    // Read n_steps
    dynamics.n_steps = pt.get<size_t>("n_steps");
    
    // TODO: load velocity in GJ-F dynamics... ?
}

int main(int argc, char *argv[]) {
    auto engine = fkpm::mk_engine_mpi<cx_flt>();
    if (engine == nullptr) std::exit(EXIT_FAILURE);
    
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <base_dir>\n";
        std::exit(EXIT_SUCCESS);
    }
    boost::filesystem::path base_dir(argv[1]);
    
    auto input_name = base_dir / "config.toml";
    cout << "Using input file " << input_name << ".\n";
    toml_ptr g = toml_from_file(input_name.string());
    
    bool overwrite_dump = toml_get<bool>(g, "overwrite_dump", false);
    fkpm::RNG rng(toml_get<int64_t>(g, "random_seed"));
    int steps_per_dump = toml_get<int64_t>(g, "steps_per_dump");
    int max_steps = toml_get<int64_t>(g, "max_steps");
    
    auto nan = std::numeric_limits<double>::quiet_NaN();
    auto ensemble_type = toml_get<std::string>(g, "ensemble.type");
    double mu=nan, filling=nan, delta_filling=nan;
    if (ensemble_type == "grand") {
        mu = toml_get<double>(g, "ensemble.mu");
    } else {
        filling = toml_get<double>(g, "ensemble.filling");
        delta_filling = toml_get<double>(g, "ensemble.delta_filling", 0.0);
    }
    
    auto m = mk_model(g);
    auto dynamics = mk_dynamics(g);
    
    // global energy scale
    fkpm::EnergyScale ges { toml_get<double>(g, "kpm.energy_scale_lo", 0.0),
                            toml_get<double>(g, "kpm.energy_scale_hi", 0.0) };
    double lanczos_extend = toml_get<double>(g, "kpm.lanczos_extend", 0.02);
    int    lanczos_iters  = toml_get<int64_t>(g, "kpm.lanczos_iters", 128);
    
    int M                = toml_get<int64_t>(g, "kpm.cheby_order");
    int M_prec           = toml_get<int64_t>(g, "kpm.cheby_order_precise");
    int Mq               = 4*M;
    int Mq_prec          = 4*M_prec;
    Vec<int> groups      = m->groups(toml_get<int64_t>(g, "kpm.n_colors"));
    Vec<int> groups_prec = m->groups(toml_get<int64_t>(g, "kpm.n_colors_precise"));
    
    // variables that will be updated in `build_kpm(spin)`
    fkpm::EnergyScale es;
    Vec<double> moments;
    Vec<double> gamma;
    double energy;
    
    // total number of sites where local spins exist
    int n_spins = toml_get<int64_t>(g, "model.n_spins", m->n_sites);
    Vec<bool> spin_exist;
    randomize_spin_exist(spin_exist, m->n_sites, n_spins, toml_get<int64_t>(g, "random_seed"));
    
    // assumes random vectors R have been set appropriately
    auto build_kpm = [&](Vec<vec3> const& spin, int M, int Mq) {
        m->set_hamiltonian(spin, spin_exist);
        es = (ges.lo < ges.hi) ? ges : engine->energy_scale(m->H, lanczos_extend, lanczos_iters);
        engine->set_H(m->H, es);
        moments = engine->moments(M);
        gamma = fkpm::moment_transform(moments, Mq);
        energy = m->energy_classical(m->spin) + dynamics->pseudo_kinetic_energy(*m);
        if (ensemble_type == "canonical") {
            mu = filling_to_mu(gamma, es, m->kT(), filling, delta_filling);
            energy += electronic_energy(gamma, es, m->kT(), filling, mu);
        } else {
            filling = mu_to_filling(gamma, es, m->kT(), mu);
            energy += electronic_grand_energy(gamma, es, m->kT(), mu);
        }
        auto c = expansion_coefficients(M, Mq, std::bind(fkpm::fermi_energy, std::placeholders::_1, m->kT(), mu), es);
        engine->autodiff_matrix(c, m->D);
    };
    
    auto calc_force = [&](Vec<vec3> const& spin, Vec<vec3>& force) {
        build_kpm(spin, M, Mq);
        m->set_forces(m->D, spin, force);
    };
    
    auto dump = [&](int step) {
        engine->set_R_correlated(groups_prec, rng);
        build_kpm(m->spin, M_prec, Mq_prec);
        double e = energy / m->n_sites;
        
        std::stringstream fname;
        fname << "dump" << std::setfill('0') << std::setw(5) << step/steps_per_dump << ".json";
        cout << "Dumping file " << base_dir/"dump"/fname.str() << ", time=" << m->time << ", energy=" << e << ", n=" << filling << ", mu=" << mu << endl;
        boost::filesystem::ofstream dump_file(base_dir/"dump"/fname.str());
        
        auto write_vec = [&](std::string name, Vec<double> &v) {
            dump_file << "\"" << name << "\":[";
            for (int i = 0; i < v.size(); i++) {
                dump_file << v[i];
                if (i < v.size()-1) dump_file << ",";
            }
            dump_file << "],\n";
        };
        
        dump_file << "{\n" <<
        "\"n_steps\":" << dynamics->n_steps << ",\n" <<
        "\"time\":" << m->time << ",\n" <<
        "\"action\":" << e << ",\n" <<
        "\"filling\":" << filling << ",\n" <<
        "\"mu\":" << mu << ",\n" <<
        "\"energy_scale_lo\":" << es.lo << ",\n" <<
        "\"energy_scale_hi\":" << es.hi << ",\n";
        
        // density of states
        Vec<double> dos_x, dos_rho, dos_irho;
        density_function(gamma, es, dos_x, dos_rho);
        integrated_density_function(gamma, es, dos_x, dos_irho);
        for (int i = 0; i < dos_x.size(); i++) {
            dos_x[i]    -= mu;
            dos_rho[i]  /= m->n_sites * m->n_orbs;
            dos_irho[i] /= m->n_sites * m->n_orbs;
        }
        write_vec("dos_x", dos_x);
        write_vec("dos_rho", dos_rho);
        write_vec("dos_irho", dos_irho);
        
        // rng state
        dump_file << "\"rng_state\":\"";
        serialize_to_hex(rng, dump_file);
        dump_file << "\",\n";
        
        // site occupations
        Vec<double> occupation(m->n_sites, 0.0);
        for (int i = 0; i < m->n_sites; i++) {
            for (int o = 0; o < m->n_orbs; o++) {
                occupation[i] += std::real(*m->D(m->n_orbs*i+o, m->n_orbs*i+o)) / m->n_orbs;
            }
        }
        write_vec("occupation", occupation);
        
        // classical magnetic moments
        dump_file << "\"spin\":[";
        for (int i = 0; i < m->n_sites; i++) {
            dump_file << m->spin[i].x << ",";
            dump_file << m->spin[i].y << ",";
            dump_file << m->spin[i].z;
            if (i < m->n_sites-1) dump_file << ",";
        }
        dump_file << "]\n";
        
        dump_file << "}\n";
        dump_file.close();
    };
    
    auto write_json_vis = [&]() {
        boost::filesystem::ofstream json_file(base_dir/"cfg.json");
        json_file <<
        R"(// Automatically generated for backward-compatibility with visualization tools
        {
            "T": 0,
            "mu": 0,
            "order": 0,
            "order_exact": 0,
            "dt_per_rand": 0,
            "nrand": 0,
            "dumpPeriod": 0,
            "initSpin": 0,
            "model": {
                )";
                auto lattice = toml_get<std::string>(g, "model.lattice");
                json_file << "    \"type\": \"" << lattice << "\",\n";
                if (lattice == "cubic") {
                    json_file << "    \"w\": " << toml_get<int64_t>(g, "model.lx") << ",\n";
                    json_file << "    \"h\": " << toml_get<int64_t>(g, "model.ly")*toml_get<int64_t>(g, "model.lz") << ",";
                }
                else {
                    json_file << "    \"w\": " << toml_get<int64_t>(g, "model.w") << ",\n";
                    json_file << "    \"h\": " << toml_get<int64_t>(g, "model.h") << ",";
                }
                json_file << R"(
                "t": 0,
                "t1": 0,
                "t2": 0,
                "J_H": 0,
                "B_n": 0
            }
        })";
        json_file.close();
    };
    
    if (!boost::filesystem::is_directory(base_dir/"dump")) {
        boost::filesystem::create_directory(base_dir/"dump");
    }
    boost::filesystem::directory_iterator it(base_dir/"dump"), end_it;
    if(!overwrite_dump && it != end_it) {
        // non-empty dump directory -- resume previous simulation
        read_state_from_dump(base_dir/"dump", rng, *m, *dynamics);
    }
    else {
        // empty dump directory -- start from scratch
        auto init_spins_type = toml_get<std::string>(g, "init_spins.type");
        if (init_spins_type == "random") {
            m->set_spins_random(rng, m->spin);
        } else {
            m->set_spins(init_spins_type, toml_get<toml_ptr>(g, "init_spins"), m->spin);
        }
        double magnitude = toml_get(g,"init_spins.magnitude", 1.0);
        for (vec3 &s : m->spin) {
            s *= magnitude;
        }
        
        write_json_vis();
        dump(dynamics->n_steps);
        
        engine->set_R_correlated(groups, rng);
        dynamics->init(calc_force, rng, *m);
    }
    
    while (dynamics->n_steps < max_steps) {
        engine->set_R_correlated(groups, rng);
        dynamics->step(calc_force, rng, *m);
        if (dynamics->n_steps % steps_per_dump == 0) {
            dump(dynamics->n_steps);
        }
    }
    std::cout << "Reached max_steps value = " << max_steps << ".\n";

    return EXIT_SUCCESS;
}
