#include "kondo.h"
#include "iostream_util.h"
#include <cstdio>
#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

using boost::property_tree::ptree;
using boost::property_tree::read_json;
//using boost::property_tree::write_json;

template <typename T>
std::vector<T> as_vector(ptree const& pt, ptree::key_type const& key)
{
    std::vector<T> r;
    for (auto& item : pt.get_child(key))
        r.push_back(item.second.get_value<T>());
    return r;
}

// same as in kondo.cpp. Should merge later.
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
    
    // setup dilute spin positions
    int num_sites = ret->n_sites;
    double filling_spins = toml_get<double>(g, "model.filling_spins", 1.0);
    auto& spin_exist = ret->spin_exist;
    assert(filling_spins > 0.0 && filling_spins <= 1.0);
    if (std::abs(filling_spins-1.0) < 1e-10) {                                  // every site occupied
        spin_exist.clear();
    } else {                                                                    // some sites empty
        spin_exist.assign(num_sites, false);
        auto rng_spin = std::mt19937(toml_get<int64_t>(g, "random_seed"));
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        std::cout << "Sites occupied by spins: " << std::endl;
        for (int i = 0; i < num_sites; i++) {
            if (dist(rng_spin) < filling_spins) {
                spin_exist[i] = true;
                std::cout << i << ",";
            }
        }
        std::cout << std::endl;
    }
    ret->allow_update.clear();
    for (int i = 0; i < num_sites; i++) {
        if (spin_exist.empty() || spin_exist[i]) ret->allow_update.push_back(i);
    }
    std::cout << "spin concentration input:    " << filling_spins << std::endl;
    std::cout << "spin concentration realized: "
    << static_cast<double>(ret->allow_update.size()) / num_sites << std::endl;
    return ret;
}

void conductivity(int argc, char *argv[]) {
    auto engine = fkpm::mk_engine_mpi<cx_flt>();
    if (engine == nullptr) std::exit(EXIT_FAILURE);
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <base_dir>\n";
        std::exit(EXIT_SUCCESS);
    }
    
    std::string base_dir(argv[1]);
    auto input_name = base_dir + "/config.toml";
    std::cout << "using toml file `" << input_name << "`!\n";
    toml_ptr g = toml_from_file(input_name);
    
    auto m = mk_model(g);
    
    std::string json_name;
    std::cout << "Input dumpfile name:" << std::endl;
    std::cin >> json_name;
    std::cout << json_name << std::endl;
    json_name = base_dir + "/dump/" + json_name;
    std::ifstream json_file(json_name);
    if (!json_file.is_open()) {
        cerr << "Unable to open file `" << json_name << "`!\n";
        std::exit(EXIT_FAILURE);
    }
    cout << "Using json file `" << json_name << "`!\n";
    ptree pt_json;        // used for reading the json file
    std::string json_eachline, json_contents;
    while (std::getline(json_file, json_eachline)) {
        json_contents += json_eachline;
    }
    json_file.close();
    std::istringstream is (json_contents);
    read_json (is, pt_json);
    auto time    = pt_json.get<int>    ("time");
    auto action  = pt_json.get<double> ("action");
    auto filling = pt_json.get<double> ("filling");
    auto mu      = pt_json.get<double> ("mu");
    auto spin    = as_vector<double>(pt_json, "spin");
    std::cout << "lattice: " << toml_get<std::string>(g, "model.lattice") << std::endl;
    std::cout << "time:    " << time << std::endl;
    std::cout << "action:  " << action << std::endl;
    std::cout << "filling: " << filling << std::endl;
    std::cout << "mu:      " << mu << std::endl;
    std::cout << "kT:      " << m->kT() << std::endl;
    
    // build spin configuration from dump file
    assert(spin.size() == m->n_sites * 3);
    for (int i = 0; i < m->n_sites; i++) {
        m->spin[i] = vec3(spin[3*i],spin[3*i+1],spin[3*i+2]);
        if (m->spin_exist.empty() || m->spin_exist[i]) {
            assert(m->spin[i].norm2() > 1e-10);
        } else {
            assert(m->spin[i].norm2() < 1e-10);
            // put (0,0,1) when running, to avoid potential normalization error
            m->spin[i] = {0.0, 0.0, 1.0};
        }
    }
    m->set_hamiltonian(m->spin);
    fkpm::EnergyScale es;
    es = engine->energy_scale(m->H, 0.05);
    std::cout << "energyscale: [" << es.lo << ", " << es.hi << "]" << std::endl << std::endl;
    
    std::cout << "---------- parameters for conductivity calculation ----------" << std::endl;
    vec3 dir1(1.0,0.0,0.0), dir2(1.0,0.0,0.0);
    std::cout << "Input x,y,z direction of the 1st current operator (3 numbers separated by space):" << std::endl;
    std::cin >> dir1.x >> dir1.y >> dir1.z;
    std::cout << dir1 << std::endl;
    std::cout << "Input x,y,z direction of the 2nd current operator (3 numbers separated by space):" << std::endl;
    std::cin >> dir2.x >> dir2.y >> dir2.z;
    std::cout << dir2 << std::endl;
    auto j1 = m->electric_current_operator(m->spin, dir1);
    auto j2 = m->electric_current_operator(m->spin, dir2);
    
    int M, Mq, n_colors, seed;
    std::cout << "Input M:" << std::endl;
    std::cin >> M;
    std::cout << M << std::endl;
    std::cout << "Input Mq:" << std::endl;
    std::cin >> Mq;
    std::cout << Mq << std::endl;
    std::cout << "Input n_colors:" << std::endl;
    std::cin >> n_colors;
    std::cout << n_colors << std::endl;
    std::cout << "Input random seed:" << std::endl;
    std::cin >> seed;
    std::cout << seed << std::endl << std::endl;
    fkpm::RNG rng(seed);
    engine->set_R_uncorrelated(m->H.n_rows, 2*n_colors, rng);
    
    auto kernel = fkpm::jackson_kernel(M);
    engine->set_H(m->H, es);
    
    // calculate dos
    auto moments_dos = engine->moments(M);
    auto gamma = fkpm::moment_transform(moments_dos, Mq);
    fkpm::Vec<double> mu_list, rho;
    fkpm::density_function(gamma, es, mu_list, rho);
    for (int i = 0; i < rho.size(); i++) {
        rho[i]/=m->n_sites;
    }
    
    // calculate conductivity
    std::cout << "calculating moments2... " << std::flush;
    fkpm::timer[0].reset();
    Vec<Vec<fkpm::cx_double>> moments_sigma = engine->moments2_v1(M, j1, j2, 10, 16);
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";
    
    cout << "calculating dc conductivities... " << std::flush;
    std::ofstream fout2("full_time"+std::to_string(time)+"_M"+std::to_string(M)+
                        "_color"+std::to_string(n_colors)+"_seed"+std::to_string(seed)+".dat", std::ios::out);
    fout2 << std::scientific << std::right;
    fout2 << std::setw(20) << "#(1)" << std::setw(20) << "(2)" << std::setw(20) << "(3)" << std::endl;
    fout2 << std::setw(20) << "mu" << std::setw(20) << "rho" << std::setw(20) << "sigma" << std::endl;
    int interval = std::max(Mq/400,5);
    for (int i = 0; i < Mq; i+=interval) {
        auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), mu_list[i], 0.0, es, kernel);
        auto sigma = std::real(fkpm::moment_product(cmn, moments_sigma));
        fout2 << std::setw(20) << mu_list[i]
              << std::setw(20) << rho[i]
              << std::setw(20) << sigma << std::endl;
    }
    fout2.close();
    cout << " done. " << fkpm::timer[0].measure() << "s.\n";
    
    std::ifstream fin0("result_time"+std::to_string(time)+".dat");
    if (! fin0.good()) {
        fin0.close();
        std::ofstream fout3("result_time"+std::to_string(time)+".dat", std::ios::out | std::ios::app );
        fout3 << std::setw(20) << "#(1)" << std::setw(20) << "(2)"
              << std::setw(20) <<  "(3)" << std::setw(20) << "(4)"
              << std::setw(20) <<  "(5)" << std::setw(20) << "(6)" << std::endl;
        fout3 << std::setw(20) << "M"    << std::setw(20) << "colors"
              << std::setw(20) << "seed" << std::setw(20) << "F"
              << std::setw(20) << "mu"   << std::setw(20) << "sigma" << std::endl;
        fout3.close();
    }
    std::ofstream fout3("result_time"+std::to_string(time)+".dat", std::ios::out | std::ios::app );
    fout3 << std::scientific << std::right;
    auto cmn = electrical_conductivity_coefficients(M, Mq, m->kT(), mu, 0.0, es, kernel);

    fout3 << std::setw(20) << M << std::setw(20) << n_colors
          << std::setw(20) << seed
          << std::setw(20) << fkpm::electronic_energy(gamma, es, m->kT(), filling, mu)/m->n_sites
          << std::setw(20) << mu
          << std::setw(20) << std::real(fkpm::moment_product(cmn, moments_sigma))
          << std::endl;
    fout3.close();
    std::cout << std::endl;
}


int main(int argc, char *argv[]) {
    conductivity(argc, argv);
}
