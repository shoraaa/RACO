/**
 * @author: Rafał Skinderowicz (rafal.skinderowicz@us.edu.pl)
 */
#pragma once

#include <string>

struct ProgramOptions {
    std::string algorithm_ = "faco";

    // If #ants is set to 0 then a default strategy is used to initiate it
    uint32_t ants_count_ = 0; 

    // When looking for a next node to visit it may happen that all of the
    // nodes on the candidates list were visited -- in such case we choose
    // one of the nodes from a "backup" list
    uint32_t backup_list_size_ = 64;

    // Relative importance of heuristic information, i.e. distances between
    // nodes
    double beta_ = 1;

    uint32_t cand_list_size_ = 16;

    std::string id_ = "default";  // Id of the comp. experiment

    // Probability of using the current global best as a source solution
    double gbest_as_source_prob_ = 0.01;

    int32_t iterations_ = 5 * 1000;

    int32_t local_search_ = 1;  // 0 - no local search, 1 - default LS

    uint32_t ls_cand_list_size_ = 20u;  // #nodes used by the LS heuristics

    uint32_t min_new_edges_ = 0;

    // Prob. that a solution will contain only edges with the
    // highest pheromone levels. Used to calculate pheromone trail limits.
    double p_best_ = 0.1;

    std::string problem_path_ = "kroA100.tsp";

    // By default the results will be stored in "results" folder
    std::string results_dir_ = "results";

    // How much of the pheromone remains after a single evaporation event
    double rho_ = 0.5;

    // Should a picture of the solution (route) be stored into SVG file?
    bool save_route_picture_ = true;

    // Random number generator seed -- 0 means that seed should be 
    // based on the built-in std::random_device
    uint64_t seed_ = 0;

    int32_t repeat_ = 1;

    int32_t threads_ = 0;  // If > 0 then force specific # of threads in OpenMP

    // Modified FACO -- should we keep the previous ant solution if new one
    // is worse? Default: true
    int32_t keep_better_ant_sol_ = 1;

    // Modified FACO -- should the (local) source solution be immediately
    // replaced with a new better solution (if its found) in the current iteration
    int32_t source_sol_local_update_ = 1;

    // Modified FACO -- should the actual # of new edges in the constructed
    // solutions be checked?
    int32_t count_new_edges_ = 0;

    // Restricted ACO -- Prob to greedily selected the nearest vertice
    double p_greed_ = 0.5;

    // Restricted ACO -- should we force the new edge to be choose?
    int32_t force_new_edge_ = 1;

    // Restricted ACO -- should we use smoothen pheromone update?
    int32_t smooth_ = 1;

    // Restricted ACO -- minimum rho for smmas
    double tau_min_ = -1;

    // Restricted ACO -- number of sub ants
    double sub_ants_ = 4;

    // doubled steps
    int32_t steps_ = 4000;
};


template<typename MapT>
void dump(const ProgramOptions &opt, MapT &map) {
    map["alg"] = opt.algorithm_;
    map["ants"] = opt.ants_count_;
    map["backup list size"] = opt.backup_list_size_;
    map["beta"] = opt.beta_;
    map["cand list size"] = opt.cand_list_size_;
    map["id"] = opt.id_;
    map["gbest as source prob"] = opt.gbest_as_source_prob_;
    map["iterations"] = opt.iterations_;
    map["local search"] = opt.local_search_;
    map["ls cand list size"] = opt.ls_cand_list_size_;
    map["min new edges"] = opt.min_new_edges_;
    map["p best"] = opt.p_best_;
    map["problem"] = opt.problem_path_;
    map["results dir"] = opt.results_dir_;
    map["rho"] = opt.rho_;
    map["seed"] = opt.seed_;
    map["picture"] = opt.save_route_picture_;
    map["repeat"] = opt.repeat_;
    map["threads"] = opt.threads_;
    map["keep better ant sol"] = opt.keep_better_ant_sol_;
    map["source sol local update"] = opt.source_sol_local_update_;
    map["count new edges"] = opt.count_new_edges_;

    map["p greed"] = opt.p_greed_;
    map["force new edge"] = opt.force_new_edge_;
    map["smooth"] = opt.smooth_;
    map["tau min"] = opt.tau_min_;
    map["sub ants"] = opt.sub_ants_;
    map["steps"] = opt.steps_;
}

ProgramOptions parse_program_options(int argc, char *argv[]);
