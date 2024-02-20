#include "InferStrategy.h"
#include "Pagmoprob.h"
#include "PagmoMultiObjCluster.h"
#include "PagmoMle.h"
#include <RInside.h>
#include <limits>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/algorithms/moead.hpp>
#include <pagmo/algorithms/moead_gen.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/utils/multi_objective.hpp>
#include <pagmo/problems/unconstrain.hpp>
#include <pagmo/algorithms/cstrs_self_adaptive.hpp>
#include <pagmo/algorithms/pso_gen.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/algorithms/sga.hpp>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <Eigen/Dense>







void findClusterParams(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3) {

    // Open the file for reading and appending
    std::string filename_cluster = "clusterMLEParams.txt";
    std::ifstream cluster_infile(filename_cluster);
    std::map<std::string, std::vector<double>> paramClusterMap;
    boost::archive::text_iarchive ia_cluster(cluster_infile);
    ia_cluster >> paramClusterMap;
    cluster_infile.close();

    std::cout << "paramClusterMap: ";
    for (const auto& entry : paramClusterMap) {
        const std::string& key = entry.first;
        const std::vector<double>& values = entry.second;

        // Print key
        std::cout << "Key: " << key << ", Values: ";

        // Print values in the vector
        for (double value : values) {
            std::cout << value << " ";
        }

        std::cout << std::endl;
    }


    std::cout << "Initializing problem class" <<std::endl;
    // Create a function to optimize

    PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
    //PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
    std::cout << "Initialized problem class" <<std::endl;

    // Create a problem using Pagmo
    problem prob{pagmoprob};

    //unconstrain unprob{prob, "kuri"};
    //2 - Instantiate a pagmo algorithm (self-adaptive differential
    ////evolution, 100 generations).
    // pagmo::algorithm algo{sade(10,2,2)};
    // //pagmo::algorithm algo{de(5)};
    // //pagmo::cstrs_self_adaptive algo{5, de(1)};
    // //algo.set_verbosity(1);
    // // ////pagmo::algorithm algo{sade(20)};

    // archipelago archi{5u, algo, prob, 10u};

    // // // ///4 - Run the evolution in parallel on the 5 separate islands 5 times.
    // archi.evolve(5);
    // // std::cout << "DONE1:"  << '\n';

    // // ///5 - Wait for the evolutions to finish.
    // archi.wait_check();

    // // ///6 - Print the fitness of the best solution in each island.

    // double champion_score = 1e8;
    // std::vector<double> dec_vec_champion;
    // for (const auto &isl : archi) {
    //     std::vector<double> dec_vec = isl.get_population().champion_x();
        
    //     // std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
    //     // for (auto const& i : dec_vec)
    //     //     std::cout << i << ", ";
    //     // std::cout << "\n" ;

    //     double champion_isl = isl.get_population().champion_f()[0];
    //     if(champion_isl < champion_score)
    //     {
    //         champion_score = champion_isl;
    //         dec_vec_champion = dec_vec;
    //     }
    // }

    // std::cout << "Final champion = " << champion_score << std::endl;
    // for (auto const& i : dec_vec_champion)
    //     std::cout << i << ", ";
    // std::cout << "\n" ;


    // pagmo::thread_bfe thread_bfe;
    // pagmo::sga method ( 10 );
    // //pagmo::gaco method(10);
    // // method.set_bfe ( pagmo::bfe { thread_bfe } );
    // pagmo::algorithm algo = pagmo::algorithm { method };
    // pagmo::population pop {prob, 100 };
    // // Evolve the population for 100 generations
    // for ( auto evolution = 0; evolution < 5; evolution++ ) {
    //     pop = algo.evolve(pop);
    // }

    pagmo::thread_bfe thread_bfe;
    pagmo::pso_gen method ( 10 );
    //pagmo::gaco method(10);
    method.set_bfe ( pagmo::bfe { thread_bfe } );
    pagmo::algorithm algo = pagmo::algorithm { method };
    // pagmo::algorithm algo(pagmo::cstrs_self_adaptive{10,method});
    pagmo::population pop { prob, thread_bfe,10 };
    // Evolve the population for 100 generations
    for ( auto evolution = 0; evolution < 2; evolution++ ) {
        pop = algo.evolve(pop);
    }

    // pagmo::algorithm algo{de(5)};
    // pagmo::population pop { unprob, 500 };
    // for ( auto evolution = 0; evolution < 10; evolution++ ) {
    //     pop = algo.evolve(pop);
    // }
    
    std::vector<double> dec_vec_champion = pop.champion_x();
    std::cout << "Final champion = " << pop.champion_f()[0] << std::endl;

    // std::cout << "dec_vec_champion: ";
    // for (const auto &x : dec_vec_champion) {
    //     std::cout << x << " ";
    // }
    // std::cout << "\n";


    const auto fv = prob.fitness(dec_vec_champion);
    std::cout << "Value of the objfun in dec_vec_champion: " << fv[0] << '\n';
    // std::cout << "Value of the eq. constraint in dec_vec_champion: " << fv[1] << '\n';
    // std::cout << "Value of the ineq. constraint in dec_vec_champion: " << fv[2] << '\n';
    // std::cout << "Value of the ineq. constraint in dec_vec_champion: " << fv[3] << '\n';
    // std::cout << "Value of the ineq. constraint in dec_vec_champion: " << fv[4] << '\n';

    // std::vector<std::pair<double, std::vector<double>>> indexedValues = pagmoprob.getIndexedValues();
    // std::cout << "indexedValues.size=" << indexedValues.size() << std::endl;
    
    // Create a new population with the filtered individuals
    // pagmo::population new_pop(prob, 0); // empty population
    // new_pop.push_back(dec_vec_champion);
    

    // std::cout << "New pop size=" << new_pop.size() << std::endl;

    // pagmo::algorithm algo2{pagmo::nlopt("cobyla")};
    // algo2.set_verbosity(5);
    // new_pop = algo2.evolve(new_pop);
    // std::vector<double> cobyla_champ_f = new_pop.champion_x();

    // std::cout << "Best fitness: " << new_pop.champion_f()[0] << "\n";
    // std::cout << "Best decision vector: ";
    // for (const auto &x : cobyla_champ_f) {
    //     std::cout << x << " ";
    // }
    // std::cout << "\n";

    std::string rat = ratdata.getRat();
    paramClusterMap[rat] = dec_vec_champion;

    std::cout << "Updated paramClusterMap: ";
    for (const auto& entry : paramClusterMap) {
        const std::string& key = entry.first;
        const std::vector<double>& values = entry.second;

        // Print key
        std::cout << "Key: " << key << ", Values: ";

        // Print values in the vector
        for (double value : values) {
            std::cout << value << " ";
        }

        std::cout << std::endl;
    }

    std::ofstream file(filename_cluster);
    boost::archive::text_oarchive oa(file);
    oa << paramClusterMap;
    file.close();
    

    return;
}

void findParams(RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3)
{
    std::vector<std::string> models = {"m1","m2","m3","m4","m5","m6"};
    std::vector<std::vector<double>> modelParams;

    for(int i=0; i<6; i++)
    {
        PagmoMle pagmomle(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3,models[i]);
        std::cout << "Initialized problem class, model=" << models[i] <<std::endl;

        // Create a problem using Pagmo
        problem prob{pagmomle};

        //pagmo::algorithm algo{de(5)};
        pagmo::algorithm algo{sade(10,2,2)};


        archipelago archi{5u, algo, prob, 10u};

        // // ///4 - Run the evolution in parallel on the 5 separate islands 5 times.
        archi.evolve(5);
        // std::cout << "DONE1:"  << '\n';

        // ///5 - Wait for the evolutions to finish.
        archi.wait_check();

        // ///6 - Print the fitness of the best solution in each island.

        double champion_score = 1e8;
        std::vector<double> dec_vec_champion;
        for (const auto &isl : archi) {
            std::vector<double> dec_vec = isl.get_population().champion_x();
            
            // std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
            // for (auto const& i : dec_vec)
            //     std::cout << i << ", ";
            // std::cout << "\n" ;

            double champion_isl = isl.get_population().champion_f()[0];
            if(champion_isl < champion_score)
            {
                champion_score = champion_isl;
                dec_vec_champion = dec_vec;
            }
        }

        // pagmo::thread_bfe thread_bfe;
        // pagmo::pso_gen method ( 10 );
        // //pagmo::gaco method(10);
        // method.set_bfe ( pagmo::bfe { thread_bfe } );
        // pagmo::algorithm algo = pagmo::algorithm { method };
        // pagmo::population pop { prob, thread_bfe, 100 };
        // // Evolve the population for 100 generations
        // for ( auto evolution = 0; evolution < 5; evolution++ ) {
        //     pop = algo.evolve(pop);
        // }

        // std::vector<double> dec_vec_champion = pop.champion_x();
        // std::cout << "Final champion = " << pop.champion_f()[0] << std::endl;

        std::cout << "Final champion = " << champion_score << std::endl;
        std::cout << "dec_vec_champion: " << std::endl;
        for (auto const& i : dec_vec_champion)
            std::cout << i << ", ";
        std::cout << "\n" ;


        modelParams.push_back(dec_vec_champion);

    }
    std::string rat = ratdata.getRat();
    std::string filename_cluster = "clusterMLE_" + rat + ".txt" ;
    std::ofstream outFile(filename_cluster);
    for (const auto &vec : modelParams) {
        for (const auto &val : vec) {
            outFile << val << ' ';
        }
        outFile << '\n';
    }
    outFile.close();


    return ;
}



std::vector<double> findMultiObjClusterParams(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3) {

   std::cout << "Initializing problem class" <<std::endl;
    // Create a function to optimize
    PagmoMultiObjCluster pagmoMultiObjProb(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
    //PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
    std::cout << "Initialized problem class" <<std::endl;

    // Create a problem using Pagmo
    problem prob{pagmoMultiObjProb};
    //problem prob{schwefel(30)};
    
    std::cout << "created problem" <<std::endl;
    // 2 - Instantiate a pagmo algorithm (self-adaptive differential
    // evolution, 100 generations).

    pagmo::thread_bfe thread_bfe;
    pagmo::moead_gen method (10);
    method.set_bfe(pagmo::bfe { thread_bfe } );
    pagmo::algorithm algo = pagmo::algorithm { method };
    pagmo::population pop { prob, thread_bfe, 56};
   

    // Evolve the population for 100 generations
    for ( auto evolution = 0; evolution < 5; evolution++ ) {
        pop = algo.evolve(pop);
    }
    

    std::cout << "DONE1:"  << '\n';
    //system("pause"); 

    // auto best = pagmo::select_best_N_mo(pop.get_f(), 10);

    // // Print the objective vectors of the best individuals
    //  std::cout << "Best " << 10 << " Individuals on Pareto Front:\n";
    // for (const auto& ind : best) {
    //     std::cout << ind << std::endl;
    // }


    auto f = pop.get_f();
    auto x = pop.get_x();

    // Sort the individuals by non-domination rank and crowding distance
    pagmo::vector_double::size_type n = pop.size();
    std::vector<pagmo::vector_double::size_type> idx = pagmo::sort_population_mo(f);

    //std::vector<double> cd = pagmo::crowding_distance(f);

    double min_lik = 100000;
    std::vector<double> dec_vec_champion;
    // Select the first 10 individuals as the best ones
    for (int i = 0; i < 10; i++) {
        std::cout << "Individual " << i + 1 << ":" << std::endl;
        double lik = std::accumulate(f[idx[i]].begin(), f[idx[i]].end(), 0.0);
        //std::cout << "Fitness: [" << f[idx[i]][0] << ", " << f[idx[i]][1] <<  ", " << f[idx[i]][2] << ", " << f[idx[i]][3] << ", " << f[idx[i]][4] << ", " << f[idx[i]][5] << "]" << ", lik=" << lik << std::endl;
        //std::cout << "Decision vector: [" << x[idx[i]][0] << "]" << std::endl;
        //std::cout << "Crowding distance: " << cd[idx[i]] << std::endl;

        std::vector<double> dec_vec = x[idx[i]];

        // std::cout << "dec_vec: ";
        // for (const auto& val : dec_vec) {
        //     std::cout << ", " << val ;
        // }

        std::cout << std::endl;

        if(lik < min_lik)
        {
            min_lik = lik;
            dec_vec_champion = dec_vec;
        }

    }

    // // Perform the fast non-dominated sorting
    // auto result = pagmo::fast_non_dominated_sorting(f);

    // std::vector<std::vector<pagmo::population::size_type>> fronts = std::get<0>(result);
    // //auto crowding = std::get<1>(result);

    // // Print the results
    // for (int i = 0; i < fronts.size(); i++) {
    //     std::cout << "Front " << i + 1 << ":" << std::endl;
    //     for (int j = 0; j < fronts[i].size(); j++) {
    //         std::cout << "\tIndividual " << fronts[i][j] + 1 << ":" << std::endl;
    //         double lik = std::accumulate(f[fronts[i][j]].begin(), f[fronts[i][j]].end(), 0.0);
    //         std::cout << "\tFitness: [" << f[fronts[i][j]][0] << ", " << f[fronts[i][j]][1] <<  ", " << f[fronts[i][j]][2] << ", " << f[fronts[i][j]][3] << ", " << f[fronts[i][j]][4] << ", " << f[fronts[i][j]][5] << "]" << ", lik=" << lik << std::endl;

    //         //std::cout << "\t\tCrowding distance: " << crowding[fronts[i][j]] << std::endl;
    //     }
    // }


    std::cout << "dec_vec_champion: ";
    for (const auto& val : dec_vec_champion) {
        std::cout << ", " << val ;
    }

       

    return dec_vec_champion;
}






void runEM(RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3, std::vector<std::vector<double>> modelParams, RInside& R, bool debug)
{
    
    std::string rat = ratdata.getRat();

    //m1 params
    std::vector<double> v1 = modelParams[0];

    double alpha_aca_subOptimal_m1 = v1[0];
    double gamma_aca_subOptimal_m1 = v1[1];
    double alpha_aca_optimal_m1 = v1[0];
    double gamma_aca_optimal_m1 = v1[1];

    int n1 = static_cast<int>(std::floor(v1[2]));
 
    //m2 params
    std::vector<double> v2 = modelParams[1];
    
    double alpha_aca_subOptimal_m2 = v2[0];
    double gamma_aca_subOptimal_m2 = v2[1];

    double alpha_drl_optimal_m2 = v2[2];
    double beta_drl_optimal_m2 = v2[3];
    double lambda_drl_optimal_m2 = v2[4];

    int n2 = static_cast<int>(std::floor(v2[5]));


    //m3 params
    std::vector<double> v3 = modelParams[2];

    double alpha_drl_subOptimal_m3 = v3[0];
    double beta_drl_subOptimal_m3 = v3[1];
    double lambda_drl_subOptimal_m3 = v3[2];

    double alpha_aca_optimal_m3 = v3[3];
    double gamma_aca_optimal_m3 = v3[4];
    int n3 = static_cast<int>(std::floor(v3[5]));   

    //m4
    std::vector<double> v4 = modelParams[3];

    double alpha_drl_subOptimal_m4 = v4[0];
    double beta_drl_subOptimal_m4 = v4[1];
    double lambda_drl_subOptimal_m4 = v4[2];

    double alpha_drl_optimal_m4 = v4[0];
    double beta_drl_optimal_m4 = v4[1];
    double lambda_drl_optimal_m4 = v4[2];

    int n4 = static_cast<int>(std::floor(v4[3]));

    //m5
    std::vector<double> v5 = modelParams[4];
    double alpha_aca_optimal_m5 = v5[0];
    double gamma_aca_optimal_m5 = v5[1];

    //m6
    std::vector<double> v6 = modelParams[5];
    double alpha_drl_optimal_m6 = v6[0];
    double beta_drl_optimal_m6 = v6[1];
    double lambda_drl_optimal_m6 = v6[2];

      
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3_m1 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal_m1, gamma_aca_subOptimal_m1, 0, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3_m1 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal_m1, gamma_aca_optimal_m1, 0, 0, 0, 0, true);

    auto aca2_Suboptimal_Hybrid3_m2 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal_m2, gamma_aca_subOptimal_m2, 0, 0, 0, 0, false);
    auto drl_Optimal_Hybrid3_m2 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal_m2, beta_drl_optimal_m2, lambda_drl_optimal_m2, 0, 0, 0, true);

    
    auto drl_Suboptimal_Hybrid3_m3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal_m3, beta_drl_subOptimal_m3, lambda_drl_subOptimal_m3, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3_m3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal_m3, gamma_aca_optimal_m3, 0, 0, 0, 0, true);


    auto drl_Suboptimal_Hybrid3_m4 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal_m4, beta_drl_subOptimal_m4, lambda_drl_subOptimal_m4, 0, 0, 0, false);
    auto drl_Optimal_Hybrid3_m4 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal_m4, beta_drl_optimal_m4, lambda_drl_optimal_m4, 0, 0, 0, true);

    auto aca2_Optimal_Hybrid3_m5 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal_m5, gamma_aca_optimal_m5, 0, 0, 0, 0, true);

    auto drl_Optimal_Hybrid3_m6 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal_m6, beta_drl_optimal_m6, lambda_drl_optimal_m6, 0, 0, 0, true);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    
    std::vector<std::string> cluster;
    std::string last_choice;

    RecordResults allResults("None", {}, {}, {}, {}, {});

    double ll1 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < n1)
        {
           ll_ses  = aca2_Suboptimal_Hybrid3_m1->getTrajectoryLikelihood(ratdata, ses);
        }else{
           ll_ses  = aca2_Optimal_Hybrid3_m1->getTrajectoryLikelihood(ratdata, ses); 
        }
        
        ll_ses = ll_ses*(-1);
        ll1 = ll1 + ll_ses;
    }
    arma::mat m1_probMat = arma::join_cols(aca2_Suboptimal_Hybrid3_m1->getPathProbMat(),aca2_Optimal_Hybrid3_m1->getPathProbMat());
    double bic_score1 = 3*log(allpaths.n_rows)+ 2*ll1;

    double ll2 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < n2)
        {
           ll_ses  = aca2_Suboptimal_Hybrid3_m2->getTrajectoryLikelihood(ratdata, ses); 
        }else{
           ll_ses  = drl_Optimal_Hybrid3_m2->getTrajectoryLikelihood(ratdata, ses); 
        }
        
        ll_ses = ll_ses*(-1);
        ll2 = ll2 + ll_ses;
    }
    arma::mat m2_probMat = arma::join_cols(aca2_Suboptimal_Hybrid3_m2->getPathProbMat(),drl_Optimal_Hybrid3_m2->getPathProbMat());
    double bic_score2 = 6*log(allpaths.n_rows)+ 2*ll2;

    double ll3 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < n3)
        {
           ll_ses  = drl_Suboptimal_Hybrid3_m3->getTrajectoryLikelihood(ratdata, ses); 
        }else{
           ll_ses  = aca2_Optimal_Hybrid3_m3->getTrajectoryLikelihood(ratdata, ses); 
        }
        
        ll_ses = ll_ses*(-1);
        ll3 = ll3 + ll_ses;
    }
    arma::mat m3_probMat = arma::join_cols(drl_Suboptimal_Hybrid3_m3->getPathProbMat(),aca2_Optimal_Hybrid3_m3->getPathProbMat());
    double bic_score3 = 6*log(allpaths.n_rows)+ 2*ll3;

    double ll4 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < n4)
        {
           ll_ses  = drl_Suboptimal_Hybrid3_m4->getTrajectoryLikelihood(ratdata, ses); 
        }else{
           ll_ses  = drl_Optimal_Hybrid3_m4->getTrajectoryLikelihood(ratdata, ses); 
        }
        
        ll_ses = ll_ses*(-1);
        ll4 = ll4 + ll_ses;
    }
    arma::mat m4_probMat = arma::join_cols(drl_Suboptimal_Hybrid3_m4->getPathProbMat(),drl_Optimal_Hybrid3_m4->getPathProbMat());
    double bic_score4 = 4*log(allpaths.n_rows)+ 2*ll4;

    double ll5 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses  = aca2_Optimal_Hybrid3_m5->getTrajectoryLikelihood(ratdata, ses); 
        
        ll_ses = ll_ses*(-1);
        ll5 = ll5 + ll_ses;
    }
    arma::mat m5_probMat = aca2_Optimal_Hybrid3_m5->getPathProbMat();
    double bic_score5 = 2*log(allpaths.n_rows)+ 2*ll5;

    double ll6 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses  = drl_Optimal_Hybrid3_m6->getTrajectoryLikelihood(ratdata, ses); 
        
        ll_ses = ll_ses*(-1);
        ll6 = ll6 + ll_ses;
    }
    arma::mat m6_probMat = drl_Optimal_Hybrid3_m6->getPathProbMat();
    double bic_score6 = 3*log(allpaths.n_rows)+ 2*ll6;

    std::cout << "acaSubOpt + aca2Opt, n=" << n1  << ", lik=" << ll1 << ", bic=" << bic_score1 << std::endl;
    std::cout << "acaSubOpt + drlOpt, n=" << n2  << ", lik=" << ll2 << ", bic=" << bic_score2 << std::endl;
    std::cout << "drlSubOpt + aca2Opt, n=" << n3  << ", lik=" << ll3 << ", bic=" << bic_score3 << std::endl;
    std::cout << "drlSubOpt + drlOpt, n=" << n4  << ", lik=" << ll4 << ", bic=" << bic_score4 << std::endl;
    std::cout << "aca2Opt"  << ", lik=" << ll5 << ", bic=" << bic_score5 << std::endl;
    std::cout << "drlOpt"  << ", lik=" << ll6 << ", bic=" << bic_score6 << std::endl; 

    std::vector<double> bic_scores = {bic_score1, bic_score2, bic_score3, bic_score4, bic_score5, bic_score6};
    std::vector<int> n = {n1,n2,n3,n4,-1,-1};
    std::vector<double> sortedVector = bic_scores;
    std::sort(sortedVector.begin(), sortedVector.end());

    // Get the largest and second-largest elements
    double smallest = sortedVector[0];
    double secondSmallest = sortedVector[1];

    int smallestIdx = -1;
    // Check if the largest element is greater than the second largest by at least 5
    if ((secondSmallest - smallest) > 2.0) {
        // Find the index of the largest element in the original vector
        auto it = std::find(bic_scores.begin(), bic_scores.end(), smallest);
        if (it != bic_scores.end()) {
            // Calculate the index using std::distance
            smallestIdx = std::distance(bic_scores.begin(), it);
            std::cout << "Index of the smallest element: " << smallestIdx << std::endl;
        } else {
            std::cout << "Error: Couldn't find the index of the smallest element." << std::endl;
        }
    }

    std::vector<std::string> selectedStrategies;
    std::vector<std::string> trueGenStrategies = ratdata.getGeneratorStrategies();
    if(smallestIdx == 0)
    {
        std::vector<std::string> v(sessions, "aca2_Suboptimal_Hybrid3"); 
        for (int i = n1; i < sessions; i++) {
            v[i] = "aca2_Optimal_Hybrid3"; // Set the i-th element to acaOpt
        }
        selectedStrategies = v;
    }else if(smallestIdx == 1)
    {
        std::vector<std::string> v(sessions, "aca2_Suboptimal_Hybrid3"); 
        for (int i = n2; i < sessions; i++) {
            v[i] = "drl_Optimal_Hybrid3"; // Set the i-th element to acaOpt
        }
        selectedStrategies = v;
    }else if(smallestIdx == 2)
    {
        std::vector<std::string> v(sessions, "drl_Suboptimal_Hybrid3"); 
        for (int i = n3; i < sessions; i++) {
            v[i] = "aca2_Optimal_Hybrid3"; // Set the i-th element to acaOpt
        }
        selectedStrategies = v;
    }else if(smallestIdx == 3)
    {
        std::vector<std::string> v(sessions, "drl_Suboptimal_Hybrid3"); 
        for (int i = n4; i < sessions; i++) {
            v[i] = "drl_Optimal_Hybrid3"; // Set the i-th element to acaOpt
        }
        selectedStrategies = v;
    }else if(smallestIdx == 4)
    {
        std::vector<std::string> v(sessions, "aca2_Optimal_Hybrid3"); 
        selectedStrategies = v;
    }else if(smallestIdx == 5)
    {
        std::vector<std::string> v(sessions, "drl_Optimal_Hybrid3"); 
        selectedStrategies = v;
    }else if(smallestIdx == 5)
    {
        std::cout << "No bic score sufficiently small";
        std::vector<std::string> v(sessions, "None"); 
        selectedStrategies = v;
    }

    // probMat.save("ProbMat_Sim_" + rat+ ".csv", arma::csv_ascii);
   

    // m1_probMat.save("m1_probMat_" + rat+ ".csv", arma::csv_ascii);
    // m2_probMat.save("m2_probMat_"+ rat+".csv", arma::csv_ascii);
    // m3_probMat.save("m3_probMat_"+ rat+".csv", arma::csv_ascii);
    // m4_probMat.save("m4_probMat_" + rat+ ".csv", arma::csv_ascii);
    // m5_probMat.save("m5_probMat_"+ rat+".csv", arma::csv_ascii);
    // m6_probMat.save("m6_probMat_" + rat+ ".csv", arma::csv_ascii);

    Rcpp::List l = Rcpp::List::create(Rcpp::Named("m1_probMat") = Rcpp::wrap(m1_probMat),
                                      Rcpp::Named("m2_probMat") = Rcpp::wrap(m2_probMat),
                                      Rcpp::Named("m3_probMat") = Rcpp::wrap(m3_probMat),
                                      Rcpp::Named("m4_probMat") = Rcpp::wrap(m4_probMat),
                                      Rcpp::Named("m5_probMat") = Rcpp::wrap(m5_probMat),
                                      Rcpp::Named("m6_probMat") = Rcpp::wrap(m6_probMat));

    std::string filename = "modelProbs_" + rat + ".RData";
    R["l"] = l;
    std::string rCode = "save(l, file='" + filename + "')";
    R.parseEvalQ(rCode.c_str());
    
    return;
}


