#include "InferStrategy.h"
#include "Pagmoprob.h"
#include "PagmoMle.h"
#include "PagmoMultiObjCluster.h"
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
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>





std::vector<double> computePrior(std::vector<std::shared_ptr<Strategy>>  strategies, std::vector<std::string>& cluster, int ses, std::string& last_choice)
{
    int strategies_size = strategies.size();
    std::vector<double> priors(strategies_size, 0);
    double crpAlpha = strategies[0]->getCrpAlpha();
    double eta = strategies[0]->getEta();
    
    
    if(ses == 0)
    {
        double p = 1.0/(double)strategies_size;
        //std::cout << "ses=0" << ", p=" << p << std::endl;
        std::fill(priors.begin(), priors.end(), p);

    }
    else
    {

        for (size_t i = 0; i < strategies.size(); ++i) {
            // Access the strategy at index i using strategies[i]
            std::shared_ptr<Strategy> strategyPtr = strategies[i];
            std::string name = strategyPtr->getName();
            //If strategy is in cluster, prior is average over past posteriors
            if (std::find(cluster.begin(), cluster.end(), name) != cluster.end())
            {
                std::vector<double> strategy_priors = strategyPtr->getCrpPosteriors();
                double crp_prior = std::accumulate(strategy_priors.begin(), strategy_priors.end(), 0.0);
                
                if (last_choice == name ) {
                    priors[i] = crp_prior + eta;
                    //std::cout << "Adding eta=" <<eta << " to prior for strategy=" << name << std::endl;
                }else{
                    priors[i] = crp_prior;
                }
            }else{ //Else,  strategy is in not cluster, set it to -1 and replace all -1 alpha/N golbally
                priors[i] = -1;
            }

        }
        //Replace all -1's globally by alpha/N, where N is the nb of strategies not in cluster
        double alpha_i = crpAlpha/(strategies_size - cluster.size());
        std::replace_if(priors.begin(), priors.end(), [](int x) { return x == -1; }, alpha_i);

        // std::cout << "Priors before transform: " ;
        // for (const double& p : priors) {
        //     std::cout << p << " ";
        // }
        // std::cout << std::endl;
        
        double sum = std::accumulate(priors.begin(), priors.end(), 0.0);
        //double normalizer = sum;
        std::transform(priors.begin(), priors.end(), priors.begin(), [sum](double x) { return x / sum; });

        
        
    }

    // std::cout << "Priors: " ;
    // for (const double& p : priors) {
    //     std::cout << p << " ";
    // }
    // std::cout << std::endl;

    return(priors);

}

arma::mat estep_cluster_update(const RatData& ratdata, int ses, std::vector<std::shared_ptr<Strategy>>  strategies, std::vector<std::string>& cluster, std::string& last_choice ,bool logger, RecordResults& sessionResults)
{
    std::vector<double> loglikelihoods;
    double max_posterior = 0.0;
    std::vector<std::shared_ptr<Strategy>> most_lik_strategies;


    int strategies_size = strategies.size();
    std::vector<double> posteriors(strategies_size, 0);

    //E-step: 1. Compute priors for all strategies
    std::vector<double> priors_ses = computePrior(strategies, cluster, ses, last_choice);

    arma::mat winningProbMat;
        
    for (size_t i = 0; i < strategies.size(); ++i) 
    {
        std::shared_ptr<Strategy> strategyPtr = strategies[i];
        double crp_posterior = 0.0;
        
        //E-step: 2. Compute likelihoods
        double log_likelihood = strategyPtr->getTrajectoryLikelihood(ratdata, ses); 
        loglikelihoods.push_back(log_likelihood);
        if(logger)
        {
            std::cout << "estep_cluster_update for " << strategyPtr->getName()  << ", ses=" << ses  << ", prior=" << priors_ses[i] << ", log_likelihood=" << log_likelihood << std::endl;  
        }

                
        crp_posterior = exp(log_likelihood) * priors_ses[i];
        
       //std::cout << "estep_cluster_update for " << strategyPtr->getName()  << ", ses=" << ses  << ", posterior=" << crp_posterior << std::endl; 

        posteriors[i] = crp_posterior;

        if (std::isnan(exp(log_likelihood))) {
                    
            std::cout << "exp(log_likelihood) is nan. Check" << std::endl;
            std::exit(EXIT_FAILURE);
        }else if(crp_posterior == 0){
            //std::cout << "Zero posterior prob. Check" << std::endl;
        }
        
    }

    double sum = std::accumulate(posteriors.begin(), posteriors.end(), 0.0);
    if(sum==0)
    {
        sum = 0.1; // To prevent NaN error
    }
    std::transform(posteriors.begin(), posteriors.end(), posteriors.begin(), [sum](double x) { return x / sum; });
    //std::cout << "estep_cluster_update " << ", ses=" << ses  << ", sumPosterior=" << sumPosterior << std::endl; 


    size_t maxIndex = 0;  // Index of the maximum element
    size_t secondMaxIndex = 0;  // Index of the second maximum element
    double maxElement = 0;  // Maximum element value
    double secondMaxElement = 0;  // Second maximum element value

    // Iterate through the vector to find the maximum and second maximum elements
    for (size_t i = 0; i < posteriors.size(); ++i) {
        if (posteriors[i] > maxElement) {
            secondMaxElement = maxElement;
            secondMaxIndex = maxIndex;

            maxElement = posteriors[i];
            maxIndex = i;
        } else if (posteriors[i] > secondMaxElement) {
            secondMaxElement = posteriors[i];
            secondMaxIndex = i;
        }
    }

    // Check if the difference between the maximum and second maximum is greater than 0.1
    if (maxElement - secondMaxElement > 0.05) {
        std::shared_ptr<Strategy> strategyPtr = strategies[maxIndex];
        std::string name = strategyPtr->getName();
        most_lik_strategies.push_back(strategyPtr);
    } else {
        std::shared_ptr<Strategy> strategyPtrMax = strategies[maxIndex];
        std::string maxName = strategyPtrMax->getName();
        most_lik_strategies.push_back(strategyPtrMax);

        std::shared_ptr<Strategy> strategyPtrSecondMax = strategies[secondMaxIndex];
        std::string secondMaxName = strategyPtrSecondMax->getName();
        most_lik_strategies.push_back(strategyPtrSecondMax);
        //std::cout << "The difference between the maximum and second maximum is less than 0.1." << std::endl;
    }


    for(size_t i = 0; i < strategies.size(); ++i)
    {
        std::shared_ptr<Strategy> strategyPtr = strategies[i];
        std::string name = strategyPtr->getName();
        if (std::isnan(posteriors[i])) {
                    
            std::cout << "Crp posteriors is nan. Check" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        //double log_likelihood = strategyPtr->getTrajectoryLikelihood(ratdata, ses); 
        double marginalLik = loglikelihoods[i] * posteriors[i];
        strategyPtr->setMarginalLikelihood(marginalLik);

        strategyPtr->setCrpPosterior(posteriors[i],ses);
        
        if(logger)
        {
            std::cout << "estep_cluster_update for " << strategyPtr->getName()  << ", ses=" << ses  << ", posterior=" << posteriors[i]  << std::endl;   
        }

    }

    if(logger)
    {
        std::cout << "Most likely strategies: " ;
        for (auto strategy : most_lik_strategies) {
            std::cout << strategy->getName() << " ";
        }
        std::cout << std::endl;

    }

    if(most_lik_strategies.size() == 1)
    {
        std::shared_ptr<Strategy> strategyPtrMax = most_lik_strategies[0];
        last_choice = strategyPtrMax->getName();
        winningProbMat = strategyPtrMax->getPathProbMat();
        // Find the indices where the condition is true
        arma::uvec indices = find(winningProbMat.col(13) == ses);

        // Extract the desired columns (1:12) for the rows that satisfy the condition
        winningProbMat = winningProbMat.rows(indices);
        winningProbMat = winningProbMat.cols(arma::span(0, 11));

        // If most_lik_strategy is not in cluster, expand cluster
        if (std::find(cluster.begin(), cluster.end(), last_choice) == cluster.end()) {
            // most_lik_strategy is not present, include it
            cluster.push_back(last_choice);
            // std::cout <<"ses=" << ses << ", Adding " << most_lik_strategy <<" to cluster." << std::endl; 
            if(logger)
            {
                std::cout <<"ses=" << ses << ", Adding " << last_choice <<" to cluster." << std::endl;   
            }
        }  

    }else
    {
        last_choice = "None";
        if(logger)
        {
            std::cout <<"Not adding any strategies to cluster as multiple strategies have same posterior" << std::endl; 
        }
    }

    for(const auto& strategyPtr : strategies)
    {
        auto it = std::find(cluster.begin(), cluster.end(), strategyPtr->getName());
        if (it == cluster.end()) {
            strategyPtr->resetCredits();
            //strategyPtr->initRewards(ratdata);
            //strategyPtr->setCrpPosterior(0,ses);
        }  
    }

    // std::cout <<"ses=" << ses << ", most_lik_strategy in estep_cluster_update = " << most_lik_strategy << std::endl; 
    // if(logger)
    // {
    //     std::cout <<"ses=" << ses << ", most_lik_strategy in estep_cluster_update = " << most_lik_strategy << std::endl;    
    // }

    if(logger)
    {
        std::cout << "Cluster: " ;
        for (const std::string& name : cluster) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
    }

    if(ratdata.getSim())
    {
        if(most_lik_strategies.size() == 1)
        {
            sessionResults.setSelectedStrategy(last_choice);
            sessionResults.setProbabilityMatrix(winningProbMat);
            
        }
        else
        {
            sessionResults.setSelectedStrategy("None");
        }

        for(size_t i = 0; i < strategies.size(); ++i)
        {
            std::shared_ptr<Strategy> strategyPtr = strategies[i];
            std::string name = strategyPtr->getName();
            sessionResults.setPosteriors(name, posteriors[i]);
        }
        
    }
    

    return winningProbMat;
}

void mstep(const RatData& ratdata, int ses, std::vector<std::shared_ptr<Strategy>> strategies, std::vector<std::string>& cluster, bool logger, RecordResults& sessionResults)
{
    for (const auto& strategyPtr : strategies) {

        //std::cout << "m-step for " << strategyPtr->getName()  << ", ses=" << ses << std::endl; 
        //M-step
        //std::pair<std::vector<double>, std::vector<double>> rewardUpdates = getRewardUpdates(ratdata, ses);
        
        std::string strategy = strategyPtr->getName();
        strategyPtr->updateRewards(ratdata, ses); 
        std::vector<double> rewardsS0 = strategyPtr->getRewardsS0();
        std::vector<double> rewardsS1 = strategyPtr->getRewardsS1();

        if(logger)
        {
            std::cout << strategy << " rewardsS0: " ;
            for (const double& reward : rewardsS0) {
                std::cout << reward << " ";
            }
            std::cout << std::endl;

            std::cout << strategy << " rewardsS1: " ;
            for (const double& reward : rewardsS1) {
                std::cout << reward << " ";
            }
            std::cout << std::endl;
        }

        //Reset posteriors
        if (std::find(cluster.begin(), cluster.end(), strategy) == cluster.end()) {
            //std::cout << "Setting crpPosterior to 0 for strategy=" << strategyPtr->getName()  << ", ses=" << ses << std::endl; 
            strategyPtr->setCrpPosterior(0,ses);
        }   

        sessionResults.setRewardVectorS0(strategy, rewardsS0);
        sessionResults.setRewardVectorS1(strategy, rewardsS1);
  
    }

    return;
}


void initRewardVals(const RatData& ratdata, int ses, std::vector<std::shared_ptr<Strategy>> strategies, bool logger)
{
    for (const auto& strategyPtr : strategies) {

       //std::cout << "initRewardVals for " << strategyPtr->getName()  << ", ses=" << ses << std::endl; 
        //M-step
        //std::pair<std::vector<double>, std::vector<double>> rewardUpdates = getRewardUpdates(ratdata, ses);
        strategyPtr->initRewards(ratdata); 
        std::string strategy = strategyPtr->getName();
        std::vector<double> rewardsS0 = strategyPtr->getRewardsS0();
        std::vector<double> rewardsS1 = strategyPtr->getRewardsS1();

        if(logger)
        {
            std::cout << strategy << " rewardsS0: " ;
            for (const double& reward : rewardsS0) {
                std::cout << reward << " ";
            }
            std::cout << std::endl;

            std::cout << strategy << " rewardsS1: " ;
            for (const double& reward : rewardsS1) {
                std::cout << reward << " ";
            }
            std::cout << std::endl;
        }
    }
    
    return;
}


void findClusterParams(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params) {

    // Open the file for reading and appending
    std::string filename_cluster = "clusterParams.txt";
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
    PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3, params);
    //PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
    std::cout << "Initialized problem class" <<std::endl;

    // Create a problem using Pagmo
    problem prob{pagmoprob};
    //problem prob{schwefel(30)};
    
    std::cout << "created problem" <<std::endl;
    // 2 - Instantiate a pagmo algorithm (self-adaptive differential
    // evolution, 100 generations).
    pagmo::algorithm algo{sade(10,2,2)};

    std::cout << "creating archipelago" <<std::endl;
    // 3 - Instantiate an archipelago with 5 islands having each 5 individuals.
    // archipelago archi{5u, algo, prob, 7u};

    // // 4 - Run the evolution in parallel on the 5 separate islands 5 times.
    // archi.evolve(5);
    // std::cout << "DONE1:"  << '\n';
    // //system("pause"); 

    // // 5 - Wait for the evolutions to finish.
    // archi.wait_check();

    // // 6 - Print the fitness of the best solution in each island.
    

    // //system("pause"); 

    // double champion_score = 1000000;
    // std::vector<double> dec_vec_champion;
    // for (const auto &isl : archi) {
    //     // std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
    //     std::vector<double> dec_vec = isl.get_population().champion_x();
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

    island isl{algo, prob, 35u};

    // 4 - Run the evolution on the single island 5 times.
    isl.evolve(5);
    //std::cout << "DONE1:" << '\n';

    // 5 - Wait for the evolution to finish.
    isl.wait_check();

    // 6 - Print the fitness of the best solution in the island.
    std::vector<double> dec_vec_champion = isl.get_population().champion_x();
    double champion_score = isl.get_population().champion_f()[0];

    // std::cout << "Final champion = " << champion_score << std::endl;
    // for (auto const& i : dec_vec_champion)
    //     std::cout << i << ", ";
    // std::cout << "\n" ;

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

void findMultiObjClusterParams(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, const std::map<std::pair<std::string, bool>, std::vector<double>>& params) {

    // Open the file for reading and appending
    std::string filename_cluster = "clusterParams.txt";
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
    PagmoMultiObjCluster pagmoMultiObjProb(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3, params);
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
    pagmo::population pop { prob, thread_bfe, 28};
   

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
        std::cout << "Fitness: [" << f[idx[i]][0] << ", " << f[idx[i]][1] <<  ", " << f[idx[i]][2] << ", " << f[idx[i]][3] << ", " << f[idx[i]][4] << ", " << f[idx[i]][5] << "]" << ", lik=" << lik << std::endl;
        //std::cout << "Decision vector: [" << x[idx[i]][0] << "]" << std::endl;
        //std::cout << "Crowding distance: " << cd[idx[i]] << std::endl;

        std::vector<double> dec_vec = x[idx[i]];

        std::cout << "dec_vec: ";
        for (const auto& val : dec_vec) {
            std::cout << ", " << val ;
        }

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


void findParams(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3)
{
    
    //std::vector<std::string> learningRules = {"aca2","arl", "drl" };
    std::vector<std::string> learningRules = {"aca2","drl"};
    std::vector<bool> mazeModels = {true, false };

    std::map<std::pair<std::string, bool>, std::vector<double>> paramStrategies;


    for (const auto &lr : learningRules) 
    {
        for (const auto &optimal : mazeModels) 
        {
            std::string learningRule =  lr;   
            MazeGraph* maze;
            if(optimal)
            {
                maze = &optimalHybrid3;
            }else
            {
                maze = &suboptimalHybrid3;
            }

            std::cout << "learningRule=" << lr << ", optimal=" << optimal << std::endl;
            
            PagmoMle pagmoMle(ratdata, *maze, learningRule, optimal);
            //std::cout << "strategy=" << strategy.getName() <<std::endl;

            // Create a problem using Pagmo
            problem prob{pagmoMle};
            //problem prob{schwefel(30)};
            
            std::cout << "created problem" <<std::endl;
            // 2 - Instantiate a pagmo algorithm (self-adaptive differential
            // evolution, 100 generations).
            pagmo::algorithm algo{sade(10,2,2)};
            //algo.set_verbosity(10);

            //pagmo::algorithm algo{de(10)};

            std::cout << "creating archipelago" <<std::endl;
            // 3 - Instantiate an archipelago with 5 islands having each 5 individuals.
            // archipelago archi{5u, algo, prob, 7u};

            // // 4 - Run the evolution in parallel on the 5 separate islands 5 times.
            // archi.evolve(5);
            // //std::cout << "DONE1:"  << '\n';
            // //system("pause"); 

            // // 5 - Wait for the evolutions to finish.
            // archi.wait_check();

            // // 6 - Print the fitness of the best solution in each island.
            

            // //system("pause"); 

            // double champion_score = 1000000;
            // std::vector<double> dec_vec_champion;
            // for (const auto &isl : archi) {
            //     // std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
            //     std::vector<double> dec_vec = isl.get_population().champion_x();
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

            island isl{algo, prob, 35u};

            // 4 - Run the evolution on the single island 5 times.
            isl.evolve(5);
            //std::cout << "DONE1:" << '\n';

            // 5 - Wait for the evolution to finish.
            isl.wait_check();

            // 6 - Print the fitness of the best solution in the island.
            std::vector<double> dec_vec_champion = isl.get_population().champion_x();
            double champion_score = isl.get_population().champion_f()[0];

            std::pair<std::string, bool> key(lr, optimal);
            paramStrategies[key] = dec_vec_champion;

        }
    }
    
    std::string rat = ratdata.getRat();
    std::string filename = rat + ".txt"; 
    std::ofstream file(filename);
    boost::archive::text_oarchive oa(file);
    oa << paramStrategies;
    file.close();

    return;
}


std::vector<RecordResults> runEM(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::map<std::string, std::vector<double>> clusterParams, bool debug)
{
    //// rat_103
    //std::vector<double> v = {0.11776, 0.163443, 0.0486187, 1e-7,0.475538, 0.272467, 1e-7 , 0.0639478, 1.9239e-06, 0.993274, 4.3431};
    
    ////rat_114
    //std::vector<double> v = {0.0334664, 0.351993, 0.00478871, 1.99929e-07, 0.687998, 0.380462, 9.68234e-07, 0.136651, 8.71086e-06, 0.292224, 3.95355};

    //ACA params
    double alpha_aca_subOptimal = params.find(std::make_pair("aca2", false))->second[0];
    double gamma_aca_subOptimal = params.find(std::make_pair("aca2", false))->second[1];

    double alpha_aca_optimal = params.find(std::make_pair("aca2", true))->second[0];
    double gamma_aca_optimal = params.find(std::make_pair("aca2", true))->second[1];

    // COMMENTING OUT ARL
    //ARL params
    // double alpha_arl_subOptimal = params.find(std::make_pair("arl", false))->second[0];
    // double beta_arl_subOptimal = params.find(std::make_pair("arl", false))->second[1];
    // double lambda_arl_subOptimal = 0;
    
    // double alpha_arl_optimal = params.find(std::make_pair("arl", true))->second[0];
    // double beta_arl_optimal = params.find(std::make_pair("arl", true))->second[1];
    // double lambda_arl_optimal = 0;
 
    //DRL params
    double alpha_drl_subOptimal = params.find(std::make_pair("drl", false))->second[0];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = params.find(std::make_pair("drl", false))->second[1];
    
    double alpha_drl_optimal = params.find(std::make_pair("drl", true))->second[0];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = params.find(std::make_pair("drl", true))->second[1];

    
    // double crpAlpha = clusterParams.find(rat)->second[0];
    // double phi = clusterParams.find(rat)->second[1];
    // double eta = 0;

    std::string rat = ratdata.getRat();
    std::vector<double>& v = clusterParams[rat];

    // double alpha_aca_subOptimal = v[0];
    // double gamma_aca_subOptimal = v[1];

    // double alpha_aca_optimal = v[2];
    // double gamma_aca_optimal = v[3];

    // //ARL params
    // double alpha_arl_subOptimal = v[4];
    // double beta_arl_subOptimal = 1e-7;
    // double lambda_arl_subOptimal = v[5];
    
    // double alpha_arl_optimal = v[6];
    // double beta_arl_optimal = 1e-7;
    // double lambda_arl_optimal = v[7];
 
    // //DRL params
    // double alpha_drl_subOptimal = v[8];
    // double beta_drl_subOptimal = 1e-4;
    // double lambda_drl_subOptimal = v[9];
    
    // double alpha_drl_optimal = v[10];
    // double beta_drl_optimal = 1e-4;
    // double lambda_drl_optimal = v[11];

    
    double crpAlpha = v[0];
    double phi = v[1];
    double eta = 0;

    // double rS0_subopt = v[1];
    // double rS1_subopt = v[2];
    // double rS0_opt = v[1];
    // double rS1_opt = v[2];
    
    // double rS0_aca = v[1];
    // double rS1_aca = v[2];
    // double rS0_arl = v[3];
    // double rS1_arl = v[4];
    // double rS0_drl = v[5];
    // double rS1_drl = v[6];

    // double rS0_subopt_aca = v[7];
    // double rS1_subopt_aca = v[8];
    // double rS0_subopt_arl = v[9];
    // double rS1_subopt_arl = v[10];
    // double rS0_subopt_drl = v[11];
    // double rS1_subopt_drl = v[12];

    

    //double phi = 0; //phi is replaced by phi_aca...

    // std::vector<double> rewardsS0_aca = {0,0,0,0,0,0,0,rS0_opt,0};
    // std::vector<double> rewardsS1_aca = {0,0,0,0,0,0,0,0,rS1_opt};

    // std::vector<double> rewardsS0_arl = {0,0,0,0,0,0,0,rS0_opt,0};
    // std::vector<double> rewardsS1_arl = {0,0,0,0,0,0,0,0,rS1_opt};
    
    // std::vector<double> rewardsS0_drl = {0,0,0,0,0,0,0,rS0_opt,0};
    // std::vector<double> rewardsS1_drl = {0,0,0,0,0,0,0,0,rS1_opt};
  
    // std::vector<double> rewardsS0_subopt_aca = {0,0,0,0,0,0,rS0_opt,rS1_opt,0,0,0,0};
    // std::vector<double> rewardsS0_subopt_arl = {0,0,0,0,0,0,rS0_opt,rS1_opt,0,0,0,0};
    // std::vector<double> rewardsS0_subopt_drl = {0,0,0,0,0,0,rS0_opt,rS1_opt,0,0,0,0};


    if(debug)
    {
        std::cout << "alpha_aca_subOptimal=" << alpha_aca_subOptimal << ", gamma_aca_subOptimal=" << gamma_aca_subOptimal << ", alpha_aca_optimal=" << alpha_aca_optimal << ", gamma_aca_optimal=" << gamma_aca_optimal << std::endl;
        std::cout << "rat=" << rat << ", crpAlpha=" << crpAlpha << ", phi=" << phi << ", eta=" <<eta << std::endl;
    }
    
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, crpAlpha, phi, eta, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, crpAlpha, phi, eta, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, crpAlpha, phi, eta, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, crpAlpha, phi, eta, true);

    // COMMENTING OUT ARL
    // auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"arl", alpha_arl_subOptimal, beta_arl_subOptimal, lambda_arl_subOptimal, crpAlpha, phi, eta, false);
    // auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"arl",alpha_arl_optimal, beta_arl_optimal, lambda_arl_optimal, crpAlpha, phi, eta, true);

    // aca2_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_aca);
    // drl_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_drl);
    // arl_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_arl);
    
    
    // aca2_Optimal_Hybrid3->setRewardsS0(rewardsS0_aca); 
    // aca2_Optimal_Hybrid3->setRewardsS1(rewardsS1_aca);

    // drl_Optimal_Hybrid3->setRewardsS0(rewardsS0_drl); 
    // drl_Optimal_Hybrid3->setRewardsS1(rewardsS1_drl);

    // arl_Optimal_Hybrid3->setRewardsS0(rewardsS0_arl); 
    // arl_Optimal_Hybrid3->setRewardsS1(rewardsS1_arl);   

    // aca2_Suboptimal_Hybrid3->setPhi(phi_acasub);
    // arl_Suboptimal_Hybrid3->setPhi(phi_arlsub);
    // drl_Suboptimal_Hybrid3->setPhi(phi_drlsub);
    // aca2_Optimal_Hybrid3->setPhi(phi_acaopt);
    // arl_Optimal_Hybrid3->setPhi(phi_arlopt);
    // drl_Optimal_Hybrid3->setPhi(phi_drlopt); 


    std::vector<std::shared_ptr<Strategy>> strategies;
    strategies.push_back(aca2_Suboptimal_Hybrid3);
    strategies.push_back(aca2_Optimal_Hybrid3);

    strategies.push_back(drl_Suboptimal_Hybrid3);
    strategies.push_back(drl_Optimal_Hybrid3);

    // COMMENTING OUT ARL
    // strategies.push_back(arl_Suboptimal_Hybrid3);
    // strategies.push_back(arl_Optimal_Hybrid3);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    std::vector<std::string> cluster;
    std::string last_choice;
    std::vector<std::string> most_likely;
    
    arma::mat probMat;

    // RecordResults allResults("None", {}, {}, {}, {}, {});
    std::vector<RecordResults> allSesResults;

    for(int ses=0; ses < sessions; ses++)
    {
        
        if(ses==0)
        {
            initRewardVals(ratdata, ses, strategies, debug);
        }

        RecordResults sessionResults;

        arma::mat probMat_sess = estep_cluster_update(ratdata, ses, strategies, cluster, last_choice, debug, sessionResults);
        mstep(ratdata, ses, strategies, cluster, debug, sessionResults);

        probMat = arma::join_cols(probMat, probMat_sess);
        most_likely.push_back(last_choice);

        allSesResults.push_back(sessionResults);
    }

    if(debug)
    {
        for(int ses=0; ses < sessions; ses++)
        {
            std::cout << "session " << ses << ", most_likely=" << most_likely[ses] << std::endl;
        }
    }
    

    arma::mat& aca2_suboptimal_probs =  aca2_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& aca2_optimal_probs =  aca2_Optimal_Hybrid3->getPathProbMat();
    arma::mat& drl_suboptimal_probs =  drl_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& drl_optimal_probs =  drl_Optimal_Hybrid3->getPathProbMat();
    
    // COMMENTING OUT ARL
    // arma::mat& arl_suboptimal_probs =  arl_Suboptimal_Hybrid3->getPathProbMat();
    // arma::mat& arl_optimal_probs =  arl_Optimal_Hybrid3->getPathProbMat();

    probMat.save("ProbMat_" + rat+ ".csv", arma::csv_ascii);

    // aca2_suboptimal_probs.save("aca2_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // aca2_optimal_probs.save("aca2_optimal_probs_"+ rat+".csv", arma::csv_ascii);
    // drl_suboptimal_probs.save("drl_suboptimal_probs_"+ rat+".csv", arma::csv_ascii);
    // drl_optimal_probs.save("drl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // arl_suboptimal_probs.save("arl_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // arl_optimal_probs.save("arl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);
    return allSesResults;
}


void runEM2(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::string, std::vector<double>> clusterParams, bool debug)
{
    //// rat_103
    //std::vector<double> v = {0.11776, 0.163443, 0.0486187, 1e-7,0.475538, 0.272467, 1e-7 , 0.0639478, 1.9239e-06, 0.993274, 4.3431};
    
    ////rat_114
    //std::vector<double> v = {0.0334664, 0.351993, 0.00478871, 1.99929e-07, 0.687998, 0.380462, 9.68234e-07, 0.136651, 8.71086e-06, 0.292224, 3.95355};

    //std::vector<double> v = {0.727389, 0.866576, 0.843969, 0.037089, 0.987229, 0.996802, 0.0137021, 0.815637, 0.996112, 0.0413516, 0.0373898, 0.876589, 4.63767, 4.87372};

    std::string rat = ratdata.getRat();
    std::vector<double>& v = clusterParams[rat];

    double crpAlpha = v[0];
    double phi = v[1];
    double eta = 0;

    double alpha_aca_subOptimal = v[2];
    double gamma_aca_subOptimal = v[3];

    double alpha_aca_optimal = v[4];
    double gamma_aca_optimal = v[5];

    // COMMENTING OUT ARL
    //ARL params
    // double alpha_arl_subOptimal = v[6];
    // double beta_arl_subOptimal = v[7];
    // double lambda_arl_subOptimal = 0;
    
    // double alpha_arl_optimal = v[8];
    // double beta_arl_optimal = v[9];
    // double lambda_arl_optimal = 0;
 
    //DRL params
    double alpha_drl_subOptimal = v[10];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[11];
    
    double alpha_drl_optimal = v[12];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[13];
       
    // std::vector<double> rewardsS0_aca = {0,0,0,0,0,0,0,rS0_opt,0};
    // std::vector<double> rewardsS1_aca = {0,0,0,0,0,0,0,0,rS1_opt};

    // std::vector<double> rewardsS0_arl = {0,0,0,0,0,0,0,rS0_opt,0};
    // std::vector<double> rewardsS1_arl = {0,0,0,0,0,0,0,0,rS1_opt};
    
    // std::vector<double> rewardsS0_drl = {0,0,0,0,0,0,0,rS0_opt,0};
    // std::vector<double> rewardsS1_drl = {0,0,0,0,0,0,0,0,rS1_opt};
  
    // std::vector<double> rewardsS0_subopt_aca = {0,0,0,0,0,0,rS0_subopt,rS1_subopt,0,0,0,0};
    // std::vector<double> rewardsS0_subopt_arl = {0,0,0,0,0,0,rS0_subopt,rS1_subopt,0,0,0,0};
    // std::vector<double> rewardsS0_subopt_drl = {0,0,0,0,0,0,rS0_subopt,rS1_subopt,0,0,0,0};


    //std::cout << "alpha_aca_subOptimal=" << alpha_aca_subOptimal << ", gamma_aca_subOptimal=" << gamma_aca_subOptimal << ", alpha_aca_optimal=" << alpha_aca_optimal << ", gamma_aca_optimal=" << gamma_aca_optimal << std::endl;
    
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, crpAlpha, phi, eta, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, crpAlpha, phi, eta, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, crpAlpha, phi, eta, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, crpAlpha, phi, eta, true);

    // COMMENTING OUT ARL
    // auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"arl", alpha_arl_subOptimal, beta_arl_subOptimal, lambda_arl_subOptimal, crpAlpha, phi, eta, false);
    // auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"arl",alpha_arl_optimal, beta_arl_optimal, lambda_arl_optimal, crpAlpha, phi, eta, true);

    // aca2_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_aca);
    // drl_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_drl);
    // arl_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_arl);
    
    // aca2_Optimal_Hybrid3->setRewardsS0(rewardsS0_aca); 
    // aca2_Optimal_Hybrid3->setRewardsS1(rewardsS1_aca);

    // drl_Optimal_Hybrid3->setRewardsS0(rewardsS0_drl); 
    // drl_Optimal_Hybrid3->setRewardsS1(rewardsS1_drl);

    // arl_Optimal_Hybrid3->setRewardsS0(rewardsS0_arl); 
    // arl_Optimal_Hybrid3->setRewardsS1(rewardsS1_arl);   

    // aca2_Suboptimal_Hybrid3->setPhi(phi_acasub);
    // arl_Suboptimal_Hybrid3->setPhi(phi_arlsub);
    // drl_Suboptimal_Hybrid3->setPhi(phi_drlsub);
    // aca2_Optimal_Hybrid3->setPhi(phi_acaopt);
    // arl_Optimal_Hybrid3->setPhi(phi_arlopt);
    // drl_Optimal_Hybrid3->setPhi(phi_drlopt);

    
    std::vector<std::shared_ptr<Strategy>> strategies;

    strategies.push_back(aca2_Suboptimal_Hybrid3);
    strategies.push_back(aca2_Optimal_Hybrid3);

    strategies.push_back(drl_Suboptimal_Hybrid3);
    strategies.push_back(drl_Optimal_Hybrid3);

    // COMMENTING OUT ARL
    // strategies.push_back(arl_Suboptimal_Hybrid3);
    // strategies.push_back(arl_Optimal_Hybrid3);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    std::vector<std::string> cluster;
    std::vector<std::string> most_likely;
    std::string last_choice;
    
    arma::mat probMat;

    RecordResults allResults("None", {}, {}, {}, {}, {});

    for(int ses=0; ses < sessions; ses++)
    {
        
        if(ses==0)
        {
            initRewardVals(ratdata, ses, strategies, debug);
        }

        arma::mat probMat_sess = estep_cluster_update(ratdata, ses, strategies, cluster, last_choice, true, allResults);
        mstep(ratdata, ses, strategies, cluster, debug, allResults);

        probMat = arma::join_cols(probMat, probMat_sess);
        most_likely.push_back(last_choice);
    }

    for(int ses=0; ses < sessions; ses++)
    {
        std::cout << "session " << ses << ", most_likely=" << most_likely[ses] << std::endl;
    }

    

    arma::mat& aca2_suboptimal_probs =  aca2_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& aca2_optimal_probs =  aca2_Optimal_Hybrid3->getPathProbMat();
    arma::mat& drl_suboptimal_probs =  drl_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& drl_optimal_probs =  drl_Optimal_Hybrid3->getPathProbMat();
    
    // COMMENTING OUT ARL
    // arma::mat& arl_suboptimal_probs =  arl_Suboptimal_Hybrid3->getPathProbMat();
    // arma::mat& arl_optimal_probs =  arl_Optimal_Hybrid3->getPathProbMat();

    probMat.save("ProbMat_" + rat+ ".csv", arma::csv_ascii);

    // aca2_suboptimal_probs.save("aca2_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // aca2_optimal_probs.save("aca2_optimal_probs_"+ rat+".csv", arma::csv_ascii);
    // drl_suboptimal_probs.save("drl_suboptimal_probs_"+ rat+".csv", arma::csv_ascii);
    // drl_optimal_probs.save("drl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // arl_suboptimal_probs.save("arl_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // arl_optimal_probs.save("arl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);

}


void testLogLik(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3)
{
    std::vector<double> v = {4.43e-2, 4.71e-6};
    double alpha = v[0];
    double gamma = v[1];
    double lambda = 0;

    double crpAlpha = 0;
    double phi = 0;
    double eta = 0;
    Strategy strategy(optimalHybrid3,"arl",alpha, gamma, lambda, crpAlpha, phi, eta, true);
    std::cout << "strategy=" << strategy.getName() <<std::endl;    

    
    // if(strategy.getName() == "aca2_Suboptimal_Hybrid3" || strategy.getName() == "aca2_Optimal_Hybrid3")
    // {
    //     gamma = v[1];
    //     lambda = 0;

    // }else
    // {
    //     // gamma = 1e-7;
    //     // lambda = v[1];
    //     gamma = v[1];
    // }

    std::vector<double> s0rewards = {0,0,0,0,0,0,0,5,0};
    std::vector<double> s1rewards = {0,0,0,0,0,0,0,0,5};
    strategy.setRewardsS0(s0rewards);
    strategy.setRewardsS1(s1rewards);

    strategy.setAlpha(alpha);
    strategy.setGamma(gamma);
    strategy.setLambda(lambda);


    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    double loglikelihood = 0;

    for(int ses=0; ses < sessions; ses++)
    {
        
        double log_likelihood_ses = strategy.getTrajectoryLikelihood(ratdata, ses); 
        loglikelihood = loglikelihood + log_likelihood_ses;
        std::cout << "alpha=" <<alpha << ", gamma=" << gamma << ", lambda=" << lambda << ", ses=" << ses << ", loglikelihood=" << loglikelihood << std::endl;

    }

    loglikelihood = loglikelihood * (-1);

    std::cout << "Using fitness function" << std::endl;
    
    strategy.resetCredits();
      
    // PagmoMle pagmoMle(ratdata,optimalHybrid3, "aca2", true);

    // std::vector<double> ret =  pagmoMle.fitness(v);

 
    std::cout << "alpha=" <<alpha << ", gamma=" << gamma << ", lambda=" << lambda <<  ", loglikelihood=" << loglikelihood  << std::endl;
    
    //return{loglikelihood};
    return;
}
