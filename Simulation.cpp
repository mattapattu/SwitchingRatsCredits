#include "Simulation.h"
#include "InferStrategy.h"
#include "Pagmoprob.h"
#include "PagmoMle.h"
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/archipelago.hpp>
#include <random>
#include <RInside.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>




RatData generateSimulation(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::map<std::string, std::vector<double>> clusterParams)
{
        //ACA params
    double alpha_aca_subOptimal = params.find(std::make_pair("aca2", false))->second[0];
    double gamma_aca_subOptimal = params.find(std::make_pair("aca2", false))->second[1];

    double alpha_aca_optimal = params.find(std::make_pair("aca2", true))->second[0];
    double gamma_aca_optimal = params.find(std::make_pair("aca2", true))->second[1];

    //ARL params
    double alpha_arl_subOptimal = params.find(std::make_pair("arl", false))->second[0];
    double beta_arl_subOptimal = 1e-7;
    double lambda_arl_subOptimal = params.find(std::make_pair("arl", false))->second[1];
    
    double alpha_arl_optimal = params.find(std::make_pair("arl", true))->second[0];
    double beta_arl_optimal = 1e-7;
    double lambda_arl_optimal = params.find(std::make_pair("arl", true))->second[1];
 
    //DRL params
    double alpha_drl_subOptimal = params.find(std::make_pair("drl", false))->second[0];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = params.find(std::make_pair("drl", false))->second[1];
    
    double alpha_drl_optimal = params.find(std::make_pair("drl", true))->second[0];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = params.find(std::make_pair("drl", true))->second[1];

    std::string rat = ratdata.getRat();
    double crpAlpha = clusterParams.find(rat)->second[0];
    double phi = clusterParams.find(rat)->second[1];
    double eta = 0;

    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, crpAlpha, phi, eta, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, crpAlpha, phi, eta, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, crpAlpha, phi, eta, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, crpAlpha, phi, eta, true);

    auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"arl", alpha_arl_subOptimal, beta_arl_subOptimal, lambda_arl_subOptimal, crpAlpha, phi, eta, false);
    auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"arl",alpha_arl_optimal, beta_arl_optimal, lambda_arl_optimal, crpAlpha, phi, eta, true);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(3, 10);
    int changepoint_ses = distribution(gen);

    arma::mat generated_PathData;
    arma::mat generated_TurnsData;

    std::cout << "Generating sim data with changepoint at ses " <<  changepoint_ses << std::endl;


    for(int ses=0; ses < sessions; ses++)
    {
        std::shared_ptr<Strategy> strategy;
        if(ses < changepoint_ses)
        {
            strategy = aca2_Suboptimal_Hybrid3;

        }else{

            strategy = aca2_Optimal_Hybrid3;

        }

        std::pair<arma::mat, arma::mat> simData = simulateAca2(ratdata, ses, *strategy);
        arma::mat generated_PathData_sess = simData.first;
        arma::mat generated_TurnsData_sess = simData.second;

        generated_PathData = arma::join_cols(generated_PathData, generated_PathData_sess);
        generated_TurnsData = arma::join_cols(generated_TurnsData, generated_TurnsData_sess);
      
    }

    RInside R;

    R["genData"] = Rcpp::wrap(generated_PathData);

    // Save the matrix as RData using RInside
    R.parseEvalQ("saveRDS(genData, file='genData.RData')");



    RatData simRatdata(generated_PathData,generated_TurnsData,rat, true);
    return simRatdata;

}


std::map<std::pair<std::string, bool>, std::vector<double>> findParamsWithSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3)
{
    std::vector<std::string> learningRules = {"aca2","arl", "drl" };
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
            archipelago archi{5u, algo, prob, 7u};

            // 4 - Run the evolution in parallel on the 5 separate islands 5 times.
            archi.evolve(5);
            //std::cout << "DONE1:"  << '\n';
            //system("pause"); 

            // 5 - Wait for the evolutions to finish.
            archi.wait_check();

            // 6 - Print the fitness of the best solution in each island.
            

            //system("pause"); 

            double champion_score = 1000000;
            std::vector<double> dec_vec_champion;
            for (const auto &isl : archi) {
                // std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
                std::vector<double> dec_vec = isl.get_population().champion_x();
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

            std::cout << "Final champion = " << champion_score << std::endl;
            for (auto const& i : dec_vec_champion)
                std::cout << i << ", ";
            std::cout << "\n" ;

            std::pair<std::string, bool> key(lr, optimal);
            paramStrategies[key] = dec_vec_champion;

        }
    }


    return paramStrategies;   
}

std::vector<double> findClusterParamsWithSimData(RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3, std::map<std::pair<std::string, bool>, std::vector<double>>& params)
{
        // Create a function to optimize
    PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3, params);
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
    archipelago archi{5u, algo, prob, 7u};

    // 4 - Run the evolution in parallel on the 5 separate islands 5 times.
    archi.evolve(5);
    std::cout << "DONE1:"  << '\n';
    //system("pause"); 

    // 5 - Wait for the evolutions to finish.
    archi.wait_check();

    // 6 - Print the fitness of the best solution in each island.
    

    //system("pause"); 

    double champion_score = 1000000;
    std::vector<double> dec_vec_champion;
    for (const auto &isl : archi) {
        // std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
        std::vector<double> dec_vec = isl.get_population().champion_x();
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

    std::cout << "Final champion = " << champion_score << std::endl;
    for (auto const& i : dec_vec_champion)
        std::cout << i << ", ";
    std::cout << "\n" ;

    return dec_vec_champion;
}

void runEMOnSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::vector<double> clusterParams, bool debug)
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

    //ARL params
    double alpha_arl_subOptimal = params.find(std::make_pair("arl", false))->second[0];
    double beta_arl_subOptimal = 1e-7;
    double lambda_arl_subOptimal = params.find(std::make_pair("arl", false))->second[1];
    
    double alpha_arl_optimal = params.find(std::make_pair("arl", true))->second[0];
    double beta_arl_optimal = 1e-7;
    double lambda_arl_optimal = params.find(std::make_pair("arl", true))->second[1];
 
    //DRL params
    double alpha_drl_subOptimal = params.find(std::make_pair("drl", false))->second[0];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = params.find(std::make_pair("drl", false))->second[1];
    
    double alpha_drl_optimal = params.find(std::make_pair("drl", true))->second[0];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = params.find(std::make_pair("drl", true))->second[1];

    std::string rat = ratdata.getRat();
    double crpAlpha = clusterParams[0];
    double phi = clusterParams[1];
    double eta = 0;

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

    auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"arl", alpha_arl_subOptimal, beta_arl_subOptimal, lambda_arl_subOptimal, crpAlpha, phi, eta, false);
    auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"arl",alpha_arl_optimal, beta_arl_optimal, lambda_arl_optimal, crpAlpha, phi, eta, true);


    std::vector<std::shared_ptr<Strategy>> strategies;
    strategies.push_back(aca2_Suboptimal_Hybrid3);
    strategies.push_back(aca2_Optimal_Hybrid3);

    strategies.push_back(drl_Suboptimal_Hybrid3);
    strategies.push_back(drl_Optimal_Hybrid3);

    strategies.push_back(arl_Suboptimal_Hybrid3);
    strategies.push_back(arl_Optimal_Hybrid3);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    std::vector<std::string> cluster;
    std::string last_choice;
    
    arma::mat probMat;

    for(int ses=0; ses < sessions; ses++)
    {
        
        if(ses==0)
        {
            initRewardVals(ratdata, ses, strategies, debug);
        }

        arma::mat probMat_sess = estep_cluster_update(ratdata, ses, strategies, cluster, last_choice, true);
        mstep(ratdata, ses, strategies, cluster, debug);

        probMat = arma::join_cols(probMat, probMat_sess);

    }

    arma::mat& aca2_suboptimal_probs =  aca2_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& aca2_optimal_probs =  aca2_Optimal_Hybrid3->getPathProbMat();
    arma::mat& drl_suboptimal_probs =  drl_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& drl_optimal_probs =  drl_Optimal_Hybrid3->getPathProbMat();
    arma::mat& arl_suboptimal_probs =  arl_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& arl_optimal_probs =  arl_Optimal_Hybrid3->getPathProbMat();

    probMat.save("ProbMat_Sim_" + rat+ ".csv", arma::csv_ascii);



    // aca2_suboptimal_probs.save("aca2_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // aca2_optimal_probs.save("aca2_optimal_probs_"+ rat+".csv", arma::csv_ascii);
    // drl_suboptimal_probs.save("drl_suboptimal_probs_"+ rat+".csv", arma::csv_ascii);
    // drl_optimal_probs.save("drl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // arl_suboptimal_probs.save("arl_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // arl_optimal_probs.save("arl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);


}


void testRecovery(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3)
{
    // Read the params from from rat param file, e.g rat_103.txt
    std::string rat = ratdata.getRat();
    std::string filename = rat + ".txt";
    std::ifstream infile(filename);
    std::map<std::pair<std::string, bool>, std::vector<double>> ratParams;
    boost::archive::text_iarchive ia(infile);
    ia >> ratParams;
    infile.close();

    //read clusterParams.txt to get the parameters for rat
    std::string filename_cluster = "clusterParams.txt";
    std::ifstream cluster_infile(filename_cluster);
    std::map<std::string, std::vector<double>> clusterParams;
    boost::archive::text_iarchive ia_cluster(cluster_infile);
    ia_cluster >> clusterParams;
    cluster_infile.close();
  
    RatData ratSimData = generateSimulation(ratdata, suboptimalHybrid3, optimalHybrid3, ratParams,clusterParams);
    std::map<std::pair<std::string, bool>, std::vector<double>> simRatParams = findParamsWithSimData(ratSimData, suboptimalHybrid3, optimalHybrid3);
    std::vector<double> simClusterParams = findClusterParamsWithSimData(ratSimData, suboptimalHybrid3, optimalHybrid3, simRatParams);

    runEMOnSimData(ratdata, suboptimalHybrid3, optimalHybrid3, simRatParams, simClusterParams, true);


}





