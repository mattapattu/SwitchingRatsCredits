#include "Simulation.h"

RatData generateSimulatedSequence(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::vector<double> v, RInside &R, std::string run)
{
    std::string rat = ratdata.getRat();
    // std::vector<double> v = clusterParams[rat]; 
        double alpha_aca_subOptimal = v[0];
    double gamma_aca_subOptimal = v[1];

    double alpha_aca_optimal = v[0];
    double gamma_aca_optimal = v[1];

    //ARL params
    // double alpha_arl_subOptimal = params.find(std::make_pair("arl", false))->second[0];
    // double beta_arl_subOptimal = 1e-7;
    // double lambda_arl_subOptimal = params.find(std::make_pair("arl", false))->second[1];
    
    // double alpha_arl_optimal = params.find(std::make_pair("arl", true))->second[0];
    // double beta_arl_optimal = 1e-7;
    // double lambda_arl_optimal = params.find(std::make_pair("arl", true))->second[1];
 
    //DRL params
    double alpha_drl_subOptimal = v[2];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[3];
    
    double alpha_drl_optimal = v[2];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[3];
    // double phi = 0.1;
    double alpha_crp = v[4];

    
    
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, 0, 0, 0, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, 0, 0, 0, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, 0, 0, 0, true);

    std::vector<std::shared_ptr<Strategy>> strategies;
    strategies.push_back(aca2_Suboptimal_Hybrid3);
    strategies.push_back(aca2_Optimal_Hybrid3);

    strategies.push_back(drl_Suboptimal_Hybrid3);
    strategies.push_back(drl_Optimal_Hybrid3);

    
    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    

    arma::mat generated_PathData;
    arma::mat generated_TurnsData;
    arma::mat genProbMat;
    
    // To store vector of true generatorStrateies
   std::vector<int> trueGenStrategies = {1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1};
    std::cout << "trueGenStrategies: ";
    for (const auto &x : trueGenStrategies)
    {
        std::cout << x << " ";
    }
    std::cout << "\n";
            
    for(int ses=0; ses < sessions; ses++)
    {
        
        std::pair<arma::mat, arma::mat> simData;
        arma::mat generated_PathData_sess;
        arma::mat generated_TurnsData_sess;
        
        //Start suboptimal portion of switching simulations
        int strategy = trueGenStrategies[ses];
        simData = simulateTrajectory(ratdata, ses, *strategies[strategy]);
        generated_PathData_sess = simData.first;
        generated_TurnsData_sess = simData.second;


        arma::uvec s0indices = arma::find(generated_PathData_sess.col(1) == 0); 
        arma::mat genDataS0 = generated_PathData_sess.rows(s0indices);

        arma::uvec s1indices = arma::find(generated_PathData_sess.col(1) == 1); 
        arma::mat genDataS1 = generated_PathData_sess.rows(s1indices);
        // std::cout << "ses=" << ses << ", strategy=" << strategy->getName() << std::endl;

        //trueGenStrategies.push_back(strategy->getName());
        // trueGenStrategies[ses] = strategy->getName();

        generated_PathData = arma::join_cols(generated_PathData,generated_PathData_sess);
        generated_TurnsData = arma::join_cols(generated_TurnsData,generated_TurnsData_sess);


    }






    Rcpp::List l = Rcpp::List::create(Rcpp::Named("genData") = Rcpp::wrap(generated_PathData),
                                      Rcpp::Named("probMat") = genProbMat );

    R["l"] = l;
    // Save the matrix as RData using RInside
    std::string filename = "generatedData_" + rat + "_" + run +".RData";
    
    std::string rCode = "save(l, file='" + filename + "')";
    R.parseEvalQ(rCode.c_str());

    // arma::mat trueProbMat = arma::join_cols(drl_Suboptimal_Hybrid3->getPathProbMat(),drl_Optimal_Hybrid3->getPathProbMat());
    // trueProbMat.save("genTrueProbMat_" + rat+ ".csv", arma::csv_ascii);
    
    RatData simRatdata(generated_PathData,generated_TurnsData,rat, true);

    arma::mat simAllpaths = simRatdata.getPaths();
    arma::vec simSessionVec = simAllpaths.col(4);
    arma::vec simUniqSessIdx = arma::unique(simSessionVec);
    // std::cout << "simUniqSessIdx.size=" << simUniqSessIdx.size() << std::endl;

    //testSimulation(simRatdata,*randomPair.first,*randomPair.second, R);
    return simRatdata;

}
