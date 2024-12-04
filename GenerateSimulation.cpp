#include "Simulation.h"

std::vector<std::vector<int>> generateStratSeq(RatData& ratdata)
{
    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    std::vector<int> x_cond(sessions, 1);

    std::vector<std::vector<int>> genSequences;
    std::vector<std::pair<int, int>> strat_pair;
    strat_pair.push_back(std::make_pair(0, 1));
    strat_pair.push_back(std::make_pair(0, 3));
    strat_pair.push_back(std::make_pair(2, 1));
    strat_pair.push_back(std::make_pair(2, 3));
    strat_pair.push_back(std::make_pair(1, 1));
    strat_pair.push_back(std::make_pair(3, 3));


    for(int j = 0; j<6; j++)
    {
        std::pair<int,int> strats = strat_pair[j];
        std::vector<int> seq;

        for(int ses=0; ses<sessions; ses++)
        {
            if(ses==0)
            {
                seq.push_back(strats.first); // start with suboptimal policy
            }
            else if(ses > 0 && ses <= 6)
            {
                std::vector<double> p = {0.7,0.3};
                int strat_selected = sample(p);
                if(strat_selected==0)
                {
                   seq.push_back(strats.first); 
                }else{
                    seq.push_back(strats.second); 
                }

            }else{
                seq.push_back(strats.second); 
            }
        }
        genSequences.push_back(seq);
    }
    
    return genSequences;
}

RatData generateSimulatedSequence(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::vector<double> v, std::vector<int> trueGenStrategies, RInside &R, std::string run)
{
    std::string rat = ratdata.getRat();
    // std::vector<double> v = clusterParams[rat]; 
        double alpha_aca_subOptimal = v[0];
    double gamma_aca_subOptimal = v[1];

    double alpha_aca_optimal = v[2];
    double gamma_aca_optimal = v[3];

    //ARL params
    // double alpha_arl_subOptimal = params.find(std::make_pair("arl", false))->second[0];
    // double beta_arl_subOptimal = 1e-7;
    // double lambda_arl_subOptimal = params.find(std::make_pair("arl", false))->second[1];
    
    // double alpha_arl_optimal = params.find(std::make_pair("arl", true))->second[0];
    // double beta_arl_optimal = 1e-7;
    // double lambda_arl_optimal = params.find(std::make_pair("arl", true))->second[1];
 
    //DRL params
    double alpha_drl_subOptimal = v[4];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[5];
    
    double alpha_drl_optimal = v[6];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[7];
    // double phi = 0.1;
    double alpha_crp = v[8];

    
    
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
//    std::vector<int> trueGenStrategies = {0, 3, 0, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
    // trueGenStrategies = {2, 1, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    

    bool endLoopOptimal = false;
    int counterOptimal = 0;
    while(!endLoopOptimal)
    {
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

        if(check_ema(generated_PathData))
        {
            std::cout << "check_ema is successful after " << counterOptimal << " iterations" <<std::endl;
            endLoopOptimal = true;

        }else if(counterOptimal==99)
        {
            endLoopOptimal = true;
        }
        else{
            endLoopOptimal = false;
            //std::cout << "check_ema failed. Re-generate Optimal trajectory: " << counterOptimal <<std::endl;
            generated_PathData.reset();
            generated_TurnsData.reset();

            aca2_Suboptimal_Hybrid3->resetCredits();
            aca2_Optimal_Hybrid3->resetCredits();
            drl_Suboptimal_Hybrid3->resetCredits();
            drl_Optimal_Hybrid3->resetCredits();
            //trueGenStrategies.clear();
        }
        counterOptimal++;

        // if(counterOptimal==100)
        // {
        //     std::cout << "Loop counter reached 100 for simulation. Exiting" << std::endl;
        //     break;

        // }

    }        
    
    
    RatData simRatdata(generated_PathData,generated_TurnsData,rat, true);

    arma::mat simAllpaths = simRatdata.getPaths();
    arma::vec simSessionVec = simAllpaths.col(4);
    arma::vec simUniqSessIdx = arma::unique(simSessionVec);
    // std::cout << "simUniqSessIdx.size=" << simUniqSessIdx.size() << std::endl;

    //testSimulation(simRatdata,*randomPair.first,*randomPair.second, R);
    return simRatdata;

}


arma::mat generateSimSeqWithProbMat(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::vector<double> v, std::vector<int> trueGenStrategies, RInside &R, std::string run)
{
    std::string rat = ratdata.getRat();
    // std::vector<double> v = clusterParams[rat]; 
        double alpha_aca_subOptimal = v[0];
    double gamma_aca_subOptimal = v[1];

    double alpha_aca_optimal = v[2];
    double gamma_aca_optimal = v[3];

    
    //DRL params
    double alpha_drl_subOptimal = v[4];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[5];
    
    double alpha_drl_optimal = v[6];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[7];
    // double phi = 0.1;
    double alpha_crp = v[8];

    
    
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
    arma::mat probMat;
    std::cout << "sessions=" << sessions << std::endl;

    bool endLoopOptimal = false;
    int counterOptimal = 0;
    while(!endLoopOptimal)
    {
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

            arma::mat probMat_sess = strategies[strategy]->getPathProbMatSes(ses); 
            probMat = arma::join_vert(probMat, probMat_sess);

            arma::uvec s0indices = arma::find(generated_PathData_sess.col(1) == 0); 
            arma::mat genDataS0 = generated_PathData_sess.rows(s0indices);

            arma::uvec s1indices = arma::find(generated_PathData_sess.col(1) == 1); 
            arma::mat genDataS1 = generated_PathData_sess.rows(s1indices);
            std::cout << "ses=" << ses << ", strategy=" << strategy << std::endl;

            //trueGenStrategies.push_back(strategy->getName());
            // trueGenStrategies[ses] = strategy->getName();

            generated_PathData = arma::join_cols(generated_PathData,generated_PathData_sess);
            generated_TurnsData = arma::join_cols(generated_TurnsData,generated_TurnsData_sess);


        }

        if(check_ema(generated_PathData))
        {
            std::cout << "check_ema is successful after " << counterOptimal << " iterations" <<std::endl;
            endLoopOptimal = true;

        }else if(counterOptimal==99)
        {
            endLoopOptimal = true;
        }
        else{
            endLoopOptimal = false;
            //std::cout << "check_ema failed. Re-generate Optimal trajectory: " << counterOptimal <<std::endl;
            generated_PathData.reset();
            generated_TurnsData.reset();

            aca2_Suboptimal_Hybrid3->resetCredits();
            aca2_Optimal_Hybrid3->resetCredits();
            drl_Suboptimal_Hybrid3->resetCredits();
            drl_Optimal_Hybrid3->resetCredits();
            //trueGenStrategies.clear();
        }
        counterOptimal++;

        // if(counterOptimal==100)
        // {
        //     std::cout << "Loop counter reached 100 for simulation. Exiting" << std::endl;
        //     break;

        // }

    }        
    
    
    RatData simRatdata(generated_PathData,generated_TurnsData,rat, true);

    arma::mat simAllpaths = simRatdata.getPaths();
    arma::vec simSessionVec = simAllpaths.col(4);
    arma::vec simUniqSessIdx = arma::unique(simSessionVec);
    // std::cout << "simUniqSessIdx.size=" << simUniqSessIdx.size() << std::endl;

    //testSimulation(simRatdata,*randomPair.first,*randomPair.second, R);
    return probMat;

}
