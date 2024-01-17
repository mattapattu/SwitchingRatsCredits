#include "InferStrategy.h"
#include "Pagmoprob.h"
#include "PagmoMle.h"
#include "Simulation.h"
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
#include <mpi.h>
#include <filesystem>
#include <fstream>






// Define a function to compute the EMA of rewards for a given state
arma::vec ema_rewards(arma::mat data, int state) {
  double alpha = 0.3;
  int n_rows = data.n_rows;
  arma::vec ema(n_rows, arma::fill::zeros);
  double ema_prev = 0;
  
  for (int i = 0; i < n_rows; i++) {
    int s = data(i, 1);
    double r = data(i, 2);
    
    // Cap the reward at 1
    r = (r > 1) ? 1 : r;
    
    double ema_curr = alpha * r + (1 - alpha) * ema_prev;
    ema(i) = ema_curr;
    ema_prev = ema_curr;
  }
  
  return ema;
}


// Define a function to check if the EMA is greater than or equal to a threshold for at least a consecutive_count number of rows
bool check_ema(arma::mat data, double threshold = 0.8, int consecutive_count = 5) {
  arma::mat dataS0 = data.rows(find(data.col(1) == 0));
  arma::mat dataS1 = data.rows(find(data.col(1) == 1));
  
  arma::vec ema0 = ema_rewards(dataS0, 0);
  arma::vec ema1 = ema_rewards(dataS1, 1);
  
  int count0 = 0;
  int count1 = 0;
  bool s0 = false;
  bool s1 = false;
  
  for (int i = 0; i < ema0.n_elem; i++) {
    if (ema0(i) >= threshold) {
      count0++;
      if (count0 == consecutive_count) {
        s0 = true;
        break;
      }
    } else {
      count0 = 0;
    }
  }
  
  count1 = 0;
  
  for (int i = 0; i < ema1.n_elem; i++) {
    if (ema1(i) >= threshold) {
      count1++;
      if (count1 == consecutive_count) {
        s1 = true;
        break;
      }
    } else {
      count1 = 0;
    }
  }
  
  if (s0 && s1) {
    return true;
  }
  
  return false;
}



// Define a function to compute the EMA of occurrence of Path5 for a given state
arma::vec ema_path5(arma::mat data) {
  
  double alpha = 0.2;
  // Get the number of rows
  int n_rows = data.n_rows;
  // Initialize the EMA vector
  arma::vec ema(n_rows, arma::fill::zeros);
  // Initialize the previous EMA value
  double ema_prev = 0;
  // Loop over the rows
  for (int i = 0; i < n_rows; i++) {
    // Get the current path
    int p = data(i, 0);
    // Define a binary variable for Path5
    int p5 = (p == 4) ? 1 : 0;
    // Compute the current EMA value
    double ema_curr = alpha * p5 + (1 - alpha) * ema_prev;
    // Store the current EMA value in the vector
    ema(i) = ema_curr;
    // Update the previous EMA value
    ema_prev = ema_curr;
  }
  // Return the EMA vector
  return ema;
}


// Define a function to check if the EMA reaches 0.5 in either state
bool check_path5(arma::mat data) {
  // Filter data for state 0
  arma::mat dataS0 = data.rows(find(data.col(1) == 0));
  // Filter data for state 1
  arma::mat dataS1 = data.rows(find(data.col(1) == 1));
  
  // Compute the EMA of occurrence of Path5 for state 0
  arma::vec ema0 = ema_path5(dataS0);
  // Compute the EMA of occurrence of Path5 for state 1
  arma::vec ema1 = ema_path5(dataS1);
  
      // Check if any element in ema0 is greater than 0.8
    bool anyGreaterThanPointEight_ema0 = std::any_of(ema0.begin(), ema0.end(), [](double element) {
        return element > 0.8;
    });

    // Check if any element in ema1 is greater than 0.8
    bool anyGreaterThanPointEight_ema1 = std::any_of(ema1.begin(), ema1.end(), [](double element) {
        return element > 0.8;
    });

    // If path5 prob goes above 0.8 for any state, return false (bad simulation)
    if(anyGreaterThanPointEight_ema1 || anyGreaterThanPointEight_ema0)
    {
        std::cout << "check_path5 failed, Path5 prob > 0.95; anyGreaterThanPointEight_ema1 = " << anyGreaterThanPointEight_ema1 << ", anyGreaterThanPointEight_ema0 = " << anyGreaterThanPointEight_ema0 <<std::endl;
        return false;
    }


  bool S0Path5 = false;
  // Loop over the EMA values
  for (int i = 0; i < ema0.n_elem; i++) {
    // If the EMA reaches 0.5 in either state
    if (ema0(i) >= 0.5) {
      // Return true
      S0Path5 = true;
    }
  }

  bool S1Path5 = false;
  for (int i = 0; i < ema1.n_elem; i++) {
    // If the EMA reaches 0.5 in either state
    if (ema1(i) >= 0.5) {
      // Return true
      S1Path5  = true;
    }
  }

  if(S0Path5 || S1Path5)
  {
    return true;
  }

    

  // Return false
  return false;
}


bool checkConsecutiveThreshold(arma::mat data, double threshold, int consecutiveCount, int changepoint) {

    arma::mat dataS0 = data.rows(find(data.col(1) == 0));
    arma::mat dataS1 = data.rows(find(data.col(1) == 1));
  
    arma::vec ema0 = ema_rewards(dataS0, 0);
    arma::vec ema1 = ema_rewards(dataS1, 1);

    bool S0condition = false;
    bool S1condition = false;

    int consecutiveRows = 0;

    // ema0 = ema0.subvec(changepoint, ema0.n_elem-1);
    // ema1 = ema1.subvec(changepoint, ema1.n_elem-1);
    
    for (unsigned int i = changepoint; i < ema0.n_elem; ++i) {
        if (ema0(i) < threshold) {
            consecutiveRows++;
            if (consecutiveRows >= consecutiveCount) {
                S0condition = true;
                break; // Exit loop if condition is met
            }
        } else {
            consecutiveRows = 0; // Reset consecutive count if the condition is not met
        }
    }


    consecutiveRows = 0;

    for (unsigned int i = changepoint; i < ema1.n_elem; ++i) {
        if (ema1(i) < threshold) {
            consecutiveRows++;
            if (consecutiveRows >= consecutiveCount) {
                S1condition = true;
                break; // Exit loop if condition is met
            }
        } else {
            consecutiveRows = 0; // Reset consecutive count if the condition is not met
        }
    }

    if(S0condition || S1condition)
    {
        return true;
    }

    // No consecutive rows meeting the condition
    return false;
}



RatData generateSimulation(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::map<std::string, std::vector<double>> clusterParams, RInside &R, int selectStrat)
{
    
    std::cout << "Inside generateSimulation, selectStrat=" << selectStrat << std::endl;
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

    std::string rat = ratdata.getRat();
    double crpAlpha = clusterParams.find(rat)->second[0];
    double phi = clusterParams.find(rat)->second[1];
    double eta = 0;

    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, crpAlpha, phi, eta, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, crpAlpha, phi, eta, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, crpAlpha, phi, eta, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, crpAlpha, phi, eta, true);

    // COMMENTING OUT ARL
    // auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"arl", alpha_arl_subOptimal, beta_arl_subOptimal, lambda_arl_subOptimal, crpAlpha, phi, eta, false);
    // auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"arl",alpha_arl_optimal, beta_arl_optimal, lambda_arl_optimal, crpAlpha, phi, eta, true);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;


    //std::srand(static_cast<unsigned>(std::time(nullptr)));

    // int start = -1;
    // int end = -1;

    // if(rat == "rat_106")
    // {
    //     start = 8;
    //     end = 12;
    // }else
    // {
    //     start = 8;
    //     end = 12;
    // }    
    // std::random_device rd;
    // std::mt19937 gen(rd());
    // std::uniform_int_distribution<int> distribution(5,9);
    // int changepoint_ses = distribution(gen);
    int changepoint_ses = -1;

    // std::vector<std::shared_ptr<Strategy>> subOptimalStrategies = {aca2_Suboptimal_Hybrid3,drl_Suboptimal_Hybrid3,arl_Suboptimal_Hybrid3};
    // std::vector<std::shared_ptr<Strategy>> optimalStrategies = {aca2_Optimal_Hybrid3,drl_Optimal_Hybrid3,arl_Optimal_Hybrid3};

    // std::uniform_int_distribution<std::size_t> distribution_strat(0, subOptimalStrategies.size() - 1);
    // // Generate a random index for vector1
    // std::size_t random_index1 = distribution_strat(gen);

    // // Generate a random index for vector2
    // std::size_t random_index2 = distribution_strat(gen);

    // Get the randomly selected elements
    // std::shared_ptr<Strategy> subOptimalStrategy = subOptimalStrategies[random_index1];
    // std::shared_ptr<Strategy> optimalStrategy = optimalStrategies[random_index2];
    // // std::shared_ptr<Strategy> subOptimalStrategy = aca2_Suboptimal_Hybrid3;
    // // std::shared_ptr<Strategy> optimalStrategy = aca2_Optimal_Hybrid3;

    std::vector<std::pair<std::shared_ptr<Strategy>, std::shared_ptr<Strategy>>> strategyPairVector;

    strategyPairVector.push_back(std::make_pair(drl_Suboptimal_Hybrid3, drl_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(aca2_Suboptimal_Hybrid3, drl_Optimal_Hybrid3));
    
    strategyPairVector.push_back(std::make_pair(drl_Suboptimal_Hybrid3, aca2_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(aca2_Suboptimal_Hybrid3, aca2_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(aca2_Optimal_Hybrid3, aca2_Optimal_Hybrid3));
    
    //strategyPairVector.push_back(std::make_pair(aca2_Suboptimal_Hybrid3, arl_Optimal_Hybrid3));
    
    //strategyPairVector.push_back(std::make_pair(drl_Suboptimal_Hybrid3, arl_Optimal_Hybrid3));
    //strategyPairVector.push_back(std::make_pair(arl_Suboptimal_Hybrid3, arl_Optimal_Hybrid3));
    //strategyPairVector.push_back(std::make_pair(arl_Suboptimal_Hybrid3, aca2_Optimal_Hybrid3));
    //strategyPairVector.push_back(std::make_pair(arl_Suboptimal_Hybrid3, drl_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(drl_Optimal_Hybrid3, drl_Optimal_Hybrid3));
    //strategyPairVector.push_back(std::make_pair(arl_Optimal_Hybrid3, arl_Optimal_Hybrid3));



    // std::random_device rd;
    // std::mt19937 gen(rd());
    //std::uniform_int_distribution<std::size_t> dist(0, strategyPairVector.size() - 1);
    
    // Selecting a random pair
    //std::size_t randomIndex = dist(gen);
    auto randomPair = strategyPairVector[selectStrat];


    arma::mat generated_PathData;
    arma::mat generated_TurnsData;
   
    
    // To store vector of true generatorStrateies
    std::vector<std::string> trueGenStrategies;
  
   bool endLoop = false;
   int loopCounter = 0;

    std::shared_ptr<Strategy> strategy;

    if(randomPair.first->getName() == randomPair.second->getName())
    {
        while(!endLoop)
        {
            //std::srand(static_cast<unsigned>(std::time(nullptr)));

            std::cout << "Generating sim data with " << randomPair.first->getName() << " and "<< randomPair.second->getName()  << " changepoint at ses " <<  changepoint_ses << std::endl;


            std::shared_ptr<Strategy> trueStrategy1 = std::make_shared<Strategy>(*randomPair.first);

            //std::vector<double> creditsS0_Subopt = {0,0,0,0,0,0,0,0,0,0,0,0};
            std::vector<double> creditsS0_Opt = {0,0,0,0,0,0,0,0,0};
            std::vector<double> creditsS1_Opt = {0,0,0,0,0,0,0,0,0};

            std::vector<double> s0rewards = {0,0,0,0,0,0,0,5,0};
            std::vector<double> s1rewards = {0,0,0,0,0,0,0,0,5};

            trueStrategy1->setRewardsS0(s0rewards);
            trueStrategy1->setRewardsS1(s1rewards);

            for(int ses=0; ses < sessions; ses++)
            {
                strategy = randomPair.first;
                trueGenStrategies.push_back(strategy->getName());

                strategy->setStateS0Credits(creditsS0_Opt);
                strategy->setStateS1Credits(creditsS1_Opt);


                std::pair<arma::mat, arma::mat> simData = simulateTrajectory(ratdata, ses, *strategy);
                arma::mat generated_PathData_sess = simData.first;
                arma::mat generated_TurnsData_sess = simData.second;

                trueStrategy1->getTrajectoryLikelihood(ratdata, ses);
                
                //creditsS0_Subopt =  trueStrategy1->getS0Credits(); 
                
                creditsS0_Opt =  trueStrategy1->getS0Credits(); 
                creditsS1_Opt =  trueStrategy1->getS1Credits(); 

                generated_PathData = arma::join_cols(generated_PathData, generated_PathData_sess);
                generated_TurnsData = arma::join_cols(generated_TurnsData, generated_TurnsData_sess);
            
            }
            bool isGenDataGood = true;

            if(isGenDataGood)
            {
                //2nd test: check if simulation is learning
                if(check_ema(generated_PathData))
                {
                    isGenDataGood = true;
                    //std::cout << "check_ema is successful. EXit loop" <<std::endl;

                }else{
                    isGenDataGood = false;
                    //std::cout << "check_ema failed. Re-generate try: " << loopCounter <<std::endl;
                }
            }
            if(isGenDataGood)
            {
                endLoop = true;
            }else
            {
                generated_PathData.reset();
                generated_TurnsData.reset();
                strategy->resetPathProbMat();
                strategy->resetCredits();
            }

            loopCounter++;
            if(loopCounter > 50)
            {
                std::cout << "Loop counter reached 50 simulations:" << randomPair.first->getName() << " and " << randomPair.second->getName() << ". Exiting" << std::endl;
                break;
            }
        }
    
    }else //Generate switching simulations
    {
        // std::vector<double> initCreditsS0Opt = {0,0,0,1.5,0,0,1.5,0};
        // std::vector<double> initCreditsS1Opt = {0,0,0,1.5,0,0,0,1.5};
        // randomPair.second->setStateS0Credits(initCreditsS0Opt);
        // randomPair.second->setStateS1Credits(initCreditsS1Opt);

        while(!endLoop)
        {
            //std::srand(static_cast<unsigned>(std::time(nullptr)));
            //std::vector<double> initCreditsS0 = {0,0,0,1.5,0,0,1.5,0,0,0,0,0};
            //randomPair.first->setStateS0Credits(initCreditsS0);

            std::random_device rd;
            std::mt19937 gen(rd());
            if(rat=="rat_103")
            {
                std::uniform_int_distribution<int> distribution(6,8);
                changepoint_ses = distribution(gen);
            }else{
                std::uniform_int_distribution<int> distribution(2,4);
                changepoint_ses = distribution(gen);
            }
   
            std::shared_ptr<Strategy> trueStrategy1 = std::make_shared<Strategy>(*randomPair.first);
            std::shared_ptr<Strategy> trueStrategy2 = std::make_shared<Strategy>(*randomPair.second);

            //std::vector<double> initCreditsS0 = {0,0,0,0,0,0,0,0,0,0,0,0};
            std::vector<double> creditsS0_Opt = {0,0,0,0,0,0,0,0,0};
            std::vector<double> creditsS1_Opt = {0,0,0,0,0,0,0,0,0};

            std::vector<double> s0rewards = {0,0,0,0,0,0,0,5,0};
            std::vector<double> s1rewards = {0,0,0,0,0,0,0,0,5};

            trueStrategy2->setRewardsS0(s0rewards);
            trueStrategy2->setRewardsS1(s1rewards);

            std::vector<double> s0rewardsSubOpt;
            if(rat=="rat_103")
            {
                s0rewardsSubOpt = {0,0,0,0,0,0,0,5,0,0,0,0};
            }else{
                s0rewardsSubOpt = {0,0,0,0,0,0,5,0,0,0,0,0};
            }
            
            trueStrategy1->setRewardsS0(s0rewardsSubOpt);



            for(int ses=0; ses < sessions; ses++)
            {
                std::pair<arma::mat, arma::mat> simData;
                arma::mat generated_PathData_sess;
                arma::mat generated_TurnsData_sess;
                // std::shared_ptr<Strategy> randomPair_first_bkp = std::make_shared<Strategy>(*randomPair.first);
                // std::shared_ptr<Strategy> randomPair_second_bkp = std::make_shared<Strategy>(*randomPair.second);

                bool path5Cond = false;
                int counter = 0;

                //Start suboptimal portion of switching simulations
                if(ses < changepoint_ses)
                {
                    
                    randomPair.first->setStateS0Credits(creditsS0_Opt);
                    // std::cout <<"creditsS0_Subopt:";
                    // std::vector<double> initCr = randomPair.first->getS0Credits(); 
                    // for (const double& value : initCr) {
                    //     std::cout << value << " ";
                    // }

                    // std::cout << std::endl;

                    strategy = randomPair.first;
                    simData = simulateTrajectory(ratdata, ses, *randomPair.first);
                    generated_PathData_sess = simData.first;
                    generated_TurnsData_sess = simData.second;

                    arma::uvec s0indices = arma::find(generated_PathData_sess.col(1) == 0); 
                    arma::mat genDataS0 = generated_PathData_sess.rows(s0indices);

                    arma::uvec s1indices = arma::find(generated_PathData_sess.col(1) == 1); 
                    arma::mat genDataS1 = generated_PathData_sess.rows(s1indices);

                }else{  //Start Optimal portion of switching simulations

                    randomPair.second->setStateS0Credits(creditsS0_Opt);
                    randomPair.second->setStateS1Credits(creditsS1_Opt);

                    
                    // std::cout <<"ses=" <<ses <<std::endl;
                    // std::cout <<"creditsS0_Opt:";
                    // for (const double& value : creditsS0_Opt) {
                    //     std::cout << value << " ";
                    // }
                    // std::cout << std::endl;

                    // std::cout <<"creditsS1_Opt:";
                    // for (const double& value : creditsS1_Opt) {
                    //     std::cout << value << " ";
                    // }
                    // std::cout << std::endl;

                    strategy = randomPair.second;
                    simData = simulateTrajectory(ratdata, ses, *randomPair.second);
                    generated_PathData_sess = simData.first;
                    generated_TurnsData_sess = simData.second;


                    arma::uvec s0indices = arma::find(generated_PathData_sess.col(1) == 0); 
                    arma::mat genDataS0 = generated_PathData_sess.rows(s0indices);

                    arma::uvec s1indices = arma::find(generated_PathData_sess.col(1) == 1); 
                    arma::mat genDataS1 = generated_PathData_sess.rows(s1indices);


                } //End session of switching simulations

                trueStrategy1->getTrajectoryLikelihood(ratdata, ses);
                trueStrategy2->getTrajectoryLikelihood(ratdata, ses);
                
                //creditsS0_Subopt =  trueStrategy1->getS0Credits(); 
                
                creditsS0_Opt =  trueStrategy2->getS0Credits(); 
                creditsS1_Opt =  trueStrategy2->getS1Credits(); 

                trueGenStrategies.push_back(strategy->getName());

                generated_PathData = arma::join_cols(generated_PathData, generated_PathData_sess);
                generated_TurnsData = arma::join_cols(generated_TurnsData, generated_TurnsData_sess);
            
            }

            // arma::mat allpaths = ratdata.getPaths();
            // arma::vec sessionVec = allpaths.col(4);
            // arma::vec uniqSessIdx = arma::unique(sessionVec);
            // std::cout << "uniqSessIdx.size=" << uniqSessIdx.size() << std::endl;

            bool isGenDataGood = true;
            if(!randomPair.first->getOptimal())
            {
                arma::uvec indices = arma::find(generated_PathData.col(4) < changepoint_ses);
                
                arma::mat subMat = generated_PathData.rows(indices); //Update
                //std::cout << "indices.size=" << indices.size() << ", subMat rows=" << subMat.n_rows << std::endl;
                //First check of genData: If Path5 prob > 0.5 for any state, it is good    
                if(check_path5(subMat))
                {
                    isGenDataGood = true;
                    std::cout << rat << ", check_path5 is successful after " << changepoint_ses << " sessions" <<std::endl;
                }else{
                    isGenDataGood = false;
                    std::cout << rat << ", check_path5 failed after " << changepoint_ses << " sessions" <<std::endl;
                }
            }


            if(isGenDataGood)
            {
                //2nd test: check if simulation is learning
                if(check_ema(generated_PathData))
                {
                    // if(!checkConsecutiveThreshold(generated_PathData, 0.6, 30, changepoint_ses))
                    // {
                    //     isGenDataGood = true;
                    //     std::cout << "checkConsecutiveThreshold false. Good sim" <<std::endl;
                    // }else{
                    //     isGenDataGood = false;
                    //     std::cout << "checkConsecutiveThreshold true. Check sim" <<std::endl;
                    // }
                    

                }else{
                    isGenDataGood = false;
                    std::cout << "check_ema failed. Re-generate try: " << loopCounter <<std::endl;
                }
            }

            if(isGenDataGood)
            {
                endLoop = true;
                // arma::vec simSessionVec = generated_PathData.col(4);
                // arma::vec simUniqSessIdx = arma::unique(simSessionVec);
                // std::cout << "simUniqSessIdx.size=" << simUniqSessIdx.size() << std::endl;


            }else
            {
                generated_PathData.reset();
                generated_TurnsData.reset();

                randomPair.first->resetPathProbMat();
                randomPair.first->resetCredits();

                randomPair.second->resetPathProbMat();
                randomPair.second->resetCredits();
            }

            loopCounter++;
            if(loopCounter > 100)
            {
                std::cout << rat <<  ", loop counter reached 100 for simulation:" << randomPair.first->getName() << " and " << randomPair.second->getName() << ". Exiting" << std::endl;
                break;
            }
        }
    }

    std::cout << "Generated sim:" << randomPair.first->getName() << " and " << randomPair.second->getName() << std::endl;
  
    // R["genData"] = Rcpp::wrap(generated_PathData);

    // // Save the matrix as RData using RInside
    // std::string filename = "generatedData_" + std::to_string(selectStrat) + "_" + rat +".RData";
    
    // std::string rCode = "saveRDS(genData, file='" + filename + "')";
    // R.parseEvalQ(rCode.c_str());
    
    RatData simRatdata(generated_PathData,generated_TurnsData,rat, true, trueGenStrategies);

    arma::mat simAllpaths = simRatdata.getPaths();
    arma::vec simSessionVec = simAllpaths.col(4);
    arma::vec simUniqSessIdx = arma::unique(simSessionVec);
    // std::cout << "simUniqSessIdx.size=" << simUniqSessIdx.size() << std::endl;

    //testSimulation(simRatdata,*randomPair.first, R);
    return simRatdata;

}


//Ignore- cannot run testSimulation as getTrajectoryLikelihood requires setting rewardsS0 & rewardsS1, which is not "possible" for suboptimal case
void testSimulation(RatData& simRatData, Strategy& trueStrategy, RInside &R)
{
    arma::mat allpaths = simRatData.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    arma::mat probMatTrueStrat = trueStrategy.getPathProbMat();

    std::cout << "Last 10 rows of matrix probMatTrueStrat:\n" << probMatTrueStrat.tail_rows(10) << "\n";

    Strategy copyTrueStrategy(trueStrategy);
    copyTrueStrategy.resetPathProbMat();
    copyTrueStrategy.resetCredits();

    std::vector<double> s0rewards;
    std::vector<double> s1rewards;

    if(trueStrategy.getOptimal())
    {
        s0rewards = {0,0,0,0,0,0,0,5,0};
        s1rewards = {0,0,0,0,0,0,0,0,5};
        copyTrueStrategy.setRewardsS0(s0rewards);
        copyTrueStrategy.setRewardsS1(s1rewards);


    }else{
        s0rewards = {0,0,0,0,0,0,5,5,0,0,0,0};
        copyTrueStrategy.setRewardsS0(s0rewards);
    }


    copyTrueStrategy.setAverageReward(0);
    

    for(int ses=0; ses < sessions; ses++)
    {
        double log_likelihood = copyTrueStrategy.getTrajectoryLikelihood(simRatData, ses); 
    }

    arma::mat probMatCopyTrueStrat = copyTrueStrategy.getPathProbMat();

    // R.assign("probMatTrueSim", probMatTrueStrat);
    // R.assign("probMatLikFunc", probMatCopyTrueStrat);

    // // Save the R variables to an RData file
    // R.parseEvalQ("save(list=c('probMatTrueSim', 'probMatLikFunc'), file='simTestProbMats.RData')");

    Rcpp::NumericMatrix probMatTrueStrat_r = Rcpp::wrap(probMatTrueStrat);
    Rcpp::NumericMatrix probMatCopyTrueStrat_r = Rcpp::wrap(probMatCopyTrueStrat);

    // Assign them to R variables
    R["probMatTrueStrat"] = probMatTrueStrat_r;
    R["probMatCopyTrueStrat"] = probMatCopyTrueStrat_r;

    // Save them as Rdata
    R.parseEvalQ("save(probMatTrueStrat, probMatCopyTrueStrat, file = 'simTestProbMats.RData')");

    arma::mat diffMatrix = arma::abs(probMatTrueStrat-probMatCopyTrueStrat);

    // Find the maximum value in the absolute difference matrix
    double maxDifference = arma::max(diffMatrix).eval()(0, 0);

    std::cout << "maxDifference:" << maxDifference << "\n";

    return;
}

std::map<std::pair<std::string, bool>, std::vector<double>> findParamsWithSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3)
{
    // COMMENTING OUT ARL
    
    //std::vector<std::string> learningRules = {"aca2","arl", "drl" };
    std::vector<std::string> learningRules = {"aca2","drl" };
    std::vector<bool> mazeModels = {false, true};

    std::map<std::pair<std::string, bool>, std::vector<double>> paramStrategies;

    std::cout << "Finding params for simdata"<< std::endl;

    //RatData sim;

    for (const auto &lr : learningRules) 
    {
        for (const auto &optimal : mazeModels) 
        {
            std::string learningRule =  lr;   
            MazeGraph* maze;
            if(optimal)
            {
                maze = &optimalHybrid3;
                // sim = ratdata;
            }else
            {
                maze = &suboptimalHybrid3;
                // arma::mat allPaths = ratdata.getPaths();
                // arma::mat hybrid3Turns = ratdata.getHybrid3();

                // arma::uvec indices1 = arma::find(hybrid3Turns.col(4) < 12);
                // arma::mat hybrid3SubMat = hybrid3Turns.rows(indices1); //Update

                // arma::uvec indices2 = arma::find(allPaths.col(4) < 12);
                // arma::mat allpathsSubMat = allPaths.rows(indices2);

                // sim.setHybrid3(hybrid3SubMat);
                // sim.setPaths(allpathsSubMat);
 
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

std::vector<double> findClusterParamsWithSimData(RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params)
{
        // Create a function to optimize
    PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3,params);
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

void updateConfusionMatrix(std::vector<RecordResults> allSesResults, std::string rat)
{

    std::string filename = "confusionMatrix_" + rat+".txt";
    std::ifstream confMat_file(filename);

    std::vector<std::string> rows = {"aca2_Suboptimal_Hybrid3", "drl_Suboptimal_Hybrid3", "arl_Suboptimal_Hybrid3", "aca2_Optimal_Hybrid3", "drl_Optimal_Hybrid3", "arl_Optimal_Hybrid3"};
    std::vector<std::string> columns = {"aca2_Suboptimal_Hybrid3", "drl_Suboptimal_Hybrid3", "arl_Suboptimal_Hybrid3", "aca2_Optimal_Hybrid3", "drl_Optimal_Hybrid3", "arl_Optimal_Hybrid3", "None"};
    //std::vector<std::vector<int>> matrix(6, std::vector<int>(7, 0));
    std::vector<std::vector<int>> matrix;

    std::vector<std::string> rownames = {"acaSubopt", "drlSubopt", "arlSubopt", "acaOpt", "drlOpt", "arlOpt"};
    std::vector<std::string> colnames = {"acaSubopt", "drlSubopt", "arlSubopt", "acaOpt", "drlOpt", "arlOpt", "None"};

    std::map<std::string, int> rowLabelToIndex;
    std::map<std::string, int> colLabelToIndex;

    // Fill the maps with indices
    for (int i = 0; i < rows.size(); ++i) {
        rowLabelToIndex[rows[i]] = i;
    }

    for (int j = 0; j < columns.size(); ++j) {
        colLabelToIndex[columns[j]] = j;
    }

    if (confMat_file.is_open()) {
        // Skip the row header
        std::string line;
        std::getline(confMat_file, line);

        // Read the matrix
        std::string rowHeader;
        
        while (std::getline(confMat_file, line)) {
            std::istringstream rowStream(line);
            std::vector<int> row;
            int value;

            // Skip the row header
            rowStream >> rowHeader;
            //std::cout << "rowHeader=" << rowHeader << std::endl;

            // Read the values in the row
            while (rowStream >> value) {
                //std::cout << "value=" << value << std::endl;
                row.push_back(value);
            }

            matrix.push_back(row);
        }

        // Close the file
        confMat_file.close();
    } else {
        std::cout << "File not found or could not be opened. Creating a new matrix.\n";
        matrix = std::vector<std::vector<int>>(6, std::vector<int>(7, 0));
    }

    // Print the matrix
    std::cout << "Matrix print before update:" << std::endl;
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << '\n';
    }


    
    for(size_t i=0; i<allSesResults.size();i++)
    {
        RecordResults recordResultsSes =  allSesResults[i];
        std::string selectedStrategy = recordResultsSes.getSelectedStrategy();   //column label
        std::string trueStrategy = recordResultsSes.getTrueGeneratingStrategy(); // rowLabel

        std::cout << "trueStrategy=" << trueStrategy << ", idx=" << rowLabelToIndex[trueStrategy] << "; selectedStrategy=" << selectedStrategy << ", idx=" << colLabelToIndex[selectedStrategy] << std::endl;

        matrix[rowLabelToIndex[trueStrategy]][colLabelToIndex[selectedStrategy]] = matrix[rowLabelToIndex[trueStrategy]][colLabelToIndex[selectedStrategy]] +1  ;
    }

    // Print the matrix
    std::cout << "Matrix print after update:" << std::endl;
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << '\n';
    }

    std::ofstream outfile(filename);

    if (outfile.is_open()) {
        // Write the column names to the first line, separated by spaces
        outfile << std::setw(10) << " "; // Leave some space for the row names
        for (const auto& colname : colnames) {
            outfile << std::setw(10) << colname;
        }
        outfile << "\n"; // End the line

        // Write the matrix elements and the row names, separated by spaces
        for (size_t i = 0; i < matrix.size(); i++) {
            outfile << std::setw(10) << rownames[i]; // Write the row name
            for (size_t j = 0; j < matrix[i].size(); j++) {
                outfile << std::setw(10) << matrix[i][j]; // Write the matrix element
            }
            outfile << "\n"; // End the line
        }

        outfile.close();
        std::cout << "Matrix written to file.\n";
    } else {
        std::cout << "File could not be opened for writing.\n";
    }

}

void writeResults(std::vector<RecordResults> allSesResults, std::string rat, int genStrategyId , int iteration)
{
    
    std::vector<std::string> rows = {"aca2_Suboptimal_Hybrid3", "drl_Suboptimal_Hybrid3", "arl_Suboptimal_Hybrid3", "aca2_Optimal_Hybrid3", "drl_Optimal_Hybrid3", "arl_Optimal_Hybrid3"};
    std::vector<std::string> columns = {"aca2_Suboptimal_Hybrid3", "drl_Suboptimal_Hybrid3", "arl_Suboptimal_Hybrid3", "aca2_Optimal_Hybrid3", "drl_Optimal_Hybrid3", "arl_Optimal_Hybrid3", "None"};
    //std::vector<std::vector<int>> matrix(6, std::vector<int>(7, 0));
    std::vector<std::vector<int>> matrix = std::vector<std::vector<int>>(6, std::vector<int>(7, 0));

    std::vector<std::string> rownames = {"acaSubopt", "drlSubopt", "arlSubopt", "acaOpt", "drlOpt", "arlOpt"};
    std::vector<std::string> colnames = {"acaSubopt", "drlSubopt", "arlSubopt", "acaOpt", "drlOpt", "arlOpt", "None"};

    std::map<std::string, int> rowLabelToIndex;
    std::map<std::string, int> colLabelToIndex;

    // Fill the maps with indices
    for (int i = 0; i < rows.size(); ++i) {
        rowLabelToIndex[rows[i]] = i;
    }

    for (int j = 0; j < columns.size(); ++j) {
        colLabelToIndex[columns[j]] = j;
    }


    
    for(size_t i=0; i<allSesResults.size();i++)
    {
        RecordResults recordResultsSes =  allSesResults[i];
        std::string selectedStrategy = recordResultsSes.getSelectedStrategy();   //column label
        std::string trueStrategy = recordResultsSes.getTrueGeneratingStrategy(); // rowLabel

        // std::cout << "trueStrategy=" << trueStrategy << ", idx=" << rowLabelToIndex[trueStrategy] << "; selectedStrategy=" << selectedStrategy << ", idx=" << colLabelToIndex[selectedStrategy] << std::endl;

        matrix[rowLabelToIndex[trueStrategy]][colLabelToIndex[selectedStrategy]] = matrix[rowLabelToIndex[trueStrategy]][colLabelToIndex[selectedStrategy]] +1  ;
    }

    // Print the matrix
    // std::cout << "Matrix print after update:" << std::endl;
    // for (size_t i = 0; i < matrix.size(); ++i) {
    //     for (size_t j = 0; j < matrix[i].size(); ++j) {
    //         std::cout << matrix[i][j] << ' ';
    //     }
    //     std::cout << '\n';
    // }


    std::string mainDirPath = "/home/amoongat/Projects/SwitchingRatsCredits/Results/" + rat;
    std::filesystem::path main_dir(mainDirPath);

     // Create the main directory if it does not exist
    if (!std::filesystem::exists(main_dir)) {
        std::filesystem::create_directory(main_dir);
    }

    std::string subDir = "Strat_" + std::to_string(genStrategyId);
    // Path to the subdirectory
    std::filesystem::path sub_dir(main_dir / subDir);

    std::string stringpath = sub_dir.generic_string();


    std::cout << "genStrategyId=" << genStrategyId << "sub_dir=" << stringpath << std::endl;

    if (!std::filesystem::exists(sub_dir)) {
        std::filesystem::create_directory(sub_dir);
    }

    //std::string fullPath = mainDirPath + subDirPath;
    std::string filename = "confusionMatrix_" + rat+ "_"+ std::to_string(iteration) +".txt";
    // std::string filePath = fullPath + "/my_file.txt";

    // if (!std::filesystem::exists(mainDirPath)) {
    //     // Create the subdirectory
    //     if (std::filesystem::create_directory(mainDirPath)) {
    //         std::cout << "Subdirectory created: " << fullPath << std::endl;
    //     } else {
    //         std::cerr << "Failed to create the subdirectory." << std::endl;
    //     }
    // } else {
    //     std::cout << "Subdirectory already exists: " << fullPath << std::endl;
    // }




    std::ofstream outfile(sub_dir / filename);

    if (outfile.is_open()) {
        // Write the column names to the first line, separated by spaces
        outfile << std::setw(10) << " "; // Leave some space for the row names
        for (const auto& colname : colnames) {
            outfile << std::setw(10) << colname;
        }
        outfile << "\n"; // End the line

        // Write the matrix elements and the row names, separated by spaces
        for (size_t i = 0; i < matrix.size(); i++) {
            outfile << std::setw(10) << rownames[i]; // Write the row name
            for (size_t j = 0; j < matrix[i].size(); j++) {
                outfile << std::setw(10) << matrix[i][j]; // Write the matrix element
            }
            outfile << "\n"; // End the line
        }

        outfile.close();
        std::cout << "Matrix written to file.\n";
    } else {
        std::cout << "File could not be opened for writing.\n";
    }
}


void runEMOnSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::vector<double> v, bool debug, int genStrategyId, int iteration)
{
    //// rat_103
    //std::vector<double> v = {0.11776, 0.163443, 0.0486187, 1e-7,0.475538, 0.272467, 1e-7 , 0.0639478, 1.9239e-06, 0.993274, 4.3431};
    
    ////rat_114
    //std::vector<double> v = {0.0334664, 0.351993, 0.00478871, 1.99929e-07, 0.687998, 0.380462, 9.68234e-07, 0.136651, 8.71086e-06, 0.292224, 3.95355};



    std::string rat = ratdata.getRat();

    // //ACA params
    double alpha_aca_subOptimal = params.find(std::make_pair("aca2", false))->second[0];
    double gamma_aca_subOptimal = params.find(std::make_pair("aca2", false))->second[1];

    double alpha_aca_optimal = params.find(std::make_pair("aca2", true))->second[0];
    double gamma_aca_optimal = params.find(std::make_pair("aca2", true))->second[1];

    //COMMENTING OUT ARL
    // //ARL params
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

    // double crpAlpha = clusterParams[0];
    // double phi = clusterParams[1];
    // double eta = 0;


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


    std::vector<std::shared_ptr<Strategy>> strategies;
    strategies.push_back(aca2_Suboptimal_Hybrid3);
    strategies.push_back(aca2_Optimal_Hybrid3);

    strategies.push_back(drl_Suboptimal_Hybrid3);
    strategies.push_back(drl_Optimal_Hybrid3);

    // COMMENTING OUT ARL
    // strategies.push_back(arl_Suboptimal_Hybrid3);
    // strategies.push_back(arl_Optimal_Hybrid3);

    // aca2_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_aca);
    // drl_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_drl);
    // arl_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_arl);
    
    // aca2_Optimal_Hybrid3->setRewardsS0(rewardsS0_aca); 
    // aca2_Optimal_Hybrid3->setRewardsS1(rewardsS1_aca);

    // drl_Optimal_Hybrid3->setRewardsS0(rewardsS0_drl); 
    // drl_Optimal_Hybrid3->setRewardsS1(rewardsS1_drl);

    // arl_Optimal_Hybrid3->setRewardsS0(rewardsS0_arl); 
    // arl_Optimal_Hybrid3->setRewardsS1(rewardsS1_arl);    


    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    std::vector<std::string> cluster;
    std::string last_choice;
    
    arma::mat probMat;

    std::vector<RecordResults> allSesResults;

    std::vector<std::string> trueStrategies = ratdata.getGeneratorStrategies();

    for(int ses=0; ses < sessions; ses++)
    {
        
        if(ses==0)
        {
            initRewardVals(ratdata, ses, strategies, debug);
        }
 
        RecordResults sessionResults;
        sessionResults.setTrueGeneratingStrategy(trueStrategies[ses]);

        arma::mat probMat_sess = estep_cluster_update(ratdata, ses, strategies, cluster, last_choice, true, sessionResults);
        mstep(ratdata, ses, strategies, cluster, debug, sessionResults);

        probMat = arma::join_cols(probMat, probMat_sess);

        allSesResults.push_back(sessionResults);

        //RecordResults sessionResults(selectedStrategy, probabilityMatrix, trueGeneratingStrategy, posteriors, rewardVectorS0, rewardVectorS1);
    }

    arma::mat& aca2_suboptimal_probs =  aca2_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& aca2_optimal_probs =  aca2_Optimal_Hybrid3->getPathProbMat();
    arma::mat& drl_suboptimal_probs =  drl_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& drl_optimal_probs =  drl_Optimal_Hybrid3->getPathProbMat();

    //// COMMENTING OUT ARL
    // arma::mat& arl_suboptimal_probs =  arl_Suboptimal_Hybrid3->getPathProbMat();
    // arma::mat& arl_optimal_probs =  arl_Optimal_Hybrid3->getPathProbMat();

    probMat.save("ProbMat_Sim_" + rat+ ".csv", arma::csv_ascii);
    //updateConfusionMatrix(allSesResults, rat);
    writeResults(allSesResults, rat, genStrategyId , iteration);


    aca2_suboptimal_probs.save("aca2_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    aca2_optimal_probs.save("aca2_optimal_probs_"+ rat+".csv", arma::csv_ascii);
    drl_suboptimal_probs.save("drl_suboptimal_probs_"+ rat+".csv", arma::csv_ascii);
    drl_optimal_probs.save("drl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // COMMENTING OUT ARL
    // arl_suboptimal_probs.save("arl_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // arl_optimal_probs.save("arl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);


}


void testRecovery(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, RInside &R)
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

    // Initialize MPI
    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Calculate the range of iterations for each process
    int partition_size = 30 / size;
    int start = rank * partition_size;
    int end = (rank == size - 1) ? 30 : start + partition_size;

    for(int i = start; i < end; i++)
    {
        int genStrategyId = i%6;
        int iteration = i/6;

        std::cout << "Rank=" <<rank << ", start=" <<start << ", end=" << end <<  ", genStrategyId=" <<genStrategyId << " and iteration=" <<iteration <<std::endl;
        
        RatData ratSimData = generateSimulation(ratdata, suboptimalHybrid3, optimalHybrid3, ratParams,clusterParams, R, genStrategyId);
        std::map<std::pair<std::string, bool>, std::vector<double>> simRatParams = findParamsWithSimData(ratSimData, suboptimalHybrid3, optimalHybrid3);
        std::vector<double> simClusterParams = findClusterParamsWithSimData(ratSimData, suboptimalHybrid3, optimalHybrid3,simRatParams);
        runEMOnSimData(ratSimData, suboptimalHybrid3, optimalHybrid3, simRatParams, simClusterParams, true, genStrategyId , iteration);

    }
  
    MPI_Finalize();
     

}





