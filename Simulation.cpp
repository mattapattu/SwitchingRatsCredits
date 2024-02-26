#include "InferStrategy.h"
#include "Pagmoprob.h"
#include "PagmoMle.h"
#include "Simulation.h"
#include "PagmoMultiObjCluster.h"
#include "ParticleFilter.h"
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/bfe.hpp>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/problems/unconstrain.hpp>
#include <pagmo/algorithms/pso_gen.hpp>
#include <pagmo/algorithms/gaco.hpp>
#include <pagmo/utils/multi_objective.hpp>
#include <pagmo/algorithms/moead.hpp>
#include <pagmo/algorithms/moead_gen.hpp>
#include <random>
#include <RInside.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>
#include <stdexcept>





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
bool check_ema(arma::mat data, double threshold = 0.8, int consecutive_count = 10) {
  arma::mat dataS0 = data.rows(find(data.col(1) == 0));
  arma::mat dataS1 = data.rows(find(data.col(1) == 1));
  
  arma::vec ema0 = ema_rewards(dataS0, 0);
  arma::vec ema1 = ema_rewards(dataS1, 1);
  
  int count0 = 0;
  int count1 = 0;
  bool s0 = false;
  bool s1 = false;

      auto middleIteratorS0 = ema0.begin() + ema0.size() / 2;


    // CHECK1: S0 does not decay suddenly after learning in the second half of exp    
    // Count the values less than 0.5 in the second half
    int countS0 = std::count_if(middleIteratorS0, ema0.end(), [](double value) {
        return value < 0.4;
    });

    // Check if the count is greater than 50
    if (countS0 > 100) {
        std::cout << "check_ema failed. S0 prob less than 0.5 for more than 200 trials" <<std::endl;
        return false;

    }

    //CHECK 2: S0 is not equal to 1 for most of trials
    int countGreaterThan099 = std::count_if(ema0.begin(), ema0.end(), [](double value) {
        return value > 0.99;
    });

    // Check if the count is greater than 50
    if (countGreaterThan099/ema0.size() > 0.9) {
        std::cout << "check_ema failed.  S0 is close to 1 for most of trials" <<std::endl;
        return false;

    }

    
    //CHECK 3: S1 does not decay suddenly after learning in second half of exp
    auto middleIteratorS1 = ema1.begin() + ema1.size() / 2;

    // Count the values less than 0.5 in the second half
    int countS1 = std::count_if(middleIteratorS1, ema1.end(), [](double value) {
        return value < 0.6;
    });

    // Check if the count is greater than 50
    if (countS1 > 100) {
        std::cout << "check_ema failed.S1 prob less than 0.5 for more than 100 trials; countS1 = " << countS1 <<std::endl;
        return false;

    }
  
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
        return element > 0.95;
    });

    // Check if any element in ema1 is greater than 0.8
    bool anyGreaterThanPointEight_ema1 = std::any_of(ema1.begin(), ema1.end(), [](double element) {
        return element > 0.95;
    });

    // If path5 prob goes above 0.8 for any state, return false (bad simulation)
    if(anyGreaterThanPointEight_ema1 || anyGreaterThanPointEight_ema0)
    {
        std::cout << "check_path5 failed, anyGreaterThanPointEight_ema0: " << anyGreaterThanPointEight_ema0 << ", anyGreaterThanPointEight_ema1:" << anyGreaterThanPointEight_ema1 <<std::endl;
        return false;
    }

    //CHECK 3: S1 does not decay suddenly after learning in second half of exp
    auto middleIteratorS0 = ema0.begin() + ema0.size() / 2;

    // Count the values less than 0.5 in the second half
    int countS0 = std::count_if(ema0.begin(), middleIteratorS0, [](double value) {
        return value > 0.3;
    });

    // Check if count of S0 path5 prob above 0.3 is greater than 20
    if (countS0 > 20) {
        std::cout << "check_path5 failed." <<std::endl;
        return false;

    }


//   bool S0Path5 = false;
//   // Loop over the EMA values
//   for (int i = 0; i < ema0.n_elem; i++) {
//     // If the EMA reaches 0.5 in either state
//     if (ema0(i) >= 0.5) {
//       // Return true
//       S0Path5 = true;
//     }
//   }

  int count1 = 0;
  bool S1Path5 = false;
  for (int i = 0; i < ema1.n_elem; i++) {
    if (ema1(i) >= 0.5) {
      count1++;
      if (count1 == 10) {
        S1Path5 = true;
        break;
      }
    } else {
      count1 = 0;
    }
  }

  if(S1Path5)
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



RatData generateSimulation(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::vector<double> v, RInside &R, int selectStrat, std::string run)
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

    // COMMENTING OUT ARL
    // auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"arl", alpha_arl_subOptimal, beta_arl_subOptimal, lambda_arl_subOptimal, crpAlpha, phi, eta, false);
    // auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"arl",alpha_arl_optimal, beta_arl_optimal, lambda_arl_optimal, crpAlpha, phi, eta, true);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    //std::srand(static_cast<unsigned>(std::time(nullptr)));

    
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

    strategyPairVector.push_back(std::make_pair(aca2_Suboptimal_Hybrid3, drl_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(aca2_Suboptimal_Hybrid3, aca2_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(drl_Suboptimal_Hybrid3, drl_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(drl_Suboptimal_Hybrid3, aca2_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(aca2_Optimal_Hybrid3, aca2_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(drl_Optimal_Hybrid3, drl_Optimal_Hybrid3));




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
    if(selectStrat==1 || selectStrat==3 || selectStrat==5)
    {
        for (int i = 0; i < sessions; i++) {
            trueGenStrategies.push_back("aca2_Optimal_Hybrid3");
        }
    }else{
        for (int i = 0; i < sessions; i++) {
            trueGenStrategies.push_back("drl_Optimal_Hybrid3");
        }
    }
  
   bool endLoop = false;
   int loopCounter = 0;

    std::shared_ptr<Strategy> strategy;
    arma::mat genProbMat;

    if(randomPair.first->getName() == randomPair.second->getName())
    {
        while(!endLoop)
        {
            //std::srand(static_cast<unsigned>(std::time(nullptr)));

            //std::cout << "Generating sim data with " << randomPair.first->getName() << " and "<< randomPair.second->getName()  << " changepoint at ses " <<  changepoint_ses << std::endl;


            std::vector<double> s0rewards = {0,0,0,0,0,0,0,5,0};
            std::vector<double> s1rewards = {0,0,0,0,0,0,0,0,5};

            // trueStrategy1->setRewardsS0(s0rewards);
            // trueStrategy1->setRewardsS1(s1rewards);

            for(int ses=0; ses < sessions; ses++)
            {
                strategy = randomPair.first;
                

                // strategy->setStateS0Credits(creditsS0_Opt);
                // strategy->setStateS1Credits(creditsS1_Opt);

                std::pair<arma::mat, arma::mat> simData = simulateTrajectory(ratdata, ses, *strategy);
                arma::mat generated_PathData_sess = simData.first;
                arma::mat generated_TurnsData_sess = simData.second;

                // trueStrategy1->getTrajectoryLikelihood(ratdata, ses);
                
                // //creditsS0_Subopt =  trueStrategy1->getS0Credits(); 
                
                // creditsS0_Opt =  trueStrategy1->getS0Credits(); 
                // creditsS1_Opt =  trueStrategy1->getS1Credits(); 

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
                //trueGenStrategies.clear();
            }

            loopCounter++;
            if(loopCounter > 50)
            {
                std::cout << "Loop counter reached 50 simulations:" << randomPair.first->getName() << " and " << randomPair.second->getName() << ". Exiting" << std::endl;
                break;
            }
        }

        genProbMat = strategy->getPathProbMat(); 
    
    }else //Generate switching simulations
    {

        std::vector<double> s0SubOptRewards = {0,0,0,0,0,0,5,0,0,0,0,0};
        std::vector<double> s0rewards = {0,0,0,0,0,0,0,5,0};
        std::vector<double> s1rewards = {0,0,0,0,0,0,0,0,5};

        std::random_device rd;
        std::mt19937 gen(rd());

        std::vector<double> initCreditsS0;
        // if(randomPair.first->getName()=="aca2_Suboptimal_Hybrid3")
        // {
        //     initCreditsS0 = {0,0,0,0.5,0,0,0.5,0,0,0,0,0.5};
        // }else{
        //     initCreditsS0 = {0,0,0,0.5,0,0,0.5,0,0,0,0,0.5};
        // }

        // randomPair.first->setStateS0Credits(initCreditsS0);

        // if(randomPair.second->getName()=="aca2_Optimal_Hybrid3")
        // {
        //     std::vector<double> initCreditsOptS0 = {0,0,0,1.5,0,0,0,1.5,0};
        //     std::vector<double> initCreditsOptS1 = {0,0,0,1.5,0,0,0,0,1.5};
        //     randomPair.second->setStateS0Credits(initCreditsOptS0);
        //     randomPair.second->setStateS1Credits(initCreditsOptS1);
        // }
        
       
        arma::mat generated_PathData_Suboptimal;
        arma::mat generated_TurnsData_Suboptimal;

        arma::mat generated_PathData_Optimal;
        arma::mat generated_TurnsData_Optimal;
        bool endLoopSuboptimal = false;
        int counterSuboptimal = 0;
        while(!endLoopSuboptimal)
        {
            if(rat=="rat_103")
            {
                changepoint_ses = 8;
                
            }else if(rat=="rat_106"){

                changepoint_ses = 3;

            }else if(rat=="rat_112")
            {
                changepoint_ses = 5;
            }else if(rat=="rat_113")
            {
                changepoint_ses = 1;
            }else if(rat=="rat_114")
            {
                changepoint_ses = 2;
            }
        
             //std::cout << "Generating sim data with " << randomPair.first->getName() << " and "<< randomPair.second->getName()  << " changepoint at ses " <<  changepoint_ses << std::endl;

            //std::srand(static_cast<unsigned>(std::time(nullptr)));
           for(int ses=0; ses < changepoint_ses; ses++)
           {

                std::pair<arma::mat, arma::mat> simData;
                arma::mat generated_PathData_sess;
                arma::mat generated_TurnsData_sess;
                strategy = randomPair.first;
                simData = simulateTrajectory(ratdata, ses, *randomPair.first);
                generated_PathData_sess = simData.first;
                generated_TurnsData_sess = simData.second;

                arma::uvec s0indices = arma::find(generated_PathData_sess.col(1) == 0); 
                arma::mat genDataS0 = generated_PathData_sess.rows(s0indices);

                arma::uvec s1indices = arma::find(generated_PathData_sess.col(1) == 1); 
                arma::mat genDataS1 = generated_PathData_sess.rows(s1indices);
                generated_PathData_Suboptimal = arma::join_cols(generated_PathData_Suboptimal, generated_PathData_sess);
                generated_TurnsData_Suboptimal = arma::join_cols(generated_TurnsData_Suboptimal, generated_TurnsData_sess);

                //trueGenStrategies.push_back(strategy->getName());
                trueGenStrategies[ses] = strategy->getName();

           } 
            
           if(check_path5(generated_PathData_Suboptimal))
            {
                std::cout << "check_path5 is successful after " << changepoint_ses << " sessions" <<std::endl;
                endLoopSuboptimal = true;

            }else if(counterSuboptimal==99)
            {
                endLoopSuboptimal = true;
            }
            else{
                endLoopSuboptimal = false;
                std::cout << "check_path5 failed. Re-generate Suboptimal trajectory: " << counterSuboptimal <<std::endl;
                generated_PathData_Suboptimal.reset();
                generated_TurnsData_Suboptimal.reset();

                randomPair.first->resetPathProbMat();
                randomPair.first->resetCredits();
            }

            counterSuboptimal++;

            if(counterSuboptimal==99)
            {
                break;
                std::cout << "Loop counter reached 100 for simulation:" << randomPair.first->getName() << " and " << randomPair.second->getName() << ". Exiting" << std::endl;

            }

        }

        bool endLoopOptimal = false;
        int counterOptimal = 0;
        while(!endLoopOptimal)
        {
            
           for(int ses=changepoint_ses; ses < sessions; ses++)
            {
                
                std::pair<arma::mat, arma::mat> simData;
                arma::mat generated_PathData_sess;
                arma::mat generated_TurnsData_sess;
                
                //Start suboptimal portion of switching simulations
                strategy = randomPair.second;
                simData = simulateTrajectory(ratdata, ses, *randomPair.second);
                generated_PathData_sess = simData.first;
                generated_TurnsData_sess = simData.second;


                arma::uvec s0indices = arma::find(generated_PathData_sess.col(1) == 0); 
                arma::mat genDataS0 = generated_PathData_sess.rows(s0indices);

                arma::uvec s1indices = arma::find(generated_PathData_sess.col(1) == 1); 
                arma::mat genDataS1 = generated_PathData_sess.rows(s1indices);
                // std::cout << "ses=" << ses << ", strategy=" << strategy->getName() << std::endl;

                //trueGenStrategies.push_back(strategy->getName());
                trueGenStrategies[ses] = strategy->getName();

                if(ses>= changepoint_ses)
                {
                    generated_PathData_Optimal = arma::join_cols(generated_PathData_Optimal, generated_PathData_sess);
                    generated_TurnsData_Optimal = arma::join_cols(generated_TurnsData_Optimal, generated_TurnsData_sess);

                }

                genProbMat = arma::join_cols(randomPair.first->getPathProbMat(), randomPair.second->getPathProbMat());             
            }

            generated_PathData = arma::join_cols(generated_PathData_Suboptimal, generated_PathData_Optimal);
            generated_TurnsData = arma::join_cols(generated_TurnsData_Suboptimal, generated_TurnsData_Optimal);

            // std::cout << "generated_PathData_Optimal size=" << generated_PathData_Optimal.n_rows << ", generated_PathData size =" << generated_PathData.n_rows << std::endl;
            endLoopOptimal = true;
            if(check_ema(generated_PathData))
            {
                std::cout << "check_ema is successful after " << changepoint_ses << " sessions" <<std::endl;
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

                generated_PathData_Optimal.reset();
                generated_TurnsData_Optimal.reset();

                randomPair.second->resetPathProbMat();
                randomPair.second->resetCredits();
                //trueGenStrategies.clear();
            }
            counterOptimal++;

            if(counterOptimal==100)
            {
                break;
                std::cout << "Loop counter reached 100 for simulation:" << randomPair.first->getName() << " and " << randomPair.second->getName() << ". Exiting" << std::endl;

            }

        }
     
    }

    std::cout << "Generated sim:" << randomPair.first->getName() << " and " << randomPair.second->getName() <<  " with changepoint = " << changepoint_ses  << std::endl;
    
    Rcpp::List l = Rcpp::List::create(Rcpp::Named("genData") = Rcpp::wrap(generated_PathData),
                                      Rcpp::Named("probMat") = genProbMat );

    R["l"] = l;
    // Save the matrix as RData using RInside
    std::string filename = "generatedData_" + std::to_string(selectStrat) + "_" + rat + "_" + run +".RData";
    
    std::string rCode = "save(l, file='" + filename + "')";
    R.parseEvalQ(rCode.c_str());

    // arma::mat trueProbMat = arma::join_cols(drl_Suboptimal_Hybrid3->getPathProbMat(),drl_Optimal_Hybrid3->getPathProbMat());
    // trueProbMat.save("genTrueProbMat_" + rat+ ".csv", arma::csv_ascii);
    
    RatData simRatdata(generated_PathData,generated_TurnsData,rat, true, trueGenStrategies,genProbMat);

    arma::mat simAllpaths = simRatdata.getPaths();
    arma::vec simSessionVec = simAllpaths.col(4);
    arma::vec simUniqSessIdx = arma::unique(simSessionVec);
    // std::cout << "simUniqSessIdx.size=" << simUniqSessIdx.size() << std::endl;

    //testSimulation(simRatdata,*randomPair.first,*randomPair.second, R);
    return simRatdata;

}


//Ignore- cannot run testSimulation as getTrajectoryLikelihood requires setting rewardsS0 & rewardsS1, which is not "possible" for suboptimal case
void testSimulation(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, RInside &R)
{
    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    
    std::map<std::string, pagmo::vector_double> testParams;

    // Add some key-value pairs using operator[]
    std::vector<double> v= {2, 8, 4, 7,  0.075,  0.235,  0.0227, 0.776};


    RatData ratSimData = generateSimulation(ratdata, suboptimalHybrid3,  optimalHybrid3, v, R, 0, "tesRun");

    arma::mat probMatTrueStrat = ratSimData.getTrueGenProb();

    std::cout << "Last 10 rows of matrix probMatTrueStrat:\n" << probMatTrueStrat.tail_rows(10) << "\n";

    // std::vector<double> v = testParams["rat_103"]; 
        double alpha_aca_subOptimal = v[4];
    double gamma_aca_subOptimal = v[5];

    double alpha_aca_optimal = v[4];
    double gamma_aca_optimal = v[5];

    //DRL params
    double alpha_drl_subOptimal = v[6];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[7];
    
    double alpha_drl_optimal = v[6];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[7];

    
   auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, 0, 0, 0, true);

    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, 0, 0, 0, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, 0, 0, 0, true);

    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> suboptimalRewardfuncs =  getRewardFunctions(ratdata, *aca2_Suboptimal_Hybrid3);
    std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> optimalRewardfuncs =  getRewardFunctions(ratdata, *aca2_Optimal_Hybrid3);

    for(int ses=0; ses < sessions; ses++)
    {
        std::cout << "subOptimal rewards S0: ";
        for (double x : suboptimalRewardfuncs.first[ses])
        {
            std::cout << x << " ";
        }    
        std::cout << "\n";

        std::cout << "Optimal rewards S0: ";
        for (double x : optimalRewardfuncs.first[ses])
        {
            std::cout << x << " ";
        }    
        std::cout << "\n";

        std::cout << "Optimal rewards S1: ";
        for (double x : optimalRewardfuncs.second[ses])
        {
            std::cout << x << " ";
        }    
        std::cout << "\n";


    }

    aca2_Suboptimal_Hybrid3->setRewardsS0(suboptimalRewardfuncs.first);
    drl_Suboptimal_Hybrid3->setRewardsS0(suboptimalRewardfuncs.first);

    aca2_Optimal_Hybrid3->setRewardsS0(optimalRewardfuncs.first);
    aca2_Optimal_Hybrid3->setRewardsS1(optimalRewardfuncs.second);

    drl_Optimal_Hybrid3->setRewardsS0(optimalRewardfuncs.first);
    drl_Optimal_Hybrid3->setRewardsS1(optimalRewardfuncs.second);
    
    double log_likelihood = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < 8)
        {
            ll_ses = drl_Suboptimal_Hybrid3->getTrajectoryLikelihood(ratSimData, ses); 
        }else{
            ll_ses = drl_Optimal_Hybrid3->getTrajectoryLikelihood(ratSimData, ses); 
        }
        log_likelihood = log_likelihood + (-1)*ll_ses;
    }
    arma::mat genProbMat = arma::join_cols(drl_Suboptimal_Hybrid3->getPathProbMat(), drl_Optimal_Hybrid3->getPathProbMat());

    std::cout << "probMatTrueStrat rows=" << probMatTrueStrat.n_rows << ", genProbMat rows = " << genProbMat.n_rows << std::endl;

    // R.assign("probMatTrueSim", probMatTrueStrat);
    // R.assign("probMatLikFunc", genProbMat);

    // Save the R variables to an RData file
    // R.parseEvalQ("save(list=c('probMatTrueSim', 'probMatLikFunc'), file='simTestProbMats.RData')");

    Rcpp::NumericMatrix probMatTrueStrat_r = Rcpp::wrap(probMatTrueStrat);
    Rcpp::NumericMatrix probMatCopyTrueStrat_r = Rcpp::wrap(genProbMat);

    // Assign them to R variables
    R["probMatTrueStrat"] = probMatTrueStrat_r;
    R["probMatCopyTrueStrat"] = probMatCopyTrueStrat_r;

    // Save them as Rdata
    R.parseEvalQ("save(probMatTrueStrat, probMatCopyTrueStrat, file = 'simTestProbMats.RData')");

    arma::mat diffMatrix = arma::abs(probMatTrueStrat-genProbMat);

    // Find the maximum value in the absolute difference matrix
    double maxDifference = arma::max(diffMatrix).eval()(0, 0);

    std::cout << "maxDifference:" << maxDifference << "\n";

    return;
}

std::vector<double> findMultiObjClusterParamsWithSim(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3) {

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
        // std::cout << "Individual " << i + 1 << ":" << std::endl;
        double lik = std::accumulate(f[idx[i]].begin(), f[idx[i]].end(), 0.0);
        // std::cout << "Fitness: [" << f[idx[i]][0] << ", " << f[idx[i]][1] <<  ", " << f[idx[i]][2] << ", " << f[idx[i]][3] << ", " << f[idx[i]][4] << ", " << f[idx[i]][5] << "]" << ", lik=" << lik << std::endl;
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


       

    return dec_vec_champion;
}

std::vector<std::vector<double>> findParamsSim(RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3)
{
    std::vector<std::string> models = {"m1","m2","m3","m4","m5","m6"};
    std::vector<std::vector<double>> modelParams;
    for(int i=0; i<6; i++)
    {
        PagmoMle pagmomle(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3,models[i]);
        std::cout << "Initialized problem class" <<std::endl;

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

        std::cout << "Final champion = " << champion_score << std::endl;
        for (auto const& i : dec_vec_champion)
            std::cout << i << ", ";
        std::cout << "\n" ;


        modelParams.push_back(dec_vec_champion);

    }

    return modelParams;
}


std::vector<double> findClusterParamsWithSimData(RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3)
{
        // Create a function to optimize
    PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
    std::cout << "Initialized problem class" <<std::endl;

    // Create a problem using Pagmo
    problem prob{pagmoprob};

    // pagmo::algorithm algo{de(3)};
    // //pagmo::algorithm algo{sade(10,2,2)};


    // archipelago archi{2u, algo, prob, 5u};

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

    pagmo::thread_bfe thread_bfe;
    pagmo::pso_gen method (2);
    //pagmo::gaco method(10);
    method.set_bfe ( pagmo::bfe { thread_bfe } );
    pagmo::algorithm algo = pagmo::algorithm { method };
    algo.set_verbosity(1);
    pagmo::population pop {prob, thread_bfe, 10 };
    // Evolve the population for 100 generations
    for ( auto evolution = 0; evolution < 2; evolution++ ) {
        pop = algo.evolve(pop);
    }

    std::vector<double> dec_vec_champion = pop.champion_x();
    std::cout << "Final champion = " << pop.champion_f()[0] << std::endl;

    // std::cout << "Final champion = " << champion_score << std::endl;
    for (auto const& i : dec_vec_champion)
        std::cout << i << ", ";
    std::cout << "\n" ;


    return dec_vec_champion;
}

void updateConfusionMatrix(std::vector<std::string> trueGenStrategies,std::vector<std::string> selectedStrategies , std::string rat, std::string run)
{

    std::string filename = "Results/confusionMatrix_" + rat+ "_" +run+ ".txt";
    std::ifstream confMat_file(filename);

    std::vector<std::string> rows = {"aca2_Suboptimal_Hybrid3", "drl_Suboptimal_Hybrid3", "aca2_Optimal_Hybrid3", "drl_Optimal_Hybrid3"};
    std::vector<std::string> columns = {"aca2_Suboptimal_Hybrid3", "drl_Suboptimal_Hybrid3", "aca2_Optimal_Hybrid3", "drl_Optimal_Hybrid3", "None"};
    //std::vector<std::vector<int>> matrix(6, std::vector<int>(7, 0));
    std::vector<std::vector<int>> matrix;

    std::vector<std::string> rownames = {"acaSubopt", "drlSubopt", "acaOpt", "drlOpt"};
    std::vector<std::string> colnames = {"acaSubopt", "drlSubopt", "acaOpt", "drlOpt", "None"};

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
        matrix = std::vector<std::vector<int>>(4, std::vector<int>(5, 0));
    }

    // Print the matrix
    std::cout << "Matrix print before update:" << std::endl;
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << '\n';
    }


    
    for(size_t i=0; i<trueGenStrategies.size();i++)
    {
        //RecordResults recordResultsSes =  allSesResults[i];
        std::string selectedStrategy = selectedStrategies[i];   //column label
        std::string trueStrategy = trueGenStrategies[i]; // rowLabel

        //std::cout << "trueStrategy=" << trueStrategy << ", idx=" << rowLabelToIndex[trueStrategy] << "; selectedStrategy=" << selectedStrategy << ", idx=" << colLabelToIndex[selectedStrategy] << std::endl;

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


void runEMOnSimData(RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3, std::vector<std::vector<double>> modelParams, bool debug, std::string run, int selectStrat, RInside& R)
{
    //// rat_103
    //std::vector<double> v = {0.11776, 0.163443, 0.0486187, 1e-7,0.475538, 0.272467, 1e-7 , 0.0639478, 1.9239e-06, 0.993274, 4.3431};
    
    ////rat_114
    //std::vector<double> v = {0.0334664, 0.351993, 0.00478871, 1.99929e-07, 0.687998, 0.380462, 9.68234e-07, 0.136651, 8.71086e-06, 0.292224, 3.95355};



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
    updateConfusionMatrix(trueGenStrategies,selectedStrategies,  rat, run);

    

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

    std::string filename = "genModelProbs_" + std::to_string(selectStrat) + "_" + rat + "_" + run +".RData";
    R["l"] = l;
    std::string rCode = "save(l, file='" + filename + "')";
    R.parseEvalQ(rCode.c_str());
                                  

    // COMMENTING OUT ARL
    // arl_suboptimal_probs.save("arl_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    // arl_optimal_probs.save("arl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);


}


void testRecovery(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, RInside &R, std::string run)
{
    // Read the params from from rat param file, e.g rat_103.txt
    // std::string rat = ratdata.getRat();
    // std::string filename = rat + ".txt";
    // std::ifstream infile(filename);
    // std::map<std::pair<std::string, bool>, std::vector<double>> ratParams;
    // boost::archive::text_iarchive ia(infile);
    // ia >> ratParams;
    // infile.close();

    ////read clusterParams.txt to get the parameters for rat
    std::string filename_cluster = "clusterMLEParams.txt";
    std::ifstream cluster_infile(filename_cluster);
    std::map<std::string, std::vector<double>> clusterParams;
    boost::archive::text_iarchive ia_cluster(cluster_infile);
    ia_cluster >> clusterParams;
    cluster_infile.close();

    // std::vector<std::vector<double>> modelParams;
    //     std::string rat = rdata.getRat();
    //     std::string filename = "clusterMLE_" + rat + ".txt" ;
    //     std::ifstream inFile(filename);
    //     std::string line;
    //     while (std::getline(inFile, line)) {
    //         std::vector<double> vec;
    //         std::istringstream iss(line);
    //         double val;
    //         while (iss >> val) {
    //             vec.push_back(val);
    //         }
    //         modelParams.push_back(vec);
    //     }
    //     inFile.close();

    std::string rat = ratdata.getRat();

    // std::vector<std::string> trueGenStrategies = {"drl_Suboptimal_Hybrid3","drl_Suboptimal_Hybrid3","drl_Suboptimal_Hybrid3"};
    // std::vector<std::string> selectedStrategies = {"aca2_Suboptimal_Hybrid3","drl_Optimal_Hybrid3","drl_Optimal_Hybrid3"};
    //updateConfusionMatrix(trueGenStrategies,selectedStrategies,  rat, run);

    //testSimulation(ratdata, suboptimalHybrid3, optimalHybrid3, R);

    for(int i=0; i < 6; i++)
    {
        //RatData ratSimData =  generateSimulationMLE(ratdata, suboptimalHybrid3, optimalHybrid3, clusterParams, R, i);
        try {
            std::vector<double> v = clusterParams[rat]; 
            std::vector<double> simClusterParams = {0.107946, 0.8247385, 0.551107, 0.182533, 0.795134};

            RatData ratSimData = generateSimulation(ratdata, suboptimalHybrid3, optimalHybrid3, simClusterParams, R, i, run);
            //std::map<std::pair<std::string, bool>, std::vector<double>> simRatParams = findParamsWithSimData(ratSimData, suboptimalHybrid3, optimalHybrid3);
            //std::vector<double> simClusterParams = findClusterParamsWithSimData(ratSimData, suboptimalHybrid3, optimalHybrid3);
            //std::vector<double> simClusterParams = findMultiObjClusterParamsWithSim(ratSimData, suboptimalHybrid3, optimalHybrid3);
            //std::vector<std::vector<double>> modelParams = findParamsSim(ratSimData, suboptimalHybrid3, optimalHybrid3);
            //runEMOnSimData(ratSimData, suboptimalHybrid3, optimalHybrid3, modelParams, true, run,i, R);
            // std::pair<std::vector<std::vector<double>>, double> q = particle_filter(1000, ratSimData, suboptimalHybrid3, optimalHybrid3, simClusterParams);
            // std::cout << "lik=" << q.second << std::endl;
            std::vector<double> params = EM(ratSimData, suboptimalHybrid3, optimalHybrid3, 500);


        }catch (const std::out_of_range& e) {
        // Handle the out_of_range exception
         std::cerr << "rat=" <<rat <<  ", i=" << i <<  ": caught out_of_range exception: " << e.what() << std::endl;
        }


    }
  
     

}




