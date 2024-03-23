#ifndef PARTICLE_FILTER_H
#define PARTICLE_FILTER_H

#include <cstdlib>
#include <pagmo/algorithm.hpp>
#include "Strategy.h"
#include "InverseRL.h"
#include <RInside.h>
#include <algorithm>
#include "BS_thread_pool.h"





using namespace Rcpp;
using namespace pagmo;

class ParticleFilter {
public:

  ParticleFilter(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_, const MazeGraph& Optimal_Hybrid3_, std::vector<double> v):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_) {

    double alpha_aca_subOptimal = v[0];
    double gamma_aca_subOptimal = v[1];

    double alpha_aca_optimal = v[0];
    double gamma_aca_optimal = v[1];

    //DRL params
    double alpha_drl_subOptimal = v[2];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[3];
    
    double alpha_drl_optimal = v[2];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[3];
    alpha_crp = 1e-6;

    initCrpProbs = {0.25,0.25,0.25,0.25};

    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, 0, 0, 0, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, 0, 0, 0, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, 0, 0, 0, true);

    strategies.push_back(aca2_Suboptimal_Hybrid3);
    strategies.push_back(aca2_Optimal_Hybrid3);

    strategies.push_back(drl_Suboptimal_Hybrid3);
    strategies.push_back(drl_Optimal_Hybrid3);



  }

  ParticleFilter(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_, const MazeGraph& Optimal_Hybrid3_, std::vector<double> v, int particleId_, double weight_):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_), particleId(particleId_), weight(weight_) {
    //std::cout << "Initializing particleId=" << particleId << std::endl;
    particleId = particleId_;
    double alpha_aca_subOptimal = v[0];
    double gamma_aca_subOptimal = v[1];

    double alpha_aca_optimal = v[0];
    double gamma_aca_optimal = v[1];

    //DRL params
    double alpha_drl_subOptimal = v[2];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[3];
    
    double alpha_drl_optimal = v[2];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[3];
    alpha_crp = 1e-6;

    initCrpProbs = {0.25,0.25,0.25,0.25};
    // normalizeCrp(initCrpProbs);

       
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, 0, 0, 0, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, 0, 0, 0, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, 0, 0, 0, true);

    strategies.push_back(aca2_Suboptimal_Hybrid3);
    strategies.push_back(aca2_Optimal_Hybrid3);

    strategies.push_back(drl_Suboptimal_Hybrid3);
    strategies.push_back(drl_Optimal_Hybrid3);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    for(int ses=0; ses<sessions; ses++)
    {
        chosenStrategy.push_back(-1);
        originalSampledStrat.push_back(-1);
        particleTrajectories.push_back(std::vector<int>(sessions, -1));
        crpPriors.push_back(std::vector<double>(4, -1));

    }

  }

  

    // ParticleFilter(const ParticleFilter& other) :
    //     ratdata(other.ratdata),
    //     Suboptimal_Hybrid3(other.Suboptimal_Hybrid3),
    //     Optimal_Hybrid3(other.Optimal_Hybrid3),
    //     weight(other.weight),
    //     particleId(other.particleId),
    //     alpha_crp(other.alpha_crp) {
    //     // Deep copy the vectors
    //     for (const auto& strategy : other.strategies) {
    //         strategies.push_back(std::make_shared<Strategy>(*strategy));
    //     }
    //     chosenStrategy = other.chosenStrategy; // Simple assignment
    // }


  // ParticleFilter& operator=(const ParticleFilter& other) {
  //       if (this != &other) {
  //         // ratdata = other.ratdata;
  //         // Suboptimal_Hybrid3 = other.Suboptimal_Hybrid3;
  //         // Optimal_Hybrid3 = other.Optimal_Hybrid3;
  //         weight = other.weight;
  //         particleId = other.particleId;
  //         alpha_crp = other.alpha_crp; 
  //         // Deep copy the vectors
  //         chosenStrategy = other.chosenStrategy; // Simple assignment
  //         originalSampledStrat = other.originalSampledStrat;
  //         stratCounts = other.stratCounts;
  //         crpPriors = other.crpPriors;
  //         initCrpProbs = other.initCrpProbs;
  //         // for (const auto& strategy : other.strategies) {
  //         //     strategies.push_back(std::make_shared<Strategy>(*strategy));
  //         // }
  //   }
  //   return *this;
  // }



  // Destructor
  ~ParticleFilter() {}

  
 void updateStratCounts(int ses)
 {
    if(ses == 0)
    {
      stratCounts.push_back({0,0,0,0});
    }
    else
    {
      std::vector<int> n(4, 0);

      for (int i = 0; i < ses; i++) {
          n[chosenStrategy[i]]++;
      }
      stratCounts.push_back(n);
    }
    
 }

 //CRP
  // std::vector<double> crpPrior(int ses)
  // {
  //   if(ses > 0)
  //   {
  //       std::vector<int> n(4, 0);

  //       // for (int i = 0; i < ses; i++) {
  //       //     n[chosenStrategy[i]]++;
  //       // }

  //       n = stratCounts[ses];
          
  //       std::vector<double> q(4, 0);

  //       for (int k = 0; k < 4; k++) {
  //           if(n[k] > 0)
  //           {
  //               q[k] = n[k] / (ses + alpha_crp);
  //           }else{
  //               q[k] = alpha_crp / (ses + alpha_crp);
  //               int zeroCount = std::count(n.begin(), n.end(), 0);
  //               q[k] = q[k]/zeroCount;

  //           }
            
  //       }

  //       return(q);

  //   }
  //   else{
  //       return initCrpProbs;
  //   }

  // }


  // std::vector<double> crpPrior2(std::vector<int> particleHistory,int ses)
  // {
    
  //   if(ses > 0)
  //   {
  //       std::vector<int> history(particleHistory.begin(), particleHistory.begin()+ses+1);

  //       // n[0] = stratCounts[ses][0];
  //       // n[1] = stratCounts[ses][1];
  //       // n[2] = stratCounts[ses][2];
  //       // n[3] = stratCounts[ses][3];

  //       std::vector<int> n(4, 0);

  //       for (int i = 0; i < ses; i++) {
  //           n[history[i]]++;
  //       }

  // //       int n_counts = std::accumulate(n.begin(), n.end(), 0.0);
          
  //       std::vector<double> q(4, 0);

  //       for (int k = 0; k < 4; k++) {
  //           if(n[k] > 0)
  //           {
  //               q[k] = n[k] / (ses + alpha_crp);
  //           }else{
  //               q[k] = alpha_crp / (ses + alpha_crp);
  //               int zeroCount = std::count(n.begin(), n.end(), 0);
  //               q[k] = q[k]/zeroCount;

  //           }
            
  //       }

  //       return(q);

  //   }
  //   else{
  //       return initCrpProbs;
  //   }

  // }


//  ////// Restricted crp
  std::vector<double> crpPrior2(std::vector<int> particleHistory,int ses)
  {
    
    if(ses > 0)
    {
        std::vector<int> history(particleHistory.begin(), particleHistory.begin()+ses+1);

        // n[0] = stratCounts[ses][0];
        // n[1] = stratCounts[ses][1];
        // n[2] = stratCounts[ses][2];
        // n[3] = stratCounts[ses][3];

        std::vector<int> n(4, 0);

        for (int i = 0; i < ses; i++) {
            n[history[i]]++;
        }

        int greaterThanZero = std::count_if(n.begin(), n.end(), [](int num) { return num > 0; });

        if(greaterThanZero < 2)
        {
          std::vector<double> q(4, 0);

          for (int k = 0; k < 4; k++) {
              if(n[k] > 0)
              {
                  q[k] = n[k] / (ses + alpha_crp);
              }else{
                  q[k] = alpha_crp / (ses + alpha_crp);
                  int zeroCount = std::count(n.begin(), n.end(), 0);
                  q[k] = q[k]/zeroCount;

              }
              
          }

          return(q);
          
        }else{
          
            std::vector<double> q(4, 0);

          for (int k = 0; k < 4; k++) {
              if(n[k] > 0)
              {
                  q[k] = (n[k] + alpha_crp / greaterThanZero )/ (ses + alpha_crp);
              }else{
                  q[k] = 0;
              }
              
          }

          normalizeCrp(q);

           return(q);
        }

          
    }
    else{
        return initCrpProbs;
    }

  }




  std::vector<double> crpPrior(int ses)
  {
        std::vector<int> n = stratCounts[ses];
        // n[0] = stratCounts[ses][0];
        // n[1] = stratCounts[ses][1];
        // n[2] = stratCounts[ses][2];
        // n[3] = stratCounts[ses][3];

        int n_counts = std::accumulate(n.begin(), n.end(), 0.0);
        if(n_counts >0 && n_counts!= ses)
        {

            std::cout << "particleId=" << particleId<<  ", ses=" <<ses << ", n_counts=" <<n_counts << ", ses = " << ses << std::endl;
            std::cout <<", n = ";
            for (auto const& i : n)
                std::cout << i << ", ";
            std::cout << "\n" ;

            std::cout << ", stratCounts[ses] = ";
            for (auto const& i : n)
                std::cout << i << ", ";
            std::cout << "\n" ;
            throw std::runtime_error("Error crp count vec is not proper");
        }
          
        //  std::cout << "particleId=" << particleId<<  ", ses=" <<ses << ", n_counts=" <<n_counts << ", ses = " << ses << std::endl;
        // std::cout << "particleId=" << particleId<<  ", ses=" <<ses << ", n = ";
        // for (auto const& i : n)
        //     std::cout << i << ", ";
        // std::cout << "\n" ;

        // std::cout << "particleId=" << particleId<<  ", ses=" <<ses <<  ", stratCounts[ses] = ";
        // for (auto const& i : n)
        //     std::cout << i << ", ";
        // std::cout << "\n" ;
        

        std::vector<double> q(4, 0);

        // If all n == 0
        if(n[0]==0 && n[1] == 0 && n[2] == 0 && n[3] == 0) //Case 1: No strategy selected
        {
            q=initCrpProbs;
        }else if((n[0] > 0 || n[2] >0) && (n[1]==0 && n[3]==0 )) //Case2: Suboptimal selected, but no optimal
        {                                                                       
            if(n[0] > 0 && n[2]==0)
            {
                q[0] = n[0] / (ses + alpha_crp);
                q[2] = 0;
            }else if(n[0] == 0 && n[2] > 0)
            {
                q[0] = 0;
                q[2] = n[2] / (ses + alpha_crp);
            }
            q[1] = alpha_crp/(2*(ses + alpha_crp));
            q[3] = alpha_crp/(2*(ses + alpha_crp));

            // std::cout << "particleId=" << particleId<<  ", ses=" <<ses <<  ", case2, q = ";
            // for (auto const& i : n)
            //     std::cout << i << ", ";
            // std::cout << "\n" ;

            normalizeCrp(q);

        }else if(((n[0] > 0 && n[2] == 0 )||(n[0] == 0 && n[2] >0)) && ((n[1] > 0 && n[3] == 0 )||(n[1] == 0 && n[3] >0))) //Case3: Suboptimal and optimal selected
        {                                                       //Either n[0]/n[2] is zero, && n[1]/n[3] is zero 
                                                                //Not true crp, requires re-weighting
            q[0] = n[0] / (ses + alpha_crp);
            q[2] = n[2] / (ses + alpha_crp);

            q[1] = n[1]/((ses + alpha_crp));
            q[3] = n[3] / (ses + alpha_crp);

            // std::replace_if(q.begin(), q.end(),
            //         [](double value) { return value == 0.0; },
            //         1e-6);

            // std::cout <<"particleId=" << particleId<<  ", ses=" <<ses <<  ", case3, q = ";
            // for (auto const& i : n)
            //     std::cout << i << ", ";
            // std::cout << "\n" ;        

            normalizeCrp(q);
        }else if((n[0] == 0 && n[2] == 0) && (n[1]>0 || n[3] > 0 )) // Case4: Optimal selected, no suboptimal
        {
            q[0] = alpha_crp/(2*(ses + alpha_crp));;
            q[2] = alpha_crp/(2*(ses + alpha_crp));;

            q[1] = n[1];
            q[3] = n[3];

            // std::replace_if(q.begin(), q.end(),
            //     [](double value) { return value == 0.0; },
            //     1e-6);

            // std::cout << "particleId=" << particleId<<  ", ses=" <<ses <<  ", case4, q = ";
            // for (auto const& i : n)
            //     std::cout << i << ", ";
            // std::cout << "\n" ;        

            
            normalizeCrp(q);
        }   

        // std::cout << "particleId=" << particleId<<  ", ses=" <<ses <<  ", q after normalizing = ";
        //     for (auto const& i : n)
        //         std::cout << i << ", ";
        //     std::cout << "\n" ;  
        
        double sum = std::accumulate(q.begin(), q.end(), 0.0);

        if (sum == 0) {
                
            std::cout << "particleId=" << particleId<<  ", ses=" <<ses << ", ses = " << ses << std::endl;
            throw std::runtime_error("Error crp prior vec is zero");
        }

        double tolerance = 1e-5;

        if(std::abs(sum - 1.0) > tolerance) {
                
            std::cout << "Error: particleId=" << particleId<<  ", ses=" <<ses << ", sum=" << sum << ", q= ";
            for (auto const& i : q)
                std::cout << i << ", ";
            std::cout << "\n" ;
            throw std::runtime_error("Error crp prior sum is not one");
        }
        
        return(q);
  }


  // std::vector<double> crpPrior2(std::vector<int> particleHistory, int ses)
  // {
  //       std::vector<int> history(particleHistory.begin(), particleHistory.begin()+ses+1);

  //       // n[0] = stratCounts[ses][0];
  //       // n[1] = stratCounts[ses][1];
  //       // n[2] = stratCounts[ses][2];
  //       // n[3] = stratCounts[ses][3];

  //       std::vector<int> n(4, 0);

  //     for (int i = 0; i < ses; i++) {
  //         n[history[i]]++;
  //     }


  //       int n_counts = std::accumulate(n.begin(), n.end(), 0.0);
  //       if(n_counts >0 && n_counts!= ses)
  //       {

  //           std::cout << "particleId=" << particleId<<  ", ses=" <<ses << ", n_counts=" <<n_counts << ", ses = " << ses << std::endl;
  //           // std::cout <<", n = ";
  //           // for (auto const& i : n)
  //           //     std::cout << i << ", ";
  //           // std::cout << "\n" ;

  //           // std::cout << ", stratCounts[ses] = ";
  //           // for (auto const& i : n)
  //           //     std::cout << i << ", ";
  //           // std::cout << "\n" ;
  //           throw std::runtime_error("Error crp count vec is not proper");
  //       }
          
  //       //  std::cout << "particleId=" << particleId<<  ", ses=" <<ses << ", n_counts=" <<n_counts << ", ses = " << ses << std::endl;
  //       // std::cout << "particleId=" << particleId<<  ", ses=" <<ses << ", n = ";
  //       // for (auto const& i : n)
  //       //     std::cout << i << ", ";
  //       // std::cout << "\n" ;

  //       // std::cout << "particleId=" << particleId<<  ", ses=" <<ses <<  ", stratCounts[ses] = ";
  //       // for (auto const& i : n)
  //       //     std::cout << i << ", ";
  //       // std::cout << "\n" ;
        

  //       std::vector<double> q(4, 0);
  //       int case_crp = -1;

  //       // If all n == 0
  //       if(n[0]==0 && n[1] == 0 && n[2] == 0 && n[3] == 0) //Case 1: No strategy selected
  //       {
  //           q=initCrpProbs;
  //           case_crp = 1;
  //       }else if((n[0] > 0 || n[2] >0) && (n[1]==0 && n[3]==0 )) //Case2: Suboptimal selected, but no optimal
  //       {                                                                       
  //           if(n[0] > 0 && n[2]==0)
  //           {
  //               q[0] = n[0] / (ses + alpha_crp);
  //               q[2] = 0;
  //           }else if(n[0] == 0 && n[2] > 0)
  //           {
  //               q[0] = 0;
  //               q[2] = n[2] / (ses + alpha_crp);
  //           }
  //           q[1] = alpha_crp/(2*(ses + alpha_crp));
  //           q[3] = alpha_crp/(2*(ses + alpha_crp));

  //           // std::cout << "particleId=" << particleId<<  ", ses=" <<ses <<  ", case2, q = ";
  //           // for (auto const& i : n)
  //           //     std::cout << i << ", ";
  //           // std::cout << "\n" ;

  //           std::replace(q.begin(), q.end(), 0.0, 1e-6);
  //           normalizeCrp(q);
  //           case_crp = 2;

  //       }else if(((n[0] > 0 && n[2] == 0 )||(n[0] == 0 && n[2] >0)) && ((n[1] > 0 && n[3] == 0 )||(n[1] == 0 && n[3] >0))) //Case3: Suboptimal and optimal selected
  //       {                                                       //Either n[0]/n[2] is zero, && n[1]/n[3] is zero 
  //                                                               //Not true crp, requires re-weighting
  //           q[0] = n[0] / (ses + alpha_crp);
  //           q[2] = n[2] / (ses + alpha_crp);

  //           q[1] = n[1]/((ses + alpha_crp));
  //           q[3] = n[3] / (ses + alpha_crp);

  //           // std::replace_if(q.begin(), q.end(),
  //           //         [](double value) { return value == 0.0; },
  //           //         1e-6);

  //           // std::cout <<"particleId=" << particleId<<  ", ses=" <<ses <<  ", case3, q = ";
  //           // for (auto const& i : n)
  //           //     std::cout << i << ", ";
  //           // std::cout << "\n" ;        
  //           std::replace(q.begin(), q.end(), 0.0, 1e-6);
  //           normalizeCrp(q);
  //           case_crp = 3;
  //       }else if((n[0] == 0 && n[2] == 0) && (n[1]>0 || n[3] > 0 )) // Case4: Optimal selected, no suboptimal
  //       {
  //           q[0] = alpha_crp/(2*(ses + alpha_crp));;
  //           q[2] = alpha_crp/(2*(ses + alpha_crp));;

  //           q[1] = n[1];
  //           q[3] = n[3];

  //           // std::replace_if(q.begin(), q.end(),
  //           //     [](double value) { return value == 0.0; },
  //           //     1e-6);

  //           // std::cout << "particleId=" << particleId<<  ", ses=" <<ses <<  ", case4, q = ";
  //           // for (auto const& i : n)
  //           //     std::cout << i << ", ";
  //           // std::cout << "\n" ;        

  //           std::replace(q.begin(), q.end(), 0.0, 1e-6);
  //           normalizeCrp(q);
  //           case_crp = 3;
  //       }   

  //       // std::cout << "particleId=" << particleId<<  ", ses=" <<ses <<  ", q after normalizing = ";
  //       //     for (auto const& i : n)
  //       //         std::cout << i << ", ";
  //       //     std::cout << "\n" ;  
        
  //       double sum = std::accumulate(q.begin(), q.end(), 0.0);

  //       // if (sum == 0) {
                
  //       //     std::cout << "particleId=" << particleId<<  ", ses=" <<ses << ", case=" << case_crp << ", n:";
  //       //     for (auto const& i : n)
  //       //         std::cout << i << ", ";
  //       //     std::cout << "\n" ; 

  //       //     std::cout << "particleId=" << particleId<<  ", ses=" <<ses << ", particleHistory:";
  //       //      for (auto const& i : particleHistory)
  //       //         std::cout << i << ", ";
  //       //     std::cout << "\n" ; 

  //       //     //throw std::runtime_error("Error crp prior vec is zero");
  //       // }

  //       // double tolerance = 1e-5;

  //       // if(std::abs(sum - 1.0) > tolerance) {
                
  //       //     std::cout << "Error: particleId=" << particleId<<  ", ses=" <<ses << ", sum=" << sum << ", q= ";
  //       //     for (auto const& i : q)
  //       //         std::cout << i << ", ";
  //       //     std::cout << "\n" ;
  //       //     throw std::runtime_error("Error crp prior sum is not one");
  //       // }
        
  //       return(q);
  // }


  

  int sample_crp(std::vector<double> q) {

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    double u = dis(gen);

    // Find the smallest index k such that the cumulative sum of q up to k is greater than or equal to u
    int k = 0;
    double s = 0;
    while (s < u) {
        k++;
        s += q[k - 1];
    }

    // Return k as the sample
    return k;

  }

  // A function that normalizes a vector of probabilities to sum to one
    void normalizeCrp(std::vector<double>& p) {
    double sum = 0;
    for (double x : p) {
        sum += x;
    }
    for (double& x : p) {
        x /= sum;
    }
    if (sum == 0) {
                        
            throw std::runtime_error("Error weight vec is zero");
        }
    }



  double getSesLikelihood(int strat, int ses)
  {
    double ll_ses  = strategies[strat]->getTrajectoryLikelihood(ratdata, ses); 
    if (std::isinf(exp(ll_ses))) {     
        std::cout << "ses=" << ses << ", strat=" << strat << ", inf lik, ll_ses=" << ll_ses << ", exp(ll_ses)=" << exp(ll_ses) << std::endl;           
        
        std::vector<double> s0Credits = strategies[strat]->getS0Credits();
        std::cout << "s0Credits: ";
        for (auto const& n : s0Credits)
                    std::cout << n << ", ";
                std::cout << "\n" ;
        if(strategies[strat]->getOptimal())
        {
          std::vector<double> s1Credits = strategies[strat]->getS1Credits();
          std::cout << "s1Credits: ";
          for (auto const& n : s1Credits)
            std::cout << n << ", ";
          std::cout << "\n" ;
        }
        
        
        throw std::runtime_error("likelihood is inf");
    }
    return(exp(ll_ses));
  }


  double getSesLogLikelihood(int strat, int ses)
  {
    double ll_ses  = strategies[strat]->getTrajectoryLikelihood(ratdata, ses); 
    // if (exp(ll_ses)==0) {                    
    //     throw std::runtime_error("likelihood is zero");
    // }
    return(ll_ses);
  }


  void addAssignment(int ses, int selectedStrat)
  {
    chosenStrategy[ses] = selectedStrat;
  }

  std::vector<int> getChosenStratgies()
  {
    return chosenStrategy;
  }

 std::vector<std::shared_ptr<Strategy>> getStrategies()
 {
    return strategies;
 }

 void backUpStratCredits()
  {
    stratCreditBackUps.clear();
    for(int i=0; i<strategies.size();i++)
    {
        std::vector<double> s0Credits = strategies[i]->getS0Credits();
        std::vector<double> s1Credits;
        if(strategies[i]->getOptimal())
        {
            s1Credits = strategies[i]->getS1Credits();
        }

        std::pair<std::vector<double>,std::vector<double>> stratCredits  = std::make_pair(s0Credits,s1Credits);
        stratCreditBackUps.push_back(stratCredits);
    }
    
  }

  std::vector<std::pair<std::vector<double>,std::vector<double>>>& getStratCreditBackUps()
  {
    return stratCreditBackUps;
  }

  void rollBackCredits()
  {
    for(int i=0; i<strategies.size();i++)
    {
        std::vector<double> s0Creditbackup = stratCreditBackUps[i].first;
        strategies[i]->setStateS0Credits(s0Creditbackup);
        if(strategies[i]->getOptimal())
        {
            std::vector<double> s1Creditbackup = stratCreditBackUps[i].second;
            strategies[i]->setStateS1Credits(s1Creditbackup);
        }     
    }
    stratCreditBackUps.clear();
  }


  int getParticleId()
  {
    return particleId;
  }

  void setStrategies(std::vector<std::shared_ptr<Strategy>> strategies_)
  {
    for(int i=0; i<strategies_.size();i++)
    {
        strategies[i]->setStateS0Credits(strategies_[i]->getS0Credits());
        if(strategies[i]->getOptimal())
        {
            strategies[i]->setStateS1Credits(strategies_[i]->getS1Credits());
        }

        
    }
  }

  void resetStrategies()
  {
    for(int i=0; i<strategies.size();i++)
    {
        strategies[i]->resetCredits();        
    }
  }

  void setStratBackups(std::vector<std::pair<std::vector<double>,std::vector<double>>>& stratBackUps)
  {
    for(int i=0; i<stratBackUps.size();i++)
    {
        strategies[i]->setStateS0Credits(stratBackUps[i].first);
        if(strategies[i]->getOptimal())
        {
            strategies[i]->setStateS1Credits(stratBackUps[i].second);
        }

        
    }
  }

  void clearStratBackUps()
  {
    stratCreditBackUps.clear();
  }

  void setChosenStrategies(std::vector<int> chosenStrategy_) 
  {
    chosenStrategy = chosenStrategy_;
  }

  void backUpChosenStrategies() 
  {
    chosenStrategy_bkp = chosenStrategy;
  }

  std::vector<int> getChosenStrategyBackups()
  {
    return chosenStrategy_bkp;
  }



  void addCrpPrior(std::vector<double> crpPrior, int ses)
  {
    crpPriors[ses]=crpPrior;
  }

  void setCrpPriors(std::vector<std::vector<double>> crpPriors_)
  {
    crpPriors = crpPriors_;
  }
  std::vector<std::vector<double>> getCrpPriors()
  {
    return(crpPriors);
  }

  void setLikelihood(double lik)
  {
    likelihoods.push_back(lik);
  }

  std::vector<double> getLikelihoods()
  {
    return likelihoods;
  }
  std::vector<int> getOriginalSampledStrats()
  {
    return originalSampledStrat;
  }

  void addOriginalSampledStrat(int ses, int sampled_strat)
  {
    originalSampledStrat[ses] = sampled_strat;
  }

  void setOriginalSampledStrats(std::vector<int> origSampledStrats)
  {
    originalSampledStrat = origSampledStrats;
  }

  std::vector<std::vector<int>> getStratCounts()
  {
    return stratCounts;
  }

  void setStratCounts(std::vector<std::vector<int>> stratCounts_)
  {
    stratCounts = stratCounts_;
  }


  void addParticleTrajectory(std::vector<int> particleTrajectory, int session)
  {
    particleTrajectories[session] = particleTrajectory;
  }


  std::vector<std::vector<int>> getParticleTrajectories()
  {
    return particleTrajectories;
  }

  

  // }


private:
  // Members
  const RatData& ratdata;
  const MazeGraph& Suboptimal_Hybrid3;
  const MazeGraph& Optimal_Hybrid3;
  //mutable std::vector<std::atomic<std::pair<double, std::vector<double>>>> indexedValues;
  // mutable std::mutex  vectorMutex;

  std::vector<std::shared_ptr<Strategy>> strategies;
  std::vector<int> chosenStrategy;
  std::vector<int> chosenStrategy_bkp;
  std::vector<int> originalSampledStrat;
  double weight;
  int particleId;
  double alpha_crp;
  std::vector<std::vector<double>> crpPriors;
  std::vector<double> likelihoods;
  std::vector<double> initCrpProbs;
  std::vector<std::vector<int>> stratCounts;
  std::vector<std::pair<std::vector<double>,std::vector<double>>> stratCreditBackUps;
  std::vector<std::vector<int>> particleTrajectories;
      
};




std::tuple<std::vector<std::vector<double>>, double, std::vector<int>> particle_filter(int N, const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, std::vector<double> v,BS::thread_pool& pool);
void print_vector(std::vector<double> v);
double M_step(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> smoothed_w, std::vector<double> params, BS::thread_pool& pool);
std::vector<double> EM(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, int N,BS::thread_pool& pool);
std::vector<double> Mle(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, int N,BS::thread_pool& pool);
std::tuple<std::vector<std::vector<double>>, double, std::vector<std::vector<double>>> particle_filter_new(int N, std::vector<ParticleFilter>&  particleFilterVec, const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3,BS::thread_pool& pool);
void testQFunc(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, BS::thread_pool& pool, RInside & R);
double M_step2(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::tuple<std::vector<std::vector<double>>, std::vector<ParticleFilter>, std::vector<std::vector<int>>> smoothedRes, std::vector<double> params, BS::thread_pool& pool);
std::tuple<std::vector<std::vector<double>>, std::vector<ParticleFilter>, std::vector<std::vector<int>>> E_step2(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::vector<double> params, BS::thread_pool& pool);
std::pair<std::vector<std::vector<double>>, double> fapf(int N, std::vector<ParticleFilter> &particleFilterVec, const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, BS::thread_pool& pool);
int sample(std::vector<double> p);
void normalize(std::vector<double> &p);
std::vector<double> stratifiedResampling(std::vector<double>&particleWeights);
std::vector<std::vector<int>> E_step3(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, int M, std::vector<double> params, BS::thread_pool& pool);
double M_step3(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int M, int k, double gamma, std::vector<std::vector<int>> smoothedTrajectories, std::vector<double> params, BS::thread_pool& pool);
std::tuple<std::vector<std::vector<double>>, double, std::vector<std::vector<int>>> cpf_as(int N, std::vector<ParticleFilter> &particleFilterVec, const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, std::vector<int> x_cond ,BS::thread_pool& pool);
double M_step4(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::vector<std::vector<int>> smoothedTrajectories, std::vector<std::vector<double>> filteredWeights, std::vector<double> params, BS::thread_pool& pool);
std::vector<double> SAEM(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, BS::thread_pool& pool);
std::vector<double> systematicResampling(const std::vector<double> &particleWeights);

#endif