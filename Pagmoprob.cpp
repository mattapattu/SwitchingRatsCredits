#include "Pagmoprob.h"
#include "RecordResults.h"
#include "Strategy.h"
#include "InferStrategy.h"
#include <algorithm>

pagmo::vector_double PagmoProb::fitness(const pagmo::vector_double& v) const
{
    double alpha_aca_subOptimal = v[0];
    double gamma_aca_subOptimal = v[1];

    double alpha_aca_optimal = v[2];
    double gamma_aca_optimal = v[3];

    // //ARL params
    // double alpha_arl_subOptimal = v[4];
    // double beta_arl_subOptimal = 1e-7;
    // double lambda_arl_subOptimal = v[5];
    
    // double alpha_arl_optimal = v[6];
    // double beta_arl_optimal = 1e-7;
    // double lambda_arl_optimal = v[7];
 
    // //DRL params
    double alpha_drl_subOptimal = v[4];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[5];
    
    double alpha_drl_optimal = v[6];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[7];

    
    double phi = v[8];
    double crpAlpha = v[9];
    double eta = v[10];
    
    
       
    // std::vector<double> rewardsS0_aca = {0,0,0,0,0,0,0,rS0_opt,0};
    // std::vector<double> rewardsS1_aca = {0,0,0,0,0,0,0,0,rS1_opt};

    // std::vector<double> rewardsS0_arl = {0,0,0,0,0,0,0,rS0_opt,0};
    // std::vector<double> rewardsS1_arl = {0,0,0,0,0,0,0,0,rS1_opt};
    
    // std::vector<double> rewardsS0_drl = {0,0,0,0,0,0,0,rS0_opt,0};
    // std::vector<double> rewardsS1_drl = {0,0,0,0,0,0,0,0,rS1_opt};
  
    // std::vector<double> rewardsS0_subopt_aca = {0,0,0,0,0,0,rS0_opt,rS1_opt,0,0,0,0};
    // std::vector<double> rewardsS0_subopt_arl = {0,0,0,0,0,0,rS0_opt,rS1_opt,0,0,0,0};
    // std::vector<double> rewardsS0_subopt_drl = {0,0,0,0,0,0,rS0_opt,rS1_opt,0,0,0,0};


    //std::cout << "alpha_aca_subOptimal=" << alpha_aca_subOptimal << ", gamma_aca_subOptimal=" << gamma_aca_subOptimal << ", alpha_aca_optimal=" << alpha_aca_optimal << ", gamma_aca_optimal=" << gamma_aca_optimal << std::endl;
    
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, crpAlpha, phi, eta, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, crpAlpha, phi, eta, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, crpAlpha, phi, eta, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, crpAlpha, phi, eta, true);

    // COMMENTING OUT ARL
    // auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"arl", alpha_arl_subOptimal, beta_arl_subOptimal, lambda_arl_subOptimal, crpAlpha, phi, eta, false);
    // auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"arl",alpha_arl_optimal, beta_arl_optimal, lambda_arl_optimal, crpAlpha, phi, eta, true);

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

    RecordResults allResults("None", {}, {}, {}, {}, {});

    for(int ses=0; ses < sessions; ses++)
    {
        
        if(ses==0)
        {
            initRewardVals(ratdata, ses, strategies);
        }

        estep_cluster_update(ratdata, ses, strategies, cluster, last_choice,false,allResults);
        mstep(ratdata, ses, strategies, cluster,false,allResults);      

    }

    double marginal_lik = 0;
    for (const auto& strategyPtr : strategies) {
        std::vector<double> strat_posteriors = strategyPtr->getMarginalLikelihood();
        marginal_lik = marginal_lik + std::accumulate(strat_posteriors.begin(), strat_posteriors.end(), 0.0);
    }
    marginal_lik = marginal_lik* (-1);

    if(marginal_lik == 0)
    {
        std::cout << "marginal_lik=0, v=";
        // Print the vector elements
        for (double x : v)
        {
            std::cout << x << ", ";
        }
        std::cout << "\n";


        //marginal_lik = 100000; // Penalize to prevent zero likelihoods
    }

    // Count the number of optimal and suboptimal strategies
    double optimalCount = std::count_if(cluster.begin(), cluster.end(),
            [](const std::string& stratName) {
                return stratName.find("Optimal") != std::string::npos;
            });

    double suboptimalCount = std::count_if(cluster.begin(), cluster.end(),
        [](const std::string& stratName) {
            return stratName.find("Suboptimal") != std::string::npos;
        });

    // // // Check if the vector contains either one optimal and one suboptimal strategy, or only one optimal strategy
    // if (!((optimalCount == 1 && suboptimalCount == 1) || (optimalCount == 1 && suboptimalCount == 0)))
    // {
    //     std::cout << "Cluster not complying with constraints\n";
    //     //marginal_lik = 100000;
    // }else{
    //     std::cout << "Cluster contains either one optimal and one suboptimal strategy, or only one optimal strategy, marginal_lik=" << marginal_lik << std::endl  ;
    // }


    double cluster_size = cluster.size();

    std::vector<double> params(v.begin(), v.end());

    //std::cout << "marginal_lik=" << marginal_lik << ", optimalCount=" << optimalCount << ", suboptimalCount=" << suboptimalCount << ", cluster_size=" << cluster_size << std::endl;

    // if(marginal_lik < 10000 && optimalCount==1 && suboptimalCount<=1 && cluster_size>=1)
    // {
    //     std::cout << "Adding to vector, indexedValues.size=" << indexedValues.size() << std::endl;
    //     addIndexedValues(std::make_pair(marginal_lik, params));

    // }

    bool isSuboptimalFollowedByOptimal = true;
    if(cluster_size == 2)
    {
        bool isFirstSuboptimal = cluster[0].find("Suboptimal") != std::string::npos;
        //bool isSecondOptimal = cluster[1].find("Optimal") != std::string::npos;

        if(!isFirstSuboptimal){
            isSuboptimalFollowedByOptimal = false;
        }

    }

    double equality2 = isSuboptimalFollowedByOptimal-1;
    
    return{marginal_lik};

}

std::pair<pagmo::vector_double, pagmo::vector_double> PagmoProb::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;

    //bounds.first={1e-2,0.5,1e-2,0.5,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8};
    //bounds.first={1e-2,0.5,1e-2,0.8,1e-2,1e-2,1e-2,1e-2,0.1,1e-8,1e-8};
    bounds.first={1e-8,0.5,1e-8,0.5,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8};
    bounds.second={1,1,1,1,1,1,1,1,1,5,5};

    // bounds.first={0,0,0,0,0,0,0,0,1e-6,0,0};
    //bounds.second={1,1,1,1,1,1,1,1,1e-3,1,10};

    return(bounds);
  }
