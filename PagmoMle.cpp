#include "PagmoMle.h"
#include "InverseRL.h"


vector_double PagmoMle::fitness(const vector_double& v) const
{
    double alpha = 0;
    double gamma = 0;
    double lambda = 0;
    double crpAlpha = 0;
    double phi = 0;
    double eta = 0;
    int n = 0;

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    double negloglikelihood = 0;
    
    
    
    if(model == "m1")
    {
        double alpha_aca_subOptimal = v[0];
        double gamma_aca_subOptimal = v[1];

        double alpha_aca_optimal = v[0];
        double gamma_aca_optimal = v[1];

        n = static_cast<int>(std::floor(v[2]));

        double phi = v[3];

        auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, 0, 0, 0, false);
        auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, 0, 0, 0, true);

        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> suboptimalRewardfuncs =  getRewardFunctions(ratdata, *aca2_Suboptimal_Hybrid3, phi);
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> optimalRewardfuncs =  getRewardFunctions(ratdata, *aca2_Optimal_Hybrid3, phi);

        aca2_Suboptimal_Hybrid3->setRewardsS0(suboptimalRewardfuncs.first);

        aca2_Optimal_Hybrid3->setRewardsS0(optimalRewardfuncs.first);
        aca2_Optimal_Hybrid3->setRewardsS1(optimalRewardfuncs.second);


        double ll1 = 0;
        for(int ses=0; ses < sessions; ses++)
        {
            double ll_ses = 0;
            if(ses < n)
            {
            ll_ses  = aca2_Suboptimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
            }else{
            ll_ses  = aca2_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
            }
            
            ll_ses = ll_ses*(-1);
            ll1 = ll1 + ll_ses;
        }

        negloglikelihood = ll1;

    }else if (model=="m2")
    {
        double alpha_aca_subOptimal = v[0];
        double gamma_aca_subOptimal = v[1];


        double alpha_drl_optimal = v[2];
        double beta_drl_optimal = v[3];
        double lambda_drl_optimal = v[4];

        n = static_cast<int>(std::floor(v[5]));

        double phi = v[6];

        auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, 0, 0, 0, false);
        auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, 0, 0, 0, true);

        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> suboptimalRewardfuncs =  getRewardFunctions(ratdata, *aca2_Suboptimal_Hybrid3, phi);
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> optimalRewardfuncs =  getRewardFunctions(ratdata, *drl_Optimal_Hybrid3, phi);

        aca2_Suboptimal_Hybrid3->setRewardsS0(suboptimalRewardfuncs.first);

        drl_Optimal_Hybrid3->setRewardsS0(optimalRewardfuncs.first);
        drl_Optimal_Hybrid3->setRewardsS1(optimalRewardfuncs.second);


        double ll2 = 0;
        for(int ses=0; ses < sessions; ses++)
        {
            double ll_ses = 0;
            if(ses < n)
            {
            ll_ses  = aca2_Suboptimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
            }else{
            ll_ses  = drl_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
            }
            
            ll_ses = ll_ses*(-1);
            ll2 = ll2 + ll_ses;
        }

        negloglikelihood = ll2;

    }else if (model=="m3")
    {
        double alpha_drl_subOptimal = v[0];
        double beta_drl_subOptimal = v[1];
        double lambda_drl_subOptimal = v[2];

        double alpha_aca_optimal = v[3];
        double gamma_aca_optimal = v[4];

        n = static_cast<int>(std::floor(v[5]));

        double phi = v[6];

   
        auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, 0, 0, 0, false);
        auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, 0, 0, 0, true);

        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> suboptimalRewardfuncs =  getRewardFunctions(ratdata, *drl_Suboptimal_Hybrid3, phi);
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> optimalRewardfuncs =  getRewardFunctions(ratdata, *aca2_Optimal_Hybrid3, phi);

        drl_Suboptimal_Hybrid3->setRewardsS0(suboptimalRewardfuncs.first);

        aca2_Optimal_Hybrid3->setRewardsS0(optimalRewardfuncs.first);
        aca2_Optimal_Hybrid3->setRewardsS1(optimalRewardfuncs.second);

        double ll3 = 0;
        for(int ses=0; ses < sessions; ses++)
        {
            double ll_ses = 0;
            if(ses < n)
            {
            ll_ses  = drl_Suboptimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
            }else{
            ll_ses  = aca2_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
            }
            
            ll_ses = ll_ses*(-1);
            ll3 = ll3 + ll_ses;
        }

        negloglikelihood = ll3;

    
    }else if (model=="m4")
    {
        double alpha_drl_subOptimal = v[0];
        double beta_drl_subOptimal = v[1];
        double lambda_drl_subOptimal = v[2];

        double alpha_drl_optimal = v[0];
        double beta_drl_optimal = v[1];
        double lambda_drl_optimal = v[2];

        n = static_cast<int>(std::floor(v[3]));

        double phi = v[4];

        auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, 0, 0, 0, false);
        auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, 0, 0, 0, true);

        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> suboptimalRewardfuncs =  getRewardFunctions(ratdata, *drl_Suboptimal_Hybrid3, phi);
        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> optimalRewardfuncs =  getRewardFunctions(ratdata, *drl_Optimal_Hybrid3, phi);

        drl_Suboptimal_Hybrid3->setRewardsS0(suboptimalRewardfuncs.first);

        drl_Optimal_Hybrid3->setRewardsS0(optimalRewardfuncs.first);
        drl_Optimal_Hybrid3->setRewardsS1(optimalRewardfuncs.second);


        double ll4 = 0;
        for(int ses=0; ses < sessions; ses++)
        {
            double ll_ses = 0;
            if(ses < n)
            {
                ll_ses  = drl_Suboptimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
            }else{
                ll_ses  = drl_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
            }
            
            ll_ses = ll_ses*(-1);
            ll4 = ll4 + ll_ses;
        }

        negloglikelihood = ll4;

    }else if (model=="m5")
    {
        double alpha_aca_optimal = v[0];
        double gamma_aca_optimal = v[1];

        double phi = v[2];

        auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, 0, 0, 0, true);

        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> optimalRewardfuncs =  getRewardFunctions(ratdata, *aca2_Optimal_Hybrid3, phi);

        aca2_Optimal_Hybrid3->setRewardsS0(optimalRewardfuncs.first);
        aca2_Optimal_Hybrid3->setRewardsS1(optimalRewardfuncs.second);

        double ll5 = 0;
        for(int ses=0; ses < sessions; ses++)
        {
            double ll_ses  = aca2_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
            
            ll_ses = ll_ses*(-1);
            ll5 = ll5 + ll_ses;
        }

        negloglikelihood = ll5;
    }else if (model=="m6")
    {
        double alpha_drl_optimal = v[0];
        double beta_drl_optimal = v[1];
        double lambda_drl_optimal = v[2];

        double phi = v[3];

        auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, 0, 0, 0, true);

        std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> optimalRewardfuncs =  getRewardFunctions(ratdata, *drl_Optimal_Hybrid3, phi);

        drl_Optimal_Hybrid3->setRewardsS0(optimalRewardfuncs.first);
        drl_Optimal_Hybrid3->setRewardsS1(optimalRewardfuncs.second);

        double ll6 = 0;
        for(int ses=0; ses < sessions; ses++)
        {
            double ll_ses  = drl_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
            
            ll_ses = ll_ses*(-1);
            ll6 = ll6 + ll_ses;
        }
        negloglikelihood = ll6;

    }
    
    return{negloglikelihood};

}

std::pair<vector_double, vector_double> PagmoMle::get_bounds() const 
{
    std::pair<vector_double, vector_double> bounds;

    if(model == "m1")
    {
        bounds.first={0,0,2,0};
        bounds.second={1,1,10,1};
    }else if(model == "m2")
    {
        bounds.first={0,0,0,0,0,2,0};
        bounds.second={1,1,1,1,1,10,1};
    }else if (model=="m3")
    {
        bounds.first={0,0,0,0,0,2,0};
        bounds.second={1,1,1,1,1,10,1};
    }else if(model == "m4")
    {
        bounds.first={0,0,0,2,0};
        bounds.second={1,1,1,10,1};
    }else if(model == "m5")
    {
        bounds.first={0,0,0};
        bounds.second={1,1,1};
    }else if(model == "m6")
    {
        bounds.first={0,0,0,0};
        bounds.second={1,1,1,1};
    }
    
    // std::cout << "bounds.first:\n";
    // for (double element : bounds.first) {
    //     std::cout << element << " ";
    // }
    // std::cout << std::endl;
    // std::cout << "bounds.second:\n";
    // for (double element : bounds.second) {
    //     std::cout << element << " ";
    // }
    // std::cout << std::endl;
    return(bounds);

}