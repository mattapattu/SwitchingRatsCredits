#include "PagmoMultiObjCluster.h"
#include "RecordResults.h"
#include "InferStrategy.h"

pagmo::vector_double PagmoMultiObjCluster::fitness(const pagmo::vector_double& v) const
{
    double crpAlpha = v[0];
    double phi = v[1];
    double eta = 0;

    double alpha_aca_subOptimal = v[2];
    double gamma_aca_subOptimal = v[3];

    double alpha_aca_optimal = v[4];
    double gamma_aca_optimal = v[5];

    //ARL params
    double alpha_arl_subOptimal = v[6];
    double beta_arl_subOptimal = 1e-7;
    double lambda_arl_subOptimal = v[7];
    
    double alpha_arl_optimal = v[8];
    double beta_arl_optimal = 1e-7;
    double lambda_arl_optimal = v[9];
 
    //DRL params
    double alpha_drl_subOptimal = v[10];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[11];
    
    double alpha_drl_optimal = v[12];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[13];


    // double rS0_subopt = v[14];
    // double rS1_subopt = v[15];
    // double rS0_opt = v[16];
    // double rS1_opt = v[17];

       
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
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, crpAlpha, phi, eta, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, crpAlpha, phi, eta, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, crpAlpha, phi, eta, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, crpAlpha, phi, eta, true);

    auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"arl", alpha_arl_subOptimal, beta_arl_subOptimal, lambda_arl_subOptimal, crpAlpha, phi, eta, false);
    auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"arl",alpha_arl_optimal, beta_arl_optimal, lambda_arl_optimal, crpAlpha, phi, eta, true);

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

    strategies.push_back(arl_Suboptimal_Hybrid3);
    strategies.push_back(arl_Optimal_Hybrid3);



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

    // for(int ses=0; ses < sessions; ses++)
    // {
    //     compute_marginal_likelihoods(ratdata, ses, strategies);
    // }

    
    double alpha_aca_suboptimal_learned = params.find(std::make_pair("aca2", false))->second[0];
    double gamma_aca_suboptimal_learned = params.find(std::make_pair("aca2", false))->second[1];

    double alpha_arl_suboptimal_learned = params.find(std::make_pair("arl", false))->second[0];
    double lambda_arl_suboptimal_learned = params.find(std::make_pair("arl", false))->second[1];
     
    double alpha_drl_suboptimal_learned = params.find(std::make_pair("drl", false))->second[0];
    double lambda_drl_suboptimal_learned = params.find(std::make_pair("drl", false))->second[1];

    
    double alpha_aca_optimal_learned = params.find(std::make_pair("aca2", true))->second[0];
    double gamma_aca_optimal_learned = params.find(std::make_pair("aca2", true))->second[1];

    double alpha_arl_optimal_learned = params.find(std::make_pair("arl", true))->second[0];
    double lambda_arl_optimal_learned = params.find(std::make_pair("arl", true))->second[1];
     
    double alpha_drl_optimal_learned = params.find(std::make_pair("drl", true))->second[0];
    double lambda_drl_optimal_learned = params.find(std::make_pair("drl", true))->second[1];

    //Suboptimal diff
    double diff_alpha_aca_subopt = std::abs(alpha_aca_suboptimal_learned-alpha_aca_subOptimal)/alpha_aca_suboptimal_learned;
    double diff_gamma_aca_subopt = std::abs(gamma_aca_suboptimal_learned-gamma_aca_subOptimal)/gamma_aca_suboptimal_learned;

    double diff_alpha_arl_subopt = std::abs(alpha_arl_suboptimal_learned-alpha_arl_subOptimal)/alpha_arl_suboptimal_learned;
    double diff_lambda_arl_subopt = std::abs(lambda_arl_suboptimal_learned-lambda_arl_subOptimal)/lambda_arl_suboptimal_learned;

    double diff_alpha_drl_subopt = std::abs(alpha_drl_suboptimal_learned-alpha_drl_subOptimal)/alpha_drl_suboptimal_learned;
    double diff_lambda_drl_subopt = std::abs(lambda_drl_suboptimal_learned-lambda_drl_subOptimal)/lambda_drl_suboptimal_learned;

    //Optimal diff
    double diff_alpha_aca = std::abs(alpha_aca_optimal_learned-alpha_aca_optimal)/alpha_aca_optimal_learned;
    double diff_gamma_aca = std::abs(gamma_aca_optimal_learned-gamma_aca_optimal)/gamma_aca_optimal_learned;

    double diff_alpha_arl = std::abs(alpha_arl_optimal_learned-alpha_arl_optimal)/alpha_arl_optimal_learned;
    double diff_lambda_arl = std::abs(lambda_arl_optimal_learned-lambda_arl_optimal)/lambda_arl_optimal_learned;

    double diff_alpha_drl = std::abs(alpha_drl_optimal_learned-alpha_drl_optimal)/alpha_drl_optimal_learned;
    double diff_lambda_drl = std::abs(lambda_drl_optimal_learned-lambda_drl_optimal)/lambda_drl_optimal_learned;


    double marginal_lik = 0;
    for (const auto& strategyPtr : strategies) {
        std::vector<double> strat_posteriors = strategyPtr->getMarginalLikelihood();
        marginal_lik = marginal_lik +  std::accumulate(strat_posteriors.begin(), strat_posteriors.end(), 0.0);
    }

    marginal_lik = marginal_lik* (-1);
    if(marginal_lik == 0)
    {
       marginal_lik = 100000; // Penalize to prevent zero likelihoods
    }


    return{diff_alpha_aca, diff_gamma_aca, diff_alpha_arl, diff_lambda_arl, diff_alpha_drl, diff_lambda_drl, marginal_lik};

}

std::pair<pagmo::vector_double, pagmo::vector_double> PagmoMultiObjCluster::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;

    bounds.first={0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    bounds.second={20,1,1,1,1,1,1,1,1,1,1,1,1,1};

    return(bounds);
  }

vector_double::size_type PagmoMultiObjCluster::get_nobj() const {
    return 7u;
}

