#include "InverseRL.h"
#include "Pagmoprob.h"
#include <RInside.h>
#include <limits>


std::vector<double> computePrior(std::vector<std::shared_ptr<Strategy>>  strategies, std::vector<std::string>& cluster, int ses, std::string& last_choice)
{
    int strategies_size = strategies.size();
    std::vector<double> priors(strategies_size, 0);
    double alpha = strategies[0]->getAlpha();
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
            // Use the strategy object here
            std::string name = strategyPtr->getName();
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
            }else{
                priors[i] = -1;
            }

        }

        double alpha_i = alpha/(strategies_size - cluster.size());
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

void estep_cluster_update(const RatData& ratdata, int ses, std::vector<std::shared_ptr<Strategy>>  strategies, std::vector<std::string>& cluster, std::string& last_choice ,bool logger=false)
{
    std::vector<double> likelihoods;
    double max_posterior = 0.0;
    std::vector<std::string> most_lik_strategies;


    int strategies_size = strategies.size();
    std::vector<double> posteriors(strategies_size, 0);

    std::vector<double> priors_ses = computePrior(strategies, cluster, ses, last_choice);
        
    //E-step: 1.Calculate likelihoods using estimated rewards
    //E-step: 2. Update cluster posterior probabilities
    for (size_t i = 0; i < strategies.size(); ++i) 
    {
        std::shared_ptr<Strategy> strategyPtr = strategies[i];
        double crp_posterior = 0.0;
        std::string name = strategyPtr->getName();

        double log_likelihood = strategyPtr->getTrajectoryLikelihood(ratdata, ses); 
        if(logger)
        {
            std::cout << "estep_cluster_update for " << strategyPtr->getName()  << ", ses=" << ses  << ", prior=" << priors_ses[i] << ", log_likelihood=" << log_likelihood << std::endl;  
        }

                
        crp_posterior = exp(log_likelihood) * priors_ses[i];
        
        if(crp_posterior > max_posterior)
        {
            max_posterior = crp_posterior;
            most_lik_strategies.clear();
            most_lik_strategies.push_back(name);
        }
        else if(crp_posterior == max_posterior)
        {
            most_lik_strategies.push_back(name);
        } 
       //std::cout << "estep_cluster_update for " << strategyPtr->getName()  << ", ses=" << ses  << ", posterior=" << crp_posterior << std::endl; 

        posteriors[i] = crp_posterior;

        if (std::isnan(exp(log_likelihood))) {
                    
            std::cout << "exp(log_likelihood) is nan. Check" << std::endl;
            std::exit(EXIT_FAILURE);
        }else if(crp_posterior == 0){
            //std::cout << "Zero posterior prob. Check" << std::endl;
        }
        double marginalLik = log_likelihood * priors_ses[i];
        strategyPtr->setMarginalLikelihood(marginalLik);
    }

    double sum = std::accumulate(posteriors.begin(), posteriors.end(), 0.0);
    if(sum==0)
    {
        sum = 0.1; // To prevent NaN error
    }
    std::transform(posteriors.begin(), posteriors.end(), posteriors.begin(), [sum](double x) { return x / sum; });
    //std::cout << "estep_cluster_update " << ", ses=" << ses  << ", sumPosterior=" << sumPosterior << std::endl; 

    for(size_t i = 0; i < strategies.size(); ++i)
    {
        std::shared_ptr<Strategy> strategyPtr = strategies[i];

        if (std::isnan(posteriors[i])) {
                    
            std::cout << "Crp posteriors is nan. Check" << std::endl;
            std::exit(EXIT_FAILURE);
        }

        strategyPtr->setCrpPosterior(posteriors[i],ses);
        
        if(logger)
        {
            std::cout << "estep_cluster_update for " << strategyPtr->getName()  << ", ses=" << ses  << ", posterior=" << posteriors[i]  << std::endl;   
        }

    }

    if(logger)
    {
        std::cout << "Most likely strategies: " ;
        for (const std::string& name : most_lik_strategies) {
            std::cout << name << " ";
        }
        std::cout << std::endl;

    }

    if(most_lik_strategies.size() == 1)
    {
        last_choice = most_lik_strategies[0];
        // If most_lik_strategy is not in cluster, expand cluster
        if (std::find(cluster.begin(), cluster.end(), most_lik_strategies[0]) == cluster.end()) {
            // most_lik_strategy is not present, include it
            cluster.push_back(most_lik_strategies[0]);
            // std::cout <<"ses=" << ses << ", Adding " << most_lik_strategy <<" to cluster." << std::endl; 
            if(logger)
            {
                std::cout <<"ses=" << ses << ", Adding " << most_lik_strategies[0] <<" to cluster." << std::endl;   
            }
        }  

    }else
    {
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
            strategyPtr->setCrpPosterior(0,ses);
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
    

    return;
}

void mstep(const RatData& ratdata, int ses, std::vector<std::shared_ptr<Strategy>> strategies, std::vector<std::string>& cluster, bool logger=false)
{
    for (const auto& strategyPtr : strategies) {

        //std::cout << "m-step for " << strategyPtr->getName()  << ", ses=" << ses << std::endl; 
        //M-step
        //std::pair<std::vector<double>, std::vector<double>> rewardUpdates = getRewardUpdates(ratdata, ses);
        
        std::string strategy = strategyPtr->getName();
        if (std::find(cluster.begin(), cluster.end(), strategy) != cluster.end()) {
        
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
  
        
        }        
    }

    return;
}

// void compute_marginal_likelihoods(const RatData& ratdata, int ses, std::vector<std::shared_ptr<Strategy>>  strategies)
// {
    
//     //E-step: 1.Calculate likelihoods using estimated rewards
//     //E-step: 2. Update cluster posterior probabilities
//     double sumPosterior = 0.0;
//     for (const auto& strategyPtr : strategies) {
//         double crp_posterior = std::numeric_limits<double>::lowest();
//         std::string name = strategyPtr->getName();
       

//         double log_likelihood = strategyPtr->getTrajectoryLikelihood(ratdata, ses); 
        
//         //std::vector<double> crpPriors = strategyPtr->getCrpPrior();
//         double crp_prior = strategyPtr->getCrpPosterior(ses);

//         //std::cout << "compute_marginal_likelihoods: " << ", ses=" << ses << ", strategy=" << name << ", prior=" << crp_prior << ", log_likelihood=" << log_likelihood << std::endl; 

//         crp_posterior = log_likelihood * crp_prior; // weighted likelihoods for cross-validation
//         //crp_posterior.push_back(crp_posterior); 
//         strategyPtr->updateWeightedLikelihood(crp_posterior);
//     }  

//     return;
// }



void initRewardVals(const RatData& ratdata, int ses, std::vector<std::shared_ptr<Strategy>> strategies)
{
    for (const auto& strategyPtr : strategies) {

       //std::cout << "initRewardVals for " << strategyPtr->getName()  << ", ses=" << ses << std::endl; 
        //M-step
        //std::pair<std::vector<double>, std::vector<double>> rewardUpdates = getRewardUpdates(ratdata, ses);
        strategyPtr->initRewards(ratdata); 
    }
    
    return;
}

pagmo::vector_double PagmoProb::fitness(const pagmo::vector_double& v) const
{
    //Call E-step & M-step for 15 sessions
    double alpha_aca = v[0];
    double gamma_aca = v[1];
    
    double alpha_drl = v[2];
    double beta_drl = 1e-7;
    double lambda_drl = v[3];

    double alpha_arl = v[4];
    double beta_arl = 1e-7;
    double lambda_arl = v[5];

    double crpAlpha = v[6];
    double phi = 0.5;
    double eta = 4;

    // double crpAlpha = v[2];
    // double phi = v[3];

    // MazeGraph suboptimalHybrid3(Suboptimal_Hybrid3, false);
    // MazeGraph optimalHybrid3(Optimal_Hybrid3, true);

    //std::shared_ptr<Strategy> aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca, gamma_aca, 0, crpAlpha, phi, eta, false);
    //std::shared_ptr<Strategy> aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca, gamma_aca, 0, crpAlpha, phi, eta, true);
    
    //std::shared_ptr<Strategy> drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl, beta_drl, lambda_drl, crpAlpha, phi, eta, false);
    std::shared_ptr<Strategy> drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl, beta_drl, lambda_drl, crpAlpha, phi, eta, true);

    //std::shared_ptr<Strategy> arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"arl", alpha_arl, beta_arl, lambda_arl, crpAlpha, phi, eta, false);
    //std::shared_ptr<Strategy> arl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"arl",alpha_arl, beta_arl, lambda_arl, crpAlpha, phi, eta, true);


    std::vector<std::shared_ptr<Strategy>> strategies;

    // strategies.push_back(aca2_Suboptimal_Hybrid3);
    // strategies.push_back(aca2_Optimal_Hybrid3);

    //strategies.push_back(drl_Suboptimal_Hybrid3);
    strategies.push_back(drl_Optimal_Hybrid3);

    // strategies.push_back(arl_Suboptimal_Hybrid3);
    // strategies.push_back(arl_Optimal_Hybrid3);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    
    std::vector<std::string> cluster;
    std::string last_choice;

    for(int ses=0; ses < sessions; ses++)
    {
        
        if(ses==0)
        {
            initRewardVals(ratdata, ses, strategies);
        }

        estep_cluster_update(ratdata, ses, strategies, cluster, last_choice);
        mstep(ratdata, ses, strategies, cluster);      

    }

    // for(int ses=0; ses < sessions; ses++)
    // {
    //     compute_marginal_likelihoods(ratdata, ses, strategies);
    // }

    double marginal_lik = 0;
    for (const auto& strategyPtr : strategies) {
        std::vector<double> strat_posteriors = strategyPtr->getMarginalLikelihood();
        marginal_lik = marginal_lik + std::accumulate(strat_posteriors.begin(), strat_posteriors.end(), 0.0);
    }
    marginal_lik = marginal_lik* (-1);

    if(marginal_lik == 0)
    {
        marginal_lik = 100000; // Penalize to prevent zero likelihoods
    }
    return{marginal_lik};

}

std::pair<pagmo::vector_double, pagmo::vector_double> PagmoProb::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;

    bounds.first={0,0,0,0,0,0,0,0,0};
    bounds.second={1,1,1,1,1,1,1e-5,1,5};

    // bounds.first={0,0,0,0};
    // bounds.second={1,1,1e-5,1};
    
    return(bounds);
  }


void optimizeRL_pagmo(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3) {

    std::cout << "Initializing problem class" <<std::endl;
    // Create a function to optimize
    PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
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

    for (const auto &isl : archi) {
        std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
        std::vector<double> dec_vec = isl.get_population().champion_x();
        for (auto const& i : dec_vec)
             std::cout << i << ", ";
        std::cout << "\n" ;
    }
    //sink();


    return;
}


void runEM(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3)
{
    //// rat_103
    std::vector<double> v = {0.11776, 0.163443, 0.0486187, 1e-7,0.475538, 0.272467, 1e-7 , 0.0639478, 1.9239e-06, 0.993274, 4.3431};
    
    ////rat_114
    //std::vector<double> v = {0.0334664, 0.351993, 0.00478871, 1.99929e-07, 0.687998, 0.380462, 9.68234e-07, 0.136651, 8.71086e-06, 0.292224, 3.95355};

    double alpha_aca = v[0];
    double gamma_aca = v[1];
    
    double alpha_drl = v[2];
    double beta_drl = v[3];
    double lambda_drl = v[4];

    double alpha_arl = v[5];
    double beta_arl = v[6];
    double lambda_arl = v[7];

    double crpAlpha = v[8];
    double phi = v[9];
    double eta = v[10];
    
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"aca2", alpha_aca, gamma_aca, 0, crpAlpha, phi, eta, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"aca2",alpha_aca, gamma_aca, 0, crpAlpha, phi, eta, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"drl", alpha_drl, beta_drl, lambda_drl, crpAlpha, phi, eta, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"drl",alpha_drl, beta_drl, lambda_drl, crpAlpha, phi, eta, true);

    auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"arl", alpha_arl, beta_arl, lambda_arl, crpAlpha, phi, eta, false);
    auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"arl",alpha_arl, beta_arl, lambda_arl, crpAlpha, phi, eta, true);


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

    for(int ses=0; ses < sessions; ses++)
    {
        
        if(ses==0)
        {
            initRewardVals(ratdata, ses, strategies);
        }

        estep_cluster_update(ratdata, ses, strategies, cluster, last_choice, true);
        mstep(ratdata, ses, strategies, cluster);

    }

    // for(int ses=15; ses < sessions; ses++)
    // {
    //     compute_marginal_likelihoods(ratdata, ses, 15,  strategies);
    // }

    // double marginal_lik = 0;
    // for (const auto& strategyPtr : strategies) {
    //     std::vector<double> strat_posteriors = strategyPtr->getPosteriors();
    //     marginal_lik = marginal_lik + std::accumulate(strat_posteriors.begin(), strat_posteriors.end(), 0.0);
    // }
    // marginal_lik = marginal_lik* (-1);
    // std::cout << "marginal_lik=" << marginal_lik << std::endl;

    // arma::mat& aca2_suboptimal_probs =  aca2_Suboptimal_Hybrid3->getPathProbMat();
    // arma::mat& aca2_optimal_probs =  aca2_Optimal_Hybrid3->getPathProbMat();
    // arma::mat& drl_suboptimal_probs =  drl_Suboptimal_Hybrid3->getPathProbMat();
    // arma::mat& drl_optimal_probs =  drl_Optimal_Hybrid3->getPathProbMat();
    // arma::mat& arl_suboptimal_probs =  arl_Suboptimal_Hybrid3->getPathProbMat();
    // arma::mat& arl_optimal_probs =  arl_Optimal_Hybrid3->getPathProbMat();

    // aca2_suboptimal_probs.save("aca2_suboptimal_probs.csv", arma::csv_ascii);
    // aca2_optimal_probs.save("aca2_optimal_probs.csv", arma::csv_ascii);
    // drl_suboptimal_probs.save("drl_suboptimal_probs.csv", arma::csv_ascii);
    // drl_optimal_probs.save("drl_optimal_probs.csv", arma::csv_ascii);
    // arl_suboptimal_probs.save("arl_suboptimal_probs.csv", arma::csv_ascii);
    // arl_optimal_probs.save("arl_optimal_probs.csv", arma::csv_ascii);
}




int main() 
{
    std::cout <<"Inside main" <<std::endl;
    // Replace with the path to your Rdata file and the S4 object name
    // std::string rdataFilePath = "/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/rat114.Rdata";
    // std::string s4ObjectName = "ratdata";
    RInside R;
        
    std::string cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/InverseRL/rat103.Rdata')";
    R.parseEvalQ(cmd);                  
    Rcpp::S4 ratdata = R.parseEval("get('ratdata')");

    cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/InverseRL/Hybrid3.Rdata')";
    R.parseEvalQ(cmd);                  
    Rcpp::S4 Optimal_Hybrid3 = R.parseEval("get('Hybrid3')"); 


    cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/InverseRL/SubOptimalHybrid3.Rdata')";
    R.parseEvalQ(cmd);                  
    Rcpp::S4 Suboptimal_Hybrid3 = R.parseEval("get('SubOptimalHybrid3')"); 

    RatData rdata(ratdata);
    MazeGraph suboptimalHybrid3(Suboptimal_Hybrid3, false);
    MazeGraph optimalHybrid3(Optimal_Hybrid3, true);
 
    runEM(rdata, suboptimalHybrid3, optimalHybrid3);
    

    //optimizeRL_pagmo(rdata, suboptimalHybrid3, optimalHybrid3);
   

}
