#include "PagmoMle.h"


vector_double PagmoMle::fitness(const vector_double& v) const
{
    double alpha = v[0];
    double gamma = 0;
    double lambda = 0;
    double crpAlpha = 0;
    double phi = 0;
    double eta = 0;
    
    Strategy strategy(mazeGraph,learningRule,alpha, gamma, lambda, crpAlpha, phi, eta, optimal);
    
    if(strategy.getName() == "aca2_Suboptimal_Hybrid3" || strategy.getName() == "aca2_Optimal_Hybrid3")
    {
        gamma = v[1];
        lambda = 0;

    }else if(strategy.getName() == "arl_Suboptimal_Hybrid3" || strategy.getName() == "arl_Optimal_Hybrid3")
    {
        gamma = v[1];
        lambda = 0;
    }else if (strategy.getName() == "drl_Suboptimal_Hybrid3" || strategy.getName() == "drl_Optimal_Hybrid3")
    {
        gamma = 1e-4;
        lambda = v[1];
    }
    

    std::vector<double> s0rewards;
    std::vector<double> s1rewards;
    if(optimal)
    {
        s0rewards = {0,0,0,0,0,0,0,5,0};
        s1rewards = {0,0,0,0,0,0,0,0,5};

        strategy.setRewardsS0(s0rewards);
        strategy.setRewardsS1(s1rewards);

    }else{
        s0rewards = {0,0,0,0,0,0,5,5,0,0,0,0};
        strategy.setRewardsS0(s0rewards);
    }
    
    strategy.setAlpha(alpha);
    strategy.setGamma(gamma);
    strategy.setLambda(lambda);

    strategy.resetCredits();

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    double loglikelihood = 0;

    if(!optimal)
    {
        sessions = 12; // For suboptimal strategies, use only first 11 sessions to learn the parameters to avoid underiftting
    }

    for(int ses=0; ses < sessions; ses++)
    {
        
        double log_likelihood_ses = strategy.getTrajectoryLikelihood(ratdata, ses); 
        loglikelihood = loglikelihood + log_likelihood_ses;
        //std::cout << "strategy=" << strategy.getName() << ", alpha=" <<strategy.getAlpha() << ", gamma=" << strategy.getGamma() << ", lambda=" << strategy.getLambda() << ", ses=" << ses << ", loglikelihood=" << loglikelihood << std::endl;
    }

    loglikelihood = loglikelihood * (-1);
 
    // if(strategy.getName() == "arl_Optimal_Hybrid3")
    // {
    //     std::cout << "strategy=" << strategy.getName() << ", alpha=" <<strategy.getAlpha() << ", gamma=" << strategy.getGamma() << ", lambda=" << strategy.getLambda() << ", loglikelihood=" << loglikelihood << std::endl;
    // }
    
    return{loglikelihood};

}

std::pair<vector_double, vector_double> PagmoMle::get_bounds() const 
{
    std::pair<vector_double, vector_double> bounds;

    if(learningRule == "arl")
    {
        bounds.first={0,1e-8};
        bounds.second={1,1e-6};
    }else
    {
        bounds.first={0,0};
        bounds.second={1,1};
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