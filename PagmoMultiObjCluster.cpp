#include "PagmoMultiObjCluster.h"
#include "RecordResults.h"
#include "InferStrategy.h"

pagmo::vector_double PagmoMultiObjCluster::fitness(const pagmo::vector_double& v) const
{
    double alpha_aca_subOptimal = v[4];
    double gamma_aca_subOptimal = v[5];

    double alpha_aca_optimal = v[4];
    double gamma_aca_optimal = v[5];

    //ARL params
    // double alpha_arl_subOptimal = params.find(std::make_pair("arl", false))->second[0];
    // double beta_arl_subOptimal = 1e-7;
    // double lambda_arl_subOptimal = params.find(std::make_pair("arl", false))->second[1];
    
    // double alpha_arl_optimal = params.find(std::make_pair("arl", true))->second[0];
    // double beta_arl_optimal = 1e-7;
    // double lambda_arl_optimal = params.find(std::make_pair("arl", true))->second[1];
 
    //DRL params
    double alpha_drl_subOptimal = v[6];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = v[7];
    
    double alpha_drl_optimal = v[6];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = v[7];

    
    int n1 = static_cast<int>(std::floor(v[0]));
    int n2 = static_cast<int>(std::floor(v[1]));
    int n3 = static_cast<int>(std::floor(v[2]));
    int n4 = static_cast<int>(std::floor(v[3]));
       
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, 0, 0, 0, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, 0, 0, 0, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, 0, 0, 0, true);

    
    
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

    double ll1 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < n1)
        {
           ll_ses  = aca2_Suboptimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
        }else{
           ll_ses  = aca2_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
        }
        
        ll_ses = ll_ses*(-1);
        ll1 = ll1 + ll_ses;
    }

    double ll2 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < n2)
        {
           ll_ses  = aca2_Suboptimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
        }else{
           ll_ses  = drl_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
        }
        
        ll_ses = ll_ses*(-1);
        ll2 = ll2 + ll_ses;
    }

    double ll3 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < n3)
        {
           ll_ses  = drl_Suboptimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
        }else{
           ll_ses  = aca2_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
        }
        
        ll_ses = ll_ses*(-1);
        ll3 = ll3 + ll_ses;
    }


    double ll4 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < n4)
        {
           ll_ses  = drl_Suboptimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
        }else{
           ll_ses  = drl_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
        }
        
        ll_ses = ll_ses*(-1);
        ll4 = ll4 + ll_ses;
    }


    double ll5 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses  = aca2_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
        
        ll_ses = ll_ses*(-1);
        ll5 = ll5 + ll_ses;
    }

    double ll6 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses  = drl_Optimal_Hybrid3->getTrajectoryLikelihood(ratdata, ses); 
        
        ll_ses = ll_ses*(-1);
        ll6 = ll6 + ll_ses;
    }

    //double min = std::min({ll1, ll2, ll3, ll4, ll5, ll6});


    return{ll1,ll2,ll3,ll4,ll5,ll6};

}

std::pair<pagmo::vector_double, pagmo::vector_double> PagmoMultiObjCluster::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;

    bounds.first={1,1,1,1,0,0,0,0};
    bounds.second={10,10,10,10,1,1,1,1};

    return(bounds);
  }

vector_double::size_type PagmoMultiObjCluster::get_nobj() const {
    return 6u;
}
