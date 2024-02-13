#include "PagmoMultiObjCluster.h"
#include "RecordResults.h"
#include "InferStrategy.h"

pagmo::vector_double PagmoMultiObjCluster::fitness(const pagmo::vector_double& v) const
{
    //m1 params
    double alpha_aca_subOptimal_m1 = v[4];
    double gamma_aca_subOptimal_m1 = v[5];
    double alpha_aca_optimal_m1 = v[4];
    double gamma_aca_optimal_m1 = v[5];
 
    //m2 params
    double alpha_drl_subOptimal_m2 = v[6];
    double beta_drl_subOptimal_m2 = v[7];
    double lambda_drl_subOptimal_m2 = v[8];

    double alpha_aca_optimal_m2 = v[9];
    double gamma_aca_optimal_m2 = v[10];

    //m3 params
    
    double alpha_aca_subOptimal_m3 = v[11];
    double gamma_aca_subOptimal_m3 = v[12];

    double alpha_drl_optimal_m3 = v[13];
    double beta_drl_optimal_m3 = v[14];
    double lambda_drl_optimal_m3 = v[15];


    //m4

    double alpha_drl_subOptimal_m4 = v[16];
    double beta_drl_subOptimal_m4 = v[17];
    double lambda_drl_subOptimal_m4 = v[18];

    double alpha_drl_optimal_m4 = v[16];
    double beta_drl_optimal_m4 = v[17];
    double lambda_drl_optimal_m4 = v[18];

    //m5

    double alpha_aca_optimal_m5 = v[19];
    double gamma_aca_optimal_m5 = v[20];

    //m6

    double alpha_drl_optimal_m6 = v[21];
    double beta_drl_optimal_m6 = v[22];
    double lambda_drl_optimal_m6 = v[23];


    
    int n1 = static_cast<int>(std::floor(v[0]));
    int n2 = static_cast<int>(std::floor(v[1]));
    int n3 = static_cast<int>(std::floor(v[2]));
    int n4 = static_cast<int>(std::floor(v[3]));
       
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3_m1 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal_m1, gamma_aca_subOptimal_m1, 0, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3_m1 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal_m1, gamma_aca_optimal_m1, 0, 0, 0, 0, true);

    auto drl_Suboptimal_Hybrid3_m2 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"drl", alpha_drl_subOptimal_m2, beta_drl_subOptimal_m2, lambda_drl_subOptimal_m2, 0, 0, 0, false);
    auto aca2_Optimal_Hybrid3_m2 = std::make_shared<Strategy>(Optimal_Hybrid3,"aca2",alpha_aca_optimal_m2, gamma_aca_optimal_m2, 0, 0, 0, 0, true);

    auto aca2_Suboptimal_Hybrid3_m3 = std::make_shared<Strategy>(Suboptimal_Hybrid3,"aca2", alpha_aca_subOptimal_m3, gamma_aca_subOptimal_m3, 0, 0, 0, 0, false);
    auto drl_Optimal_Hybrid3_m3 = std::make_shared<Strategy>(Optimal_Hybrid3,"drl",alpha_drl_optimal_m3, beta_drl_optimal_m3, lambda_drl_optimal_m3, 0, 0, 0, true);

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
    double bic_score1 = 3*log(allpaths.n_rows)+ 2*ll1;

    double ll2 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < n2)
        {
           ll_ses  = drl_Suboptimal_Hybrid3_m2->getTrajectoryLikelihood(ratdata, ses); 
        }else{
           ll_ses  = aca2_Optimal_Hybrid3_m2->getTrajectoryLikelihood(ratdata, ses); 
        }
        
        ll_ses = ll_ses*(-1);
        ll2 = ll2 + ll_ses;
    }
    double bic_score2 = 6*log(allpaths.n_rows)+ 2*ll2;

    double ll3 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses = 0;
        if(ses < n3)
        {
           ll_ses  = aca2_Suboptimal_Hybrid3_m3->getTrajectoryLikelihood(ratdata, ses); 
        }else{
           ll_ses  = drl_Optimal_Hybrid3_m3->getTrajectoryLikelihood(ratdata, ses); 
        }
        
        ll_ses = ll_ses*(-1);
        ll3 = ll3 + ll_ses;
    }
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
    double bic_score4 = 4*log(allpaths.n_rows)+ 2*ll4;

    double ll5 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses  = aca2_Optimal_Hybrid3_m5->getTrajectoryLikelihood(ratdata, ses); 
        
        ll_ses = ll_ses*(-1);
        ll5 = ll5 + ll_ses;
    }
    double bic_score5 = 2*log(allpaths.n_rows)+ 2*ll5;

    double ll6 = 0;
    for(int ses=0; ses < sessions; ses++)
    {
        double ll_ses  = drl_Optimal_Hybrid3_m6->getTrajectoryLikelihood(ratdata, ses); 
        
        ll_ses = ll_ses*(-1);
        ll6 = ll6 + ll_ses;
    }
    double bic_score6 = 3*log(allpaths.n_rows)+ 2*ll6;

    //double min = std::min({ll1, ll2, ll3, ll4, ll5, ll6});


    return{ll1,ll2,ll3,ll4,ll5,ll6};

}

std::pair<pagmo::vector_double, pagmo::vector_double> PagmoMultiObjCluster::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;

    bounds.first={1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
    bounds.second={10,10,10,10,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};


    return(bounds);
  }

vector_double::size_type PagmoMultiObjCluster::get_nobj() const {
    return 6u;
}

