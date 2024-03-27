#include "Pagmoprob.h"
#include "RecordResults.h"
#include "Strategy.h"
#include "InferStrategy.h"
#include <algorithm>

// pagmo::vector_double PagmoProb::fitness(const pagmo::vector_double& v) const
// {
//    double Q = M_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, resTuple, v, pool);

//    Q = (-1)*Q;


//    return{Q};

// }

pagmo::vector_double PagmoProb::fitness(const pagmo::vector_double& v) const
{
  //  double Q = M_step2(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, resTuple, v, pool);
                                                                  
  double Q_prev = 0;
  if(k>1)
  {
    Q_prev =  M_step4(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, prevSmoothedTrajectories, prevFilteredWeights, v, pool);
  }

  double Q_k_ = M_step4(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, smoothedTrajectories, filteredWeights, v, pool);

  double gamma_k = (double) gamma/(double) k;
  double Q_k = (1-gamma_k)*Q_prev + gamma_k*Q_k_;

  Q_k = (-1)*Q_k;

  
   return{Q_k};

}

std::pair<pagmo::vector_double, pagmo::vector_double> PagmoProb::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;

    bounds.first={1e-6,1e-6,1e-6,1e-6};
    bounds.second={1,1,1,1};

    return(bounds);
  }
