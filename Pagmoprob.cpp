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
   double Q = M_step2(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, resTuple, v, pool);

   Q = (-1)*Q;

  
   return{Q};

}

std::pair<pagmo::vector_double, pagmo::vector_double> PagmoProb::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;

    bounds.first={0,0,0,0};
    bounds.second={1,1,1,1};

    return(bounds);
  }
