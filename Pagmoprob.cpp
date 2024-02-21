#include "Pagmoprob.h"
#include "RecordResults.h"
#include "Strategy.h"
#include "InferStrategy.h"
#include <algorithm>
#include "ParticleFilter.h"

pagmo::vector_double PagmoProb::fitness(const pagmo::vector_double& v) const
{
   double Q = M_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, resTuple, v);
   Q = (-1)*Q;
   return{Q};

}

std::pair<pagmo::vector_double, pagmo::vector_double> PagmoProb::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;

    bounds.first={1e-8,1e-8,1e-8,1e-8,1e-8};
    bounds.second={1,1,1,1,5};

    return(bounds);
  }
