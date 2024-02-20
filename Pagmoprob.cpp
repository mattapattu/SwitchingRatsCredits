#include "Pagmoprob.h"
#include "RecordResults.h"
#include "Strategy.h"
#include "InferStrategy.h"
#include <algorithm>
#include "ParticleFilter.h"

pagmo::vector_double PagmoProb::fitness(const pagmo::vector_double& v) const
{
   std::pair<std::vector<std::vector<double>>, double> q = particle_filter(1000, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, v );
   double lik = q.second;
   lik = lik *(-1);

   return{lik};

}

std::pair<pagmo::vector_double, pagmo::vector_double> PagmoProb::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;

    //bounds.first={1e-2,0.5,1e-2,0.5,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8};
    //bounds.first={1e-2,0.5,1e-2,0.8,1e-2,1e-2,1e-2,1e-2,0.1,1e-8,1e-8};
    bounds.first={0,0,0,0,0,0, 0, 0, 0, 0};
    bounds.second={1,1,1,1,1,1, 1, 1, 1, 1};
    // bounds.first={0,0,0,0,0,0,0,0,1e-6,0,0};
    //bounds.second={1,1,1,1,1,1,1,1,1e-3,1,10};

    return(bounds);
  }
