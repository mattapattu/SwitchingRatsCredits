#include "Pagmoprob.h"
#include "RecordResults.h"
#include "Strategy.h"
#include "InferStrategy.h"
#include <algorithm>

pagmo::vector_double PagmoProb::fitness(const pagmo::vector_double& v) const
{
   double Q = M_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, resTuple, v);
   Q = (-1)*Q;

  //  std::vector<ParticleFilter>  particleFilterVec;
  //   for (int i = 0; i < N; i++) {
  //       auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, v, i, 1.0);
  //       particleFilterVec.push_back(pf);
  //       //std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;

  //   }
  //   std::tuple<std::vector<std::vector<double>>, double, std::vector<std::vector<double>>> q = particle_filter_new(N,particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3);
  //   //std::cout << "loglikelihood=" << q.second << std::endl;
  //  double Q = (-1)*std::get<1>(q);
   return{Q};

}

std::pair<pagmo::vector_double, pagmo::vector_double> PagmoProb::get_bounds() const
  {
    std::pair<vector_double, vector_double> bounds;

    bounds.first={0.1,0.7,1e-8,1e-8,1e-8,1e-8,1e-8,1e-8};
    bounds.second={1,1,1,1,1,1,1,1};

    return(bounds);
  }
