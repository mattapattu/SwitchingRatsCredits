#ifndef PAGMOPROB_H
#define PAGMOPROB_H


#include <cstdlib>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problems/schwefel.hpp>
#include "Strategy.h"




using namespace Rcpp;
using namespace pagmo;

class PagmoProb {
public:
  PagmoProb();
  // Constructor
  PagmoProb(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_,  
  const MazeGraph& Optimal_Hybrid3_, const std::map<std::pair<std::string, bool>, std::vector<double>>& params_ ):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_), params(params_) {}

  PagmoProb(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_,  
  const MazeGraph& Optimal_Hybrid3_):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_) {}


  // Destructor
  ~PagmoProb() {}

  // Fitness function
  vector_double fitness(const vector_double& v) const;

  // Bounds function
  std::pair<vector_double, vector_double> get_bounds() const;


private:
  // Members
  const RatData& ratdata;
  const MazeGraph& Suboptimal_Hybrid3;
  const MazeGraph& Optimal_Hybrid3;
  const std::map<std::pair<std::string, bool>, std::vector<double>> params;

  
};



void optimizeRL_pagmo(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3);

#endif