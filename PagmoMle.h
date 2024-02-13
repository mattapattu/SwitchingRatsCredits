#ifndef PAGMOMLE_H
#define PAGMOMLE_H


#include <cstdlib>
#include <pagmo/problems/schwefel.hpp>
#include "Strategy.h"



using namespace Rcpp;
using namespace pagmo;

class PagmoMle {
public:
  PagmoMle();
  // Constructor
  PagmoMle(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_,  
  const MazeGraph& Optimal_Hybrid3_, std::string model_):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_), model(model_) {}


  // Destructor
  ~PagmoMle() {}

  // Fitness function
  vector_double fitness(const vector_double& v) const;

  // Bounds function
  std::pair<vector_double, vector_double> get_bounds() const;


private:
  // Members
  const RatData& ratdata;
  const MazeGraph& Suboptimal_Hybrid3;
  const MazeGraph& Optimal_Hybrid3;
  const std::string model;

  
};



//void optimizeRL_pagmo(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3);

#endif