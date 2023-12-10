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
  PagmoMle(const RatData& ratdata_, const MazeGraph& mazeGraph_,  
  std::string learningRule_, bool optimal_):
  ratdata(ratdata_),  mazeGraph(mazeGraph_), learningRule(learningRule_), optimal(optimal_) {}


  // Destructor
  ~PagmoMle() {}

  // Fitness function
  vector_double fitness(const vector_double& v) const;

  // Bounds function
  std::pair<vector_double, vector_double> get_bounds() const;


private:
  // Members
  const RatData& ratdata;
  const MazeGraph& mazeGraph;
  const std::string learningRule;
  const bool optimal;

  
};



//void optimizeRL_pagmo(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3);

#endif