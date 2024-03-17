#ifndef PAGMOPROB_H
#define PAGMOPROB_H


#include <cstdlib>
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problems/schwefel.hpp>
#include "Strategy.h"
#include "ParticleFilter.h"
#include <pagmo/utils/gradients_and_hessians.hpp>






using namespace Rcpp;
using namespace pagmo;

class PagmoProb {
public:
  PagmoProb();
  // Constructor


  // PagmoProb(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_,  
  // const MazeGraph& Optimal_Hybrid3_, int N_, std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> resTuple_, BS::thread_pool& pool_):
  // ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_), N(N_), resTuple(resTuple_), pool(pool_)  {}

  PagmoProb(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_,  
  const MazeGraph& Optimal_Hybrid3_, int N_, std::tuple<std::vector<std::vector<double>>, std::vector<ParticleFilter>, std::vector<std::vector<int>>> resTuple_, BS::thread_pool& pool_):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_), N(N_), resTuple(resTuple_), pool(pool_)  {}

  PagmoProb(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_,  
  const MazeGraph& Optimal_Hybrid3_, int M_, int k_, double gamma_, std::vector<std::vector<int>> smoothedTrajectories_, BS::thread_pool& pool_):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_), M(M_), k(k_), gamma(gamma_), smoothedTrajectories(smoothedTrajectories_), pool(pool_)  {}

  PagmoProb(const RatData& ratdata_, const MazeGraph& Suboptimal_Hybrid3_,  
  const MazeGraph& Optimal_Hybrid3_, int M_, int k_, double gamma_, std::vector<std::vector<int>> smoothedTrajectories_, std::vector<std::vector<int>> prevSmoothedTrajectories_, BS::thread_pool& pool_):
  ratdata(ratdata_),  Suboptimal_Hybrid3(Suboptimal_Hybrid3_), Optimal_Hybrid3(Optimal_Hybrid3_), M(M_), k(k_), gamma(gamma_), smoothedTrajectories(smoothedTrajectories_),prevSmoothedTrajectories(prevSmoothedTrajectories_), pool(pool_)  {}
              

  // Destructor
  ~PagmoProb() {}

  // Fitness function
  vector_double fitness(const vector_double& v) const;

  // Bounds function
  std::pair<vector_double, vector_double> get_bounds() const;

  
  vector_double gradient(const vector_double &dv) const
  {
    return estimate_gradient([this](const vector_double &x) {return this->fitness(x);},dv);
  }


  // vector_double::size_type get_nix() const
  // {
  //   return 4;
  // }


  // vector_double::size_type get_nec() const
  // {
  //   return 0;
  // }
  // vector_double::size_type get_nic() const
  // {
  //   return 0;
  // }


  // std::vector<std::atomic<std::pair<double, std::vector<double>>>> getIndexedValues()
  // {
  //   return indexedValues;
  // }

  // void addIndexedValues(std::pair<double, std::vector<double>> x) const
  // {
  //   //std::lock_guard<std::mutex> lock(vectorMutex);
  //   // Perform operations on threadSafeVector
  //   indexedValues.push_back(x);

  // }


private:
  // Members
  const RatData& ratdata;
  const MazeGraph& Suboptimal_Hybrid3;
  const MazeGraph& Optimal_Hybrid3;
  const std::map<std::pair<std::string, bool>, std::vector<double>> params;
  int N;

  int M;
  int k;
  double gamma;
  std::vector<std::vector<int>> smoothedTrajectories;
  std::vector<std::vector<int>> prevSmoothedTrajectories;
  //std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> resTuple;
  std::tuple<std::vector<std::vector<double>>, std::vector<ParticleFilter>, std::vector<std::vector<int>>> resTuple;
  BS::thread_pool& pool;
  //mutable std::vector<std::atomic<std::pair<double, std::vector<double>>>> indexedValues;
  // mutable std::mutex  vectorMutex;

  
};



void optimizeRL_pagmo(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3);

#endif