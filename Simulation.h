#ifndef SIMULATION_H
#define SIMULATION_H

#include <RInside.h>
#include "InverseRL.h"
#include "ParticleFilter.h"


RatData generateSimulation(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::map<std::string, std::vector<double>> clusterParams,RInside &R,  int selectStrat);
std::map<std::pair<std::string, bool>, std::vector<double>> findParamsWithSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3);
std::vector<double> findClusterParamsWithSimData(RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params);
void runEMOnSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::vector<double> v, bool debug, std::string run);
void testRecovery(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, RInside &R, std::string run, BS::thread_pool& pool);
void updateConfusionMatrix(std::vector<RecordResults> allResults, std::string run);
void testSimulation(RatData& simRatData, Strategy& trueStrategy, RInside &R);
RatData generateSimulation(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::vector<double> v, RInside &R, int selectStrat, std::string run, BS::thread_pool& pool);

#endif