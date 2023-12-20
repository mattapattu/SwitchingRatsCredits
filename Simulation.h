#ifndef SIMULATION_H
#define SIMULATION_H

#include <RInside.h>
#include "InverseRL.h"

RatData generateSimulation(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::map<std::string, std::vector<double>> clusterParams,RInside &R);
std::map<std::pair<std::string, bool>, std::vector<double>> findParamsWithSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3);
std::vector<double> findClusterParamsWithSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>>& params);
void runEMOnSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::vector<double> clusterParams, bool debug=false);
void testRecovery(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, RInside &R);

#endif