#ifndef INFERSTRATEGY_H
#define INFERSTRATEGY_H

#include "InverseRL.h"
#include "RecordResults.h"
#include <optional>


std::vector<double> computePrior(std::vector<std::shared_ptr<Strategy>>  strategies, std::vector<std::string>& cluster, int ses, std::string& last_choice);
arma::mat estep_cluster_update(const RatData& ratdata, int ses, std::vector<std::shared_ptr<Strategy>>  strategies, std::vector<std::string>& cluster, std::string& last_choice ,bool logger, RecordResults& sessionResults);
void mstep(const RatData& ratdata, int ses, std::vector<std::shared_ptr<Strategy>> strategies, std::vector<std::string>& cluster, bool logger, RecordResults& sessionResults);
void initRewardVals(const RatData& ratdata, int ses, std::vector<std::shared_ptr<Strategy>> strategies, bool logger=false);
void findClusterParams(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params);
void findParams(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3);
void runEM(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params,  std::map<std::string, std::vector<double>> clusterParams, bool debug=false);
void testLogLik(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3);
void findMultiObjClusterParams(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, const std::map<std::pair<std::string, bool>, std::vector<double>>& params);
void runEM2(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::string, std::vector<double>> clusterParams, bool debug);



#endif