#ifndef INVERSERL_H
#define INVERSERL_H

#include "Strategy.h"

double computeTrajectoryLik(const RatData& ratdata, int session, Strategy& strategy);
void updateRewardFunction(const RatData& ratdata, int session, Strategy& strategy);
void initializeRewards(const RatData& ratdata, int session, Strategy& strategy);

void acaCreditUpdate(std::vector<std::string> episodeTurns, std::vector<int> episodeTurnStates, std::vector<double> episodeTurnTimes, double score_episode, Strategy& strategy);
double getAca2SessionLikelihood(const RatData& ratdata, int session, Strategy& strategy);
double getDiscountedRwdQlearningLik(const RatData& ratdata, int session, Strategy& strategy);
double getAvgRwdQLearningLik(const RatData& ratdata, int session, Strategy& strategy);
//void updateAca2Rewards(const Rcpp::S4& ratdata, int session, Strategy& strategy, bool sim=false);
//void initializeAca2Rewards(const Rcpp::S4& ratdata, int session, Strategy& strategy, bool sim=false);
void printFirst5Rows(const arma::mat& matrix, std::string matname);
std::vector<std::string> generatePathTrajectory(Strategy& strategy, BoostGraph* graph, BoostGraph::Vertex rootNode);
int getNextState(int curr_state, int action);
double simulateTurnDuration(arma::mat hybridTurnTimes, int hybridTurnId, int state, int session, Strategy& strategy);
std::pair<arma::mat, arma::mat> simulateAca2(const RatData& ratdata, int session, Strategy& strategy);
std::pair<arma::mat, arma::mat> simulateAvgRwdQLearning(const RatData& ratdata, int session, Strategy& strategy);
std::pair<arma::mat, arma::mat> simulateDiscountedRwdQlearning(const RatData& ratdata, int session, Strategy& strategy);
std::pair<arma::mat, arma::mat> simulateTrajectory(const RatData& ratdata, int session, Strategy& strategy);
std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>> getRewardFunctions(const RatData& ratdata, Strategy& strategy, double phi=0.1);


#endif
