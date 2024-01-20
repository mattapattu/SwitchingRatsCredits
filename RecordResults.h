#ifndef RECORDRESULTS_H
#define RECORDRESULTS_H

#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <RcppArmadillo.h>


class RecordResults {
public:

    RecordResults(){}
    // Constructor for simulation results
    RecordResults(const std::string& selectedStrategy,
                  const arma::mat& probabilityMatrix,
                  const std::string& trueGeneratingStrategy,
                  const std::map<std::string, double>& posteriors,
                  const std::map<std::string, std::vector<double>>& rewardVectorS0,
                  const std::map<std::string, std::vector<double>>& rewardVectorS1)
        : selectedStrategy(selectedStrategy),
          probabilityMatrix(probabilityMatrix),
          trueGeneratingStrategy(trueGeneratingStrategy),
          posteriors(posteriors),
          rewardVectorS0(rewardVectorS0),
          rewardVectorS1(rewardVectorS1) {}

    // Constructor for non-simulation results
    RecordResults(const std::string& selectedStrategy,
                  const arma::mat& probabilityMatrix,
                  const std::map<std::string, double>& posteriors,
                  const std::map<std::string, std::vector<double>>& rewardVectorS0,
                  const std::map<std::string, std::vector<double>>& rewardVectorS1)
        : selectedStrategy(selectedStrategy),
          probabilityMatrix(probabilityMatrix),
          trueGeneratingStrategy("None"),  // Default to "None" for non-simulation results
          posteriors(posteriors),
          rewardVectorS0(rewardVectorS0),
          rewardVectorS1(rewardVectorS1) {}

    // Getter and Setter for selectedStrategy
    std::string getSelectedStrategy() const {
        return selectedStrategy;
    }

    void setSelectedStrategy(const std::string& strategy) {
        selectedStrategy = strategy;
    }

    // Getter and Setter for probabilityMatrix
    const arma::mat& getProbabilityMatrix() const {
        return probabilityMatrix;
    }

    void setProbabilityMatrix(const arma::mat& matrix) {
        probabilityMatrix = matrix;
    }

    // Getter and Setter for trueGeneratingStrategy
    std::string getTrueGeneratingStrategy() const {
        return trueGeneratingStrategy;
    }

    void setTrueGeneratingStrategy(const std::string& strategy) {
        trueGeneratingStrategy = strategy;
    }

    // Getter and Setter for posteriors
    const std::map<std::string, double>& getPosteriors() const {
        return posteriors;
    }

    void setPosteriors(const std::string& key, double value) {
        posteriors[key] = value;
    }


    // Getter and Setter for rewardVectorS0
    std::map<std::string, std::vector<double>>& getRewardVectorS0() {
        return rewardVectorS0;
    }

    void setRewardVectorS0(const std::string& key, const std::vector<double>& vector) {
        rewardVectorS0[key] = vector;
    }

    // Getter and Setter for rewardVectorS1
    std::map<std::string, std::vector<double>>& getRewardVectorS1() {
        return rewardVectorS1;
    }

    void setRewardVectorS1(const std::string& key, const std::vector<double>& vector) {
        rewardVectorS1[key] = vector;
    }

    // Getter and Setter for allResults
    const std::vector<RecordResults>& getAllResults() const {
        return allResults;
    }

    // Add results for a session
    void addResults(const RecordResults& sessionResults) {
        allResults.push_back(sessionResults);
    }

    void addCrpPriorSes(std::vector<double> v)
    {
        crpPriorSes = v;
    }


    std::vector<double> getCrpPriorSes()
    {
        return crpPriorSes;
    }



    // // Display results for a single session
    // void displayResults() const {
    //     std::cout << "Selected Strategy: " << getSelectedStrategy() << std::endl;
    //     std::cout << "Probability Matrix:" << std::endl;
    //     displayMatrix(getProbabilityMatrix());
    //     std::cout << "True Generating Strategy: " << getTrueGeneratingStrategy() << std::endl;
    //     std::cout << "Posteriors:" << std::endl;
    //     displayPosteriors();
    //     std::cout << "Reward Vector for S0:" << std::endl;
    //     displayVector(getRewardVectorS0());
    //     std::cout << "Reward Vector for S1:" << std::endl;
    //     displayVector(getRewardVectorS1());
    //     std::cout << "-----------------------------------" << std::endl;
    // }

private:
    std::string selectedStrategy;
    arma::mat probabilityMatrix;
    std::string trueGeneratingStrategy;
    std::map<std::string, double> posteriors;
    std::map<std::string, std::vector<double>> rewardVectorS0;
    std::map<std::string, std::vector<double>> rewardVectorS1;
    std::vector<double> crpPriorSes;

    // Container to store results for all sessions
    std::vector<RecordResults> allResults;
};

#endif
