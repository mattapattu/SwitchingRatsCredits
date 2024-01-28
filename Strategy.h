#ifndef STRATEGY_H
#define STRATEGY_H

#include "BoostGraph.h"
#include "RatData.h"

using namespace Rcpp;

class Strategy
{
public:

    Strategy() {}

    // Constructor
    Strategy(const MazeGraph &testModel_, std::string learningRule_, double alpha_, double gamma_, double lambda_, double crpAlpha_, double phi_, double eta_, bool optimal_) : stateS0(testModel_, 0, optimal_), stateS1(testModel_, 1, optimal_)
    {

        alpha = alpha_;
        gamma = gamma_;
        lambda = lambda_;
        crpAlpha = crpAlpha_;
        learningRule = learningRule_;
        phi = phi_;
        eta = eta_;
        averageReward = 0;
        generatedTurnCounter = 0;

        std::string model = testModel_.getName();
        setName(model);

        // Rcpp::S4 s0graph = Rcpp::as<Rcpp::S4>(testModel_.slot("S0"));
        std::vector<std::string> s0nodes = testModel_.getNodeListS0();
        int s0nodes_size = s0nodes.size();
        std::vector<double> vecObj(s0nodes_size, 0);
        rewardsS0 = vecObj;

        optimal = optimal_;
        int s1nodes_size = 0;
        // stateS0 = BoostGraph(testModel_, 0);

        if (optimal_)
        {
            // stateS1 = BoostGraph(testModel_, 1);

            // Rcpp::S4 s1graph = Rcpp::as<Rcpp::S4>(turnModel.slot("S1"));
            std::vector<std::string> s1nodes = testModel_.getNodeListS1();
            s1nodes_size = s1nodes.size();

            std::vector<double> vecObj(s1nodes_size, 0);
            rewardsS1 = vecObj;
        }

        return;
    }

    void setName(std::string testModel)
    {
        if (learningRule == "aca2" && testModel == "SubOptimalHybrid3")
        {
            name = "aca2_Suboptimal_Hybrid3";
        }
        else if (learningRule == "aca2" && testModel == "Hybrid3")
        {
            name = "aca2_Optimal_Hybrid3";
        }
        else if (learningRule == "arl" && testModel == "SubOptimalHybrid3")
        {
            name = "arl_Suboptimal_Hybrid3";
        }
        else if (learningRule == "arl" && testModel == "Hybrid3")
        {
            name = "arl_Optimal_Hybrid3";
        }
        else if (learningRule == "drl" && testModel == "SubOptimalHybrid3")
        {
            name = "drl_Suboptimal_Hybrid3";
        }
        else if (learningRule == "drl" && testModel == "Hybrid3")
        {
            name = "drl_Optimal_Hybrid3";
        }

        // std::cout << "learningRule=" << learningRule << ", testModel=" << testModel << ", name=" << name << std::endl;
    }

    std::string getName()
    {
        return name;
    }

    std::string getLearningRule()
    {
        return learningRule;
    }

    double getCrpAlpha()
    {
        return crpAlpha;
    }

    double getAlpha()
    {
        return alpha;
    }

    void setAlpha(double alpha_)
    {
        alpha = alpha_;
    }


    double getGamma()
    {
        return gamma;
    }

    void setGamma(double gamma_)
    {
        gamma = gamma_;
    }

    double getLambda()
    {
        return lambda;
    }

    void setLambda(double lambda_)
    {
        lambda = lambda_;
    }

    double getAverageReward()
    {
        return averageReward;
    }

    void setAverageReward(double averageReward_)
    {
        averageReward = averageReward_;
    }

    double getGeneratedTurnCounter()
    {
        return generatedTurnCounter;
    }

    void incrementGeneratedTurnCounter()
    {
        generatedTurnCounter++;
    }


    double getPhi()
    {
        return phi;
    }

    void setPhi(double phi_)
    {
        phi = phi_;
    }

    bool getOptimal()
    {
        return optimal;
    }

    double getEta()
    {
        return eta;
    }

    void setRewardsS0(std::vector<double> rewards)
    {
        rewardsS0 = rewards;
    }

    void setRewardsS1(std::vector<double> rewards)
    {
        rewardsS1 = rewards;
    }

    std::vector<double> getRewardsS0()
    {
        return rewardsS0;
    }

    std::vector<double> getRewardsS1()
    {
        return rewardsS1;
    }

    void setMarginalLikelihood(double likelihood)
    {
        // std::cout << "Adding posterior=" << posterior << " to vector" << std::endl;
        marginalLikelihood.push_back(likelihood);
    }

    std::vector<double> getMarginalLikelihood()
    {
        // std::cout << "Returning posterior vector of size=" << posteriors.size() <<  std::endl;
        return marginalLikelihood;
    }

    BoostGraph& getStateS0()
    {
        return stateS0;
    }

    std::vector<double> getS0Credits()
    {
        std::vector<double> v = stateS0.getVertexCredits();
        return v;
    }

    void setStateS0Credits(std::vector<double> v)
    {
        stateS0.setVertexCredits(v);
        stateS0.updateEdgeProbabilitiesSoftmax();
    }

    BoostGraph& getStateS1()
    {
        return stateS1;
    }

    std::vector<double> getS1Credits()
    {
        std::vector<double> v = stateS1.getVertexCredits();
        return v;
    }

    void setStateS1Credits(std::vector<double> v)
    {
        stateS1.setVertexCredits(v);
        stateS1.updateEdgeProbabilitiesSoftmax();
    }

    


    void setCrpPosterior(double crp, int t)
    {
        if (crpPosteriors.size() < (t + 1))
        {
            // std::cout << "crpPrior.size()= " << crpPrior.size() << "adding element = " << t-1  << " to" << crp << std::endl;
            crpPosteriors.push_back(crp);
        }
        else
        {
            // std::cout << "crpPrior.size()= " << crpPrior.size() << "setting element = " << t-1  << " to" << crp << std::endl;
            crpPosteriors[t] = crp;
        }

        return;
    }

    double getCrpPosterior(int t)
    {
        // std::cout << "crpPrior.size()= " << crpPrior.size() << "getting element = " << t-1 << " = " << crpPrior[t-1] << std::endl;
        return crpPosteriors[t];
    }

    std::vector<double> getCrpPosteriors()
    {
        return crpPosteriors;
    }

    void setCrpPriorInEachTrial(double v)
    {
        crpPriorInEachTrial.push_back(v);
    }

    std::vector<double> getCrpPriorInEachTrial()
    {
        return crpPriorInEachTrial;
    }


    void initRewards(const RatData &ratdata);

    void updateRewards(const RatData &ratdata, int session);

    double getTrajectoryLikelihood(const RatData &ratdata, int session);

    void resetCredits()
    {
        stateS0.resetNodeCredits();
        stateS0.updateEdgeProbabilitiesSoftmax();
        if (optimal)
        {
            stateS1.resetNodeCredits();
            stateS1.updateEdgeProbabilitiesSoftmax();
        }
        return;
    }

    // void updateEdgeProbabilities()
    // {
    //     stateS0.updateEdgeProbabilitiesSoftmax();
    //     if (optimal)
    //     {
    //         stateS1.updateEdgeProbabilitiesSoftmax();
    //     }
    //     return;
    // }

    void resetRewards()
    {
        std::fill(rewardsS0.begin(), rewardsS0.end(), 0);
        if (optimal)
        {
            std::fill(rewardsS1.begin(), rewardsS1.end(), 0);
        }
        return;
    }

    
    void updatePathProbMat(int ses);

    void plotPathProbs();

    arma::mat& getPathProbMat()
    {
        return pathProbMat;
    }

    void resetPathProbMat()
    {
        pathProbMat.reset();
    }

    

private:
    BoostGraph stateS0;       // To store credits of actions from each session
    BoostGraph stateS1;       // To store credits of actions from each session
    std::string learningRule; // creditAssignment
    std::vector<double> crpPosteriors;
    std::vector<double> rewardsS0;
    std::vector<double> rewardsS1;
    std::vector<double> marginalLikelihood;
    std::vector<double> crpPriorInEachTrial;
    bool optimal;
    double alpha;
    double gamma;
    double lambda;
    double phi;
    double crpAlpha;
    double eta;
    double averageReward;
    double generatedTurnCounter;
    std::string name;
    arma::mat pathProbMat;
};

#endif
