
#include "Strategy.h"
#include "InverseRL.h"



void Strategy::initRewards(const RatData& ratdata)
{
    initializeRewards(ratdata, 0, *this);
    return;

}   

void Strategy::updateRewards(const RatData& ratdata, int session)
{
    updateRewardFunction(ratdata, session, *this);
    return;

}   

double Strategy::getTrajectoryLikelihood(const RatData& ratdata, int session)
{
    double lik = computeTrajectoryLik(ratdata, session, *this);
    return lik;
} 

void Strategy::updatePathProbMat(int ses)
{

    arma::rowvec probRow(14);
    probRow.fill(-1);
    probRow(12) = pathProbMat.n_rows;
    probRow(13) = ses;

    for (int path = 0; path < 6; path++)
    {
        // Rcpp::Rcout << "path=" << path << ", state=" << S << std::endl;

        for (int S = 0; S < 2; S++)
        {
            std::vector<std::string> turnVec;

            bool optimal = getOptimal();
            BoostGraph *graph;

            if (S == 0 && optimal)
            {
                turnVec = stateS0.getTurnsFromPaths(path, S, optimal);
                turnVec.insert(turnVec.begin(), "E");
                graph = &stateS0;
            }
            else if (S == 1 && optimal)
            {
                turnVec = stateS1.getTurnsFromPaths(path, S, optimal);
                turnVec.insert(turnVec.begin(), "I");
                graph = &stateS1;
            }
            else if (S == 0 && !optimal)
            {
                turnVec = stateS0.getTurnsFromPaths(path, S, optimal);
                turnVec.insert(turnVec.begin(), "E");
                graph = &stateS0;
            }
            else if (S == 1 && !optimal)
            {
                turnVec = stateS0.getTurnsFromPaths(path, S, optimal);
                turnVec.insert(turnVec.begin(), "I");
                graph = &stateS0;
            }

            double pathProb = 1;

            for (int k = 0; k < (turnVec.size() - 1); k++)
            {
                std::string turn1 = turnVec[k];
                std::string turn2 = turnVec[k + 1];
                // Rcpp::Rcout << "turn1=" << turn1 << ", turn2=" << turn2 << std::endl;

                auto v1 = graph->findNode(turn1);
                auto v2 = graph->findNode(turn2);

                auto e = graph->findEdge(v1, v2);
                double probability = exp(graph->getEdgeProbability(e));

                //Rcpp::Rcout << "Edge src="<< turn1 << ", dest=" << turn2  << ", prob=" << probability << std::endl;
                pathProb = probability * pathProb;
            }

            int index = path + (6 * S);
            probRow[index] = pathProb;
        }
    }
    pathProbMat.insert_rows(pathProbMat.n_rows, probRow);
    return;
}



