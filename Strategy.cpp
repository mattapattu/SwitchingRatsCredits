
#include "Strategy.h"
#include "InverseRL.h"
#include "matplotlibcpp.h"



// void Strategy::initRewards(const RatData& ratdata)
// {
//     initializeRewards(ratdata, 0, *this);
//     return;

// }   

// void Strategy::updateRewards(const RatData& ratdata, int session)
// {
//     updateRewardFunction(ratdata, session, *this);
//     return;

// }   

double Strategy::getTrajectoryLikelihood(const RatData& ratdata, int session)
{
    double lik = computeTrajectoryLik(ratdata, session, *this);
    return lik;
} 

void Strategy::updatePathProbMat(int ses)
{

    arma::rowvec probRow(15);
    probRow.fill(-1);
    probRow(12) = pathProbMat.n_rows;
    probRow(13) = ses;
    if(getOptimal())
    {
        probRow(14) = 1;
    }else{
        probRow(14) = 0;
    }

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


void Strategy::plotPathProbs()
{

    namespace plt = matplotlibcpp;

    arma::uvec selectedColumnsIndices = {0, 3, 4, 6, 9, 10};
    arma::mat selectedColumns = pathProbMat.cols(selectedColumnsIndices);

    // Extract column 13 for the x-axis
    arma::vec xAxis = pathProbMat.col(12);

    // Convert Armadillo matrices to std::vector for plotting
    std::vector<double> xData = arma::conv_to<std::vector<double>>::from(xAxis);

    std::vector<std::string> columnLabels = {"Path1, S0", "Path4, S0", "Path5, S0", "Path1, S1", "Path4, S1", "Path5, S1"};

    // Plot each selected column against the x-axis
    for (size_t i = 0; i < selectedColumns.n_cols; ++i)
    {
        std::vector<double> yData = arma::conv_to<std::vector<double>>::from(selectedColumns.col(i));

        // Plot the data
        plt::plot(xData, yData);
    }

    // Add labels and legend
    plt::xlabel("Column 13");
    plt::ylabel("Selected Columns");
    plt::legend();

    // Show the plot
    plt::show();

    // std::vector<double> y = { 1, 3, 2, 4 };
    // plt::plot(y);

    // plt::show();


    return;
}


