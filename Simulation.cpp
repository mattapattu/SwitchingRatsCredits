#include "InferStrategy.h"
#include "Pagmoprob.h"
#include "PagmoMle.h"
#include "Simulation.h"
#include <pagmo/algorithm.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/archipelago.hpp>
#include <random>
#include <RInside.h>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>




RatData generateSimulation(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::map<std::string, std::vector<double>> clusterParams, RInside &R, int selectStrat)
{
        //ACA params
    double alpha_aca_subOptimal = params.find(std::make_pair("aca2", false))->second[0];
    double gamma_aca_subOptimal = params.find(std::make_pair("aca2", false))->second[1];

    double alpha_aca_optimal = params.find(std::make_pair("aca2", true))->second[0];
    double gamma_aca_optimal = params.find(std::make_pair("aca2", true))->second[1];

    //ARL params
    double alpha_arl_subOptimal = params.find(std::make_pair("arl", false))->second[0];
    double beta_arl_subOptimal = 1e-7;
    double lambda_arl_subOptimal = params.find(std::make_pair("arl", false))->second[1];
    
    double alpha_arl_optimal = params.find(std::make_pair("arl", true))->second[0];
    double beta_arl_optimal = 1e-7;
    double lambda_arl_optimal = params.find(std::make_pair("arl", true))->second[1];
 
    //DRL params
    double alpha_drl_subOptimal = params.find(std::make_pair("drl", false))->second[0];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = params.find(std::make_pair("drl", false))->second[1];
    
    double alpha_drl_optimal = params.find(std::make_pair("drl", true))->second[0];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = params.find(std::make_pair("drl", true))->second[1];

    std::string rat = ratdata.getRat();
    double crpAlpha = clusterParams.find(rat)->second[0];
    double phi = clusterParams.find(rat)->second[1];
    double eta = 0;

    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, crpAlpha, phi, eta, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, crpAlpha, phi, eta, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, crpAlpha, phi, eta, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, crpAlpha, phi, eta, true);

    auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"arl", alpha_arl_subOptimal, beta_arl_subOptimal, lambda_arl_subOptimal, crpAlpha, phi, eta, false);
    auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"arl",alpha_arl_optimal, beta_arl_optimal, lambda_arl_optimal, crpAlpha, phi, eta, true);

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;


    std::srand(static_cast<unsigned>(std::time(nullptr)));


    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> distribution(5, 10);
    int changepoint_ses = distribution(gen);
    //int changepoint_ses = 50;

    std::vector<std::shared_ptr<Strategy>> subOptimalStrategies = {aca2_Suboptimal_Hybrid3,drl_Suboptimal_Hybrid3,arl_Suboptimal_Hybrid3};
    std::vector<std::shared_ptr<Strategy>> optimalStrategies = {aca2_Optimal_Hybrid3,drl_Optimal_Hybrid3,arl_Optimal_Hybrid3};

    // std::uniform_int_distribution<std::size_t> distribution_strat(0, subOptimalStrategies.size() - 1);
    // // Generate a random index for vector1
    // std::size_t random_index1 = distribution_strat(gen);

    // // Generate a random index for vector2
    // std::size_t random_index2 = distribution_strat(gen);

    // Get the randomly selected elements
    // std::shared_ptr<Strategy> subOptimalStrategy = subOptimalStrategies[random_index1];
    // std::shared_ptr<Strategy> optimalStrategy = optimalStrategies[random_index2];
    // // std::shared_ptr<Strategy> subOptimalStrategy = aca2_Suboptimal_Hybrid3;
    // // std::shared_ptr<Strategy> optimalStrategy = aca2_Optimal_Hybrid3;

    std::vector<std::pair<std::shared_ptr<Strategy>, std::shared_ptr<Strategy>>> strategyPairVector;
    strategyPairVector.push_back(std::make_pair(drl_Optimal_Hybrid3, drl_Optimal_Hybrid3));
    strategyPairVector.push_back(std::make_pair(aca2_Optimal_Hybrid3, aca2_Optimal_Hybrid3));
    strategyPairVector.push_back(std::make_pair(arl_Optimal_Hybrid3, arl_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(drl_Suboptimal_Hybrid3, drl_Optimal_Hybrid3));
    strategyPairVector.push_back(std::make_pair(arl_Suboptimal_Hybrid3, arl_Optimal_Hybrid3));
    strategyPairVector.push_back(std::make_pair(aca2_Suboptimal_Hybrid3, aca2_Optimal_Hybrid3));


    strategyPairVector.push_back(std::make_pair(drl_Suboptimal_Hybrid3, arl_Optimal_Hybrid3));
    strategyPairVector.push_back(std::make_pair(drl_Suboptimal_Hybrid3, aca2_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(arl_Suboptimal_Hybrid3, aca2_Optimal_Hybrid3));
    strategyPairVector.push_back(std::make_pair(arl_Suboptimal_Hybrid3, drl_Optimal_Hybrid3));

    strategyPairVector.push_back(std::make_pair(aca2_Suboptimal_Hybrid3, drl_Optimal_Hybrid3));
    strategyPairVector.push_back(std::make_pair(aca2_Suboptimal_Hybrid3, arl_Optimal_Hybrid3));

    //strategyPairVector.push_back(std::make_pair(arl_Suboptimal_Hybrid3, arl_Optimal_Hybrid3));



    // std::random_device rd;
    // std::mt19937 gen(rd());
    //std::uniform_int_distribution<std::size_t> dist(0, strategyPairVector.size() - 1);
    
    // Selecting a random pair
    //std::size_t randomIndex = dist(gen);
    auto randomPair = strategyPairVector[selectStrat];


    arma::mat generated_PathData;
    arma::mat generated_TurnsData;
   
    std::cout << "Generating sim data with " << randomPair.first->getName() << " and "<< randomPair.second->getName()  << " changepoint at ses " <<  changepoint_ses << std::endl;
    
    // To store vector of true generatorStrateies
    std::vector<std::string> trueGenStrategies;

    for(int ses=0; ses < sessions; ses++)
    {
        //std::cout << "ses= " <<  ses << std::endl;

        std::shared_ptr<Strategy> strategy;
        if(ses < changepoint_ses)
        {
            strategy = randomPair.first;

        }else{

            strategy = randomPair.second;

        }

        trueGenStrategies.push_back(strategy->getName());

        std::pair<arma::mat, arma::mat> simData = simulateTrajectory(ratdata, ses, *strategy);
        arma::mat generated_PathData_sess = simData.first;
        arma::mat generated_TurnsData_sess = simData.second;

        generated_PathData = arma::join_cols(generated_PathData, generated_PathData_sess);
        generated_TurnsData = arma::join_cols(generated_TurnsData, generated_TurnsData_sess);
      
    }

    
    R["genData"] = Rcpp::wrap(generated_PathData);

    // Save the matrix as RData using RInside
    std::string filename = "generatedData_" + std::to_string(selectStrat) + "_" + rat +".RData";
    
    std::string rCode = "saveRDS(genData, file='" + filename + "')";
    R.parseEvalQ(rCode.c_str());
    
    RatData simRatdata(generated_PathData,generated_TurnsData,rat, true, trueGenStrategies);

    //testSimulation(simRatdata,*randomPair.first, R);
    return simRatdata;

}


//Ignore- cannot run testSimulation as getTrajectoryLikelihood requires setting rewardsS0 & rewardsS1, which is not "possible" for suboptimal case
void testSimulation(RatData& simRatData, Strategy& trueStrategy, RInside &R)
{
    arma::mat allpaths = simRatData.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    arma::mat probMatTrueStrat = trueStrategy.getPathProbMat();

    std::cout << "Last 10 rows of matrix probMatTrueStrat:\n" << probMatTrueStrat.tail_rows(10) << "\n";

    Strategy copyTrueStrategy(trueStrategy);
    copyTrueStrategy.resetPathProbMat();
    copyTrueStrategy.resetCredits();

    std::vector<double> s0rewards;
    std::vector<double> s1rewards;

    if(trueStrategy.getOptimal())
    {
        s0rewards = {0,0,0,0,0,0,0,5,0};
        s1rewards = {0,0,0,0,0,0,0,0,5};
        copyTrueStrategy.setRewardsS0(s0rewards);
        copyTrueStrategy.setRewardsS1(s1rewards);


    }else{
        s0rewards = {0,0,0,0,0,0,5,5,0,0,0,0};
        copyTrueStrategy.setRewardsS0(s0rewards);
    }


    copyTrueStrategy.setAverageReward(0);
    

    for(int ses=0; ses < sessions; ses++)
    {
        double log_likelihood = copyTrueStrategy.getTrajectoryLikelihood(simRatData, ses); 
    }

    arma::mat probMatCopyTrueStrat = copyTrueStrategy.getPathProbMat();

    // R.assign("probMatTrueSim", probMatTrueStrat);
    // R.assign("probMatLikFunc", probMatCopyTrueStrat);

    // // Save the R variables to an RData file
    // R.parseEvalQ("save(list=c('probMatTrueSim', 'probMatLikFunc'), file='simTestProbMats.RData')");

    Rcpp::NumericMatrix probMatTrueStrat_r = Rcpp::wrap(probMatTrueStrat);
    Rcpp::NumericMatrix probMatCopyTrueStrat_r = Rcpp::wrap(probMatCopyTrueStrat);

    // Assign them to R variables
    R["probMatTrueStrat"] = probMatTrueStrat_r;
    R["probMatCopyTrueStrat"] = probMatCopyTrueStrat_r;

    // Save them as Rdata
    R.parseEvalQ("save(probMatTrueStrat, probMatCopyTrueStrat, file = 'simTestProbMats.RData')");

    arma::mat diffMatrix = arma::abs(probMatTrueStrat-probMatCopyTrueStrat);

    // Find the maximum value in the absolute difference matrix
    double maxDifference = arma::max(diffMatrix).eval()(0, 0);

    std::cout << "maxDifference:" << maxDifference << "\n";

    return;
}

std::map<std::pair<std::string, bool>, std::vector<double>> findParamsWithSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3)
{
    std::vector<std::string> learningRules = {"aca2","arl", "drl" };
    std::vector<bool> mazeModels = {true, false };

    std::map<std::pair<std::string, bool>, std::vector<double>> paramStrategies;

    std::cout << "Finding params for simdata"<< std::endl;

    for (const auto &lr : learningRules) 
    {
        for (const auto &optimal : mazeModels) 
        {
            std::string learningRule =  lr;   
            MazeGraph* maze;
            if(optimal)
            {
                maze = &optimalHybrid3;
            }else
            {
                maze = &suboptimalHybrid3;
            }

            std::cout << "learningRule=" << lr << ", optimal=" << optimal << std::endl;
            
            PagmoMle pagmoMle(ratdata, *maze, learningRule, optimal);
            //std::cout << "strategy=" << strategy.getName() <<std::endl;

            // Create a problem using Pagmo
            problem prob{pagmoMle};
            //problem prob{schwefel(30)};
            
            std::cout << "created problem" <<std::endl;
            // 2 - Instantiate a pagmo algorithm (self-adaptive differential
            // evolution, 100 generations).
            pagmo::algorithm algo{sade(10,2,2)};
            //algo.set_verbosity(10);

            //pagmo::algorithm algo{de(10)};

            std::cout << "creating archipelago" <<std::endl;
            // 3 - Instantiate an archipelago with 5 islands having each 5 individuals.
            archipelago archi{5u, algo, prob, 7u};

            // 4 - Run the evolution in parallel on the 5 separate islands 5 times.
            archi.evolve(5);
            //std::cout << "DONE1:"  << '\n';
            //system("pause"); 

            // 5 - Wait for the evolutions to finish.
            archi.wait_check();

            // 6 - Print the fitness of the best solution in each island.
            

            //system("pause"); 

            double champion_score = 1000000;
            std::vector<double> dec_vec_champion;
            for (const auto &isl : archi) {
                // std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
                std::vector<double> dec_vec = isl.get_population().champion_x();
                // for (auto const& i : dec_vec)
                //     std::cout << i << ", ";
                // std::cout << "\n" ;

                double champion_isl = isl.get_population().champion_f()[0];
                if(champion_isl < champion_score)
                {
                    champion_score = champion_isl;
                    dec_vec_champion = dec_vec;
                }
            }

            std::cout << "Final champion = " << champion_score << std::endl;
            for (auto const& i : dec_vec_champion)
                std::cout << i << ", ";
            std::cout << "\n" ;

            std::pair<std::string, bool> key(lr, optimal);
            paramStrategies[key] = dec_vec_champion;

        }
    }


    return paramStrategies;   
}

std::vector<double> findClusterParamsWithSimData(RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params)
{
        // Create a function to optimize
    PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3,params);
    std::cout << "Initialized problem class" <<std::endl;

    // Create a problem using Pagmo
    problem prob{pagmoprob};
    //problem prob{schwefel(30)};
    
    std::cout << "created problem" <<std::endl;
    // 2 - Instantiate a pagmo algorithm (self-adaptive differential
    // evolution, 100 generations).
    pagmo::algorithm algo{sade(10,2,2)};

    std::cout << "creating archipelago" <<std::endl;
    // 3 - Instantiate an archipelago with 5 islands having each 5 individuals.
    archipelago archi{5u, algo, prob, 7u};

    // 4 - Run the evolution in parallel on the 5 separate islands 5 times.
    archi.evolve(5);
    std::cout << "DONE1:"  << '\n';
    //system("pause"); 

    // 5 - Wait for the evolutions to finish.
    archi.wait_check();

    // 6 - Print the fitness of the best solution in each island.
    

    //system("pause"); 

    double champion_score = 1000000;
    std::vector<double> dec_vec_champion;
    for (const auto &isl : archi) {
        // std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
        std::vector<double> dec_vec = isl.get_population().champion_x();
        // for (auto const& i : dec_vec)
        //     std::cout << i << ", ";
        // std::cout << "\n" ;

        double champion_isl = isl.get_population().champion_f()[0];
        if(champion_isl < champion_score)
        {
            champion_score = champion_isl;
            dec_vec_champion = dec_vec;
        }
    }

    std::cout << "Final champion = " << champion_score << std::endl;
    for (auto const& i : dec_vec_champion)
        std::cout << i << ", ";
    std::cout << "\n" ;

    return dec_vec_champion;
}

void updateConfusionMatrix(std::vector<RecordResults> allSesResults)
{

    std::string filename = "confusionMatrix.txt";
    std::ifstream confMat_file(filename);

    std::vector<std::string> rows = {"aca2_Suboptimal_Hybrid3", "drl_Suboptimal_Hybrid3", "arl_Suboptimal_Hybrid3", "aca2_Optimal_Hybrid3", "drl_Optimal_Hybrid3", "arl_Optimal_Hybrid3"};
    std::vector<std::string> columns = {"aca2_Suboptimal_Hybrid3", "drl_Suboptimal_Hybrid3", "arl_Suboptimal_Hybrid3", "aca2_Optimal_Hybrid3", "drl_Optimal_Hybrid3", "arl_Optimal_Hybrid3", "None"};
    //std::vector<std::vector<int>> matrix(6, std::vector<int>(7, 0));
    std::vector<std::vector<int>> matrix;

    std::vector<std::string> rownames = {"acaSubopt", "drlSubopt", "arlSubopt", "acaOpt", "drlOpt", "arlOpt"};
    std::vector<std::string> colnames = {"acaSubopt", "drlSubopt", "arlSubopt", "acaOpt", "drlOpt", "arlOpt", "None"};

    std::map<std::string, int> rowLabelToIndex;
    std::map<std::string, int> colLabelToIndex;

    // Fill the maps with indices
    for (int i = 0; i < rows.size(); ++i) {
        rowLabelToIndex[rows[i]] = i;
    }

    for (int j = 0; j < columns.size(); ++j) {
        colLabelToIndex[columns[j]] = j;
    }

    if (confMat_file.is_open()) {
        // Skip the row header
        std::string line;
        std::getline(confMat_file, line);

        // Read the matrix
        std::string rowHeader;
        
        while (std::getline(confMat_file, line)) {
            std::istringstream rowStream(line);
            std::vector<int> row;
            int value;

            // Skip the row header
            rowStream >> rowHeader;
            //std::cout << "rowHeader=" << rowHeader << std::endl;

            // Read the values in the row
            while (rowStream >> value) {
                //std::cout << "value=" << value << std::endl;
                row.push_back(value);
            }

            matrix.push_back(row);
        }

        // Close the file
        confMat_file.close();
    } else {
        std::cout << "File not found or could not be opened. Creating a new matrix.\n";
    }

    // Print the matrix
    std::cout << "Matrix print before update:" << std::endl;
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << '\n';
    }


    
    for(size_t i=0; i<allSesResults.size();i++)
    {
        RecordResults recordResultsSes =  allSesResults[i];
        std::string selectedStrategy = recordResultsSes.getSelectedStrategy();   //column label
        std::string trueStrategy = recordResultsSes.getTrueGeneratingStrategy(); // rowLabel

        matrix[rowLabelToIndex[trueStrategy]][colLabelToIndex[selectedStrategy]] = matrix[rowLabelToIndex[trueStrategy]][colLabelToIndex[selectedStrategy]] +1  ;
        std::cout << "trueStrategy=" << trueStrategy << ", idx=" << rowLabelToIndex[trueStrategy] << "; selectedStrategy=" << selectedStrategy << ", idx=" << colLabelToIndex[selectedStrategy] << std::endl;
    }

    // Print the matrix
    std::cout << "Matrix print after update:" << std::endl;
    for (size_t i = 0; i < matrix.size(); ++i) {
        for (size_t j = 0; j < matrix[i].size(); ++j) {
            std::cout << matrix[i][j] << ' ';
        }
        std::cout << '\n';
    }

    std::ofstream outfile(filename);

    if (outfile.is_open()) {
        // Write the column names to the first line, separated by spaces
        outfile << std::setw(10) << " "; // Leave some space for the row names
        for (const auto& colname : colnames) {
            outfile << std::setw(10) << colname;
        }
        outfile << "\n"; // End the line

        // Write the matrix elements and the row names, separated by spaces
        for (size_t i = 0; i < matrix.size(); i++) {
            outfile << std::setw(10) << rownames[i]; // Write the row name
            for (size_t j = 0; j < matrix[i].size(); j++) {
                outfile << std::setw(10) << matrix[i][j]; // Write the matrix element
            }
            outfile << "\n"; // End the line
        }

        outfile.close();
        std::cout << "Matrix written to file.\n";
    } else {
        std::cout << "File could not be opened for writing.\n";
    }

}


void runEMOnSimData(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, std::map<std::pair<std::string, bool>, std::vector<double>> params, std::vector<double> v, bool debug)
{
    //// rat_103
    //std::vector<double> v = {0.11776, 0.163443, 0.0486187, 1e-7,0.475538, 0.272467, 1e-7 , 0.0639478, 1.9239e-06, 0.993274, 4.3431};
    
    ////rat_114
    //std::vector<double> v = {0.0334664, 0.351993, 0.00478871, 1.99929e-07, 0.687998, 0.380462, 9.68234e-07, 0.136651, 8.71086e-06, 0.292224, 3.95355};



    std::string rat = ratdata.getRat();

    // //ACA params
    double alpha_aca_subOptimal = params.find(std::make_pair("aca2", false))->second[0];
    double gamma_aca_subOptimal = params.find(std::make_pair("aca2", false))->second[1];

    double alpha_aca_optimal = params.find(std::make_pair("aca2", true))->second[0];
    double gamma_aca_optimal = params.find(std::make_pair("aca2", true))->second[1];

    //ARL params
    double alpha_arl_subOptimal = params.find(std::make_pair("arl", false))->second[0];
    double beta_arl_subOptimal = 1e-7;
    double lambda_arl_subOptimal = params.find(std::make_pair("arl", false))->second[1];
    
    double alpha_arl_optimal = params.find(std::make_pair("arl", true))->second[0];
    double beta_arl_optimal = 1e-7;
    double lambda_arl_optimal = params.find(std::make_pair("arl", true))->second[1];
 
    //DRL params
    double alpha_drl_subOptimal = params.find(std::make_pair("drl", false))->second[0];
    double beta_drl_subOptimal = 1e-4;
    double lambda_drl_subOptimal = params.find(std::make_pair("drl", false))->second[1];
    
    double alpha_drl_optimal = params.find(std::make_pair("drl", true))->second[0];
    double beta_drl_optimal = 1e-4;
    double lambda_drl_optimal = params.find(std::make_pair("drl", true))->second[1];

    // double crpAlpha = clusterParams[0];
    // double phi = clusterParams[1];
    // double eta = 0;


    // double alpha_aca_subOptimal = v[0];
    // double gamma_aca_subOptimal = v[1];

    // double alpha_aca_optimal = v[2];
    // double gamma_aca_optimal = v[3];

    // //ARL params
    // double alpha_arl_subOptimal = v[4];
    // double beta_arl_subOptimal = 1e-7;
    // double lambda_arl_subOptimal = v[5];
    
    // double alpha_arl_optimal = v[6];
    // double beta_arl_optimal = 1e-7;
    // double lambda_arl_optimal = v[7];
 
    // //DRL params
    // double alpha_drl_subOptimal = v[8];
    // double beta_drl_subOptimal = 1e-4;
    // double lambda_drl_subOptimal = v[9];
    
    // double alpha_drl_optimal = v[10];
    // double beta_drl_optimal = 1e-4;
    // double lambda_drl_optimal = v[11];

    
    double crpAlpha = v[0];
    double phi = v[1];
    double eta = v[14];

    double rS0_aca = v[2];
    double rS1_aca = v[3];
    double rS0_arl = v[4];
    double rS1_arl = v[5];
    double rS0_drl = v[6];
    double rS1_drl = v[7];

    double rS0_subopt_aca = v[8];
    double rS1_subopt_aca = v[9];
    double rS0_subopt_arl = v[10];
    double rS1_subopt_arl = v[11];
    double rS0_subopt_drl = v[12];
    double rS1_subopt_drl = v[13];

    std::vector<double> rewardsS0_aca = {0,0,0,0,0,0,0,rS0_aca,0};
    std::vector<double> rewardsS1_aca = {0,0,0,0,0,0,0,0,rS1_aca};
    std::vector<double> rewardsS0_arl = {0,0,0,0,0,0,0,rS0_arl,0};
    std::vector<double> rewardsS1_arl = {0,0,0,0,0,0,0,0,rS1_arl};
    std::vector<double> rewardsS0_drl = {0,0,0,0,0,0,0,rS0_drl,0};
    std::vector<double> rewardsS1_drl = {0,0,0,0,0,0,0,0,rS1_drl};
  
    std::vector<double> rewardsS0_subopt_aca = {0,0,0,0,0,0,rS0_subopt_aca,rS1_subopt_aca,0,0,0,0};
    std::vector<double> rewardsS0_subopt_arl = {0,0,0,0,0,0,rS0_subopt_arl,rS1_subopt_arl,0,0,0,0};
    std::vector<double> rewardsS0_subopt_drl = {0,0,0,0,0,0,rS0_subopt_drl,rS1_subopt_drl,0,0,0,0};


    if(debug)
    {
        std::cout << "alpha_aca_subOptimal=" << alpha_aca_subOptimal << ", gamma_aca_subOptimal=" << gamma_aca_subOptimal << ", alpha_aca_optimal=" << alpha_aca_optimal << ", gamma_aca_optimal=" << gamma_aca_optimal << std::endl;
        std::cout << "rat=" << rat << ", crpAlpha=" << crpAlpha << ", phi=" << phi << ", eta=" <<eta << std::endl;
    }
    
    // Create instances of Strategy
    auto aca2_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"aca2", alpha_aca_subOptimal, gamma_aca_subOptimal, 0, crpAlpha, phi, eta, false);
    auto aca2_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"aca2",alpha_aca_optimal, gamma_aca_optimal, 0, crpAlpha, phi, eta, true);
    
    auto drl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"drl", alpha_drl_subOptimal, beta_drl_subOptimal, lambda_drl_subOptimal, crpAlpha, phi, eta, false);
    auto drl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"drl",alpha_drl_optimal, beta_drl_optimal, lambda_drl_optimal, crpAlpha, phi, eta, true);

    auto arl_Suboptimal_Hybrid3 = std::make_shared<Strategy>(suboptimalHybrid3,"arl", alpha_arl_subOptimal, beta_arl_subOptimal, lambda_arl_subOptimal, crpAlpha, phi, eta, false);
    auto arl_Optimal_Hybrid3 = std::make_shared<Strategy>(optimalHybrid3,"arl",alpha_arl_optimal, beta_arl_optimal, lambda_arl_optimal, crpAlpha, phi, eta, true);


    std::vector<std::shared_ptr<Strategy>> strategies;
    strategies.push_back(aca2_Suboptimal_Hybrid3);
    strategies.push_back(aca2_Optimal_Hybrid3);

    strategies.push_back(drl_Suboptimal_Hybrid3);
    strategies.push_back(drl_Optimal_Hybrid3);

    strategies.push_back(arl_Suboptimal_Hybrid3);
    strategies.push_back(arl_Optimal_Hybrid3);

    aca2_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_aca);
    drl_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_drl);
    arl_Suboptimal_Hybrid3->setRewardsS0(rewardsS0_subopt_arl);
    
    aca2_Optimal_Hybrid3->setRewardsS0(rewardsS0_aca); 
    aca2_Optimal_Hybrid3->setRewardsS1(rewardsS1_aca);

    drl_Optimal_Hybrid3->setRewardsS0(rewardsS0_drl); 
    drl_Optimal_Hybrid3->setRewardsS1(rewardsS1_drl);

    arl_Optimal_Hybrid3->setRewardsS0(rewardsS0_arl); 
    arl_Optimal_Hybrid3->setRewardsS1(rewardsS1_arl);    


    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    std::vector<std::string> cluster;
    std::string last_choice;
    
    arma::mat probMat;

    std::vector<RecordResults> allSesResults;

    std::vector<std::string> trueStrategies = ratdata.getGeneratorStrategies();

    for(int ses=0; ses < sessions; ses++)
    {
        
        // if(ses==0)
        // {
        //     initRewardVals(ratdata, ses, strategies, debug);
        // }
 
        RecordResults sessionResults;
        sessionResults.setTrueGeneratingStrategy(trueStrategies[ses]);

        arma::mat probMat_sess = estep_cluster_update(ratdata, ses, strategies, cluster, last_choice, true, sessionResults);
        mstep(ratdata, ses, strategies, cluster, debug, sessionResults);

        probMat = arma::join_cols(probMat, probMat_sess);

        allSesResults.push_back(sessionResults);

        //RecordResults sessionResults(selectedStrategy, probabilityMatrix, trueGeneratingStrategy, posteriors, rewardVectorS0, rewardVectorS1);
    }

    arma::mat& aca2_suboptimal_probs =  aca2_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& aca2_optimal_probs =  aca2_Optimal_Hybrid3->getPathProbMat();
    arma::mat& drl_suboptimal_probs =  drl_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& drl_optimal_probs =  drl_Optimal_Hybrid3->getPathProbMat();
    arma::mat& arl_suboptimal_probs =  arl_Suboptimal_Hybrid3->getPathProbMat();
    arma::mat& arl_optimal_probs =  arl_Optimal_Hybrid3->getPathProbMat();

    probMat.save("ProbMat_Sim_" + rat+ ".csv", arma::csv_ascii);
    updateConfusionMatrix(allSesResults);


    aca2_suboptimal_probs.save("aca2_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    aca2_optimal_probs.save("aca2_optimal_probs_"+ rat+".csv", arma::csv_ascii);
    drl_suboptimal_probs.save("drl_suboptimal_probs_"+ rat+".csv", arma::csv_ascii);
    drl_optimal_probs.save("drl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);
    arl_suboptimal_probs.save("arl_suboptimal_probs_" + rat+ ".csv", arma::csv_ascii);
    arl_optimal_probs.save("arl_optimal_probs_" + rat+ ".csv", arma::csv_ascii);


}


void testRecovery(RatData& ratdata, MazeGraph& suboptimalHybrid3, MazeGraph& optimalHybrid3, RInside &R)
{
    // Read the params from from rat param file, e.g rat_103.txt
    std::string rat = ratdata.getRat();
    std::string filename = rat + ".txt";
    std::ifstream infile(filename);
    std::map<std::pair<std::string, bool>, std::vector<double>> ratParams;
    boost::archive::text_iarchive ia(infile);
    ia >> ratParams;
    infile.close();

    //read clusterParams.txt to get the parameters for rat
    std::string filename_cluster = "clusterParams.txt";
    std::ifstream cluster_infile(filename_cluster);
    std::map<std::string, std::vector<double>> clusterParams;
    boost::archive::text_iarchive ia_cluster(cluster_infile);
    ia_cluster >> clusterParams;
    cluster_infile.close();

    for(int i=0; i < 12; i++)
    {
        RatData ratSimData = generateSimulation(ratdata, suboptimalHybrid3, optimalHybrid3, ratParams,clusterParams, R, i);
        std::map<std::pair<std::string, bool>, std::vector<double>> simRatParams = findParamsWithSimData(ratSimData, suboptimalHybrid3, optimalHybrid3);
        std::vector<double> simClusterParams = findClusterParamsWithSimData(ratSimData, suboptimalHybrid3, optimalHybrid3,simRatParams);
        runEMOnSimData(ratSimData, suboptimalHybrid3, optimalHybrid3, simRatParams, simClusterParams, true);

    }
  
     

}





