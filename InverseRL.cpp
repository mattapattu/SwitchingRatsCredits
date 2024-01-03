#include "InverseRL.h"
#include <cmath>

double computeTrajectoryLik(const RatData& ratdata, int session, Strategy& strategy)
{
    bool debug = false;
    std::string creditAssignment = strategy.getLearningRule();
    //std::cout << "creditAssignment=" << creditAssignment << std::endl;
    double lik;
    if(creditAssignment == "aca2")
    {
      lik =   getAca2SessionLikelihood(ratdata, session, strategy);
    }
    else if(creditAssignment == "arl")
    {
      lik =   getAvgRwdQLearningLik(ratdata, session, strategy);
    }
    else if(creditAssignment == "drl")
    {
      lik =   getDiscountedRwdQlearningLik(ratdata, session, strategy);
    }

    if (std::isnan(lik)) {
                    
      std::cout << "Likelihood is nan. Check" << std::endl;
      std::exit(EXIT_FAILURE);
    }

    return(lik);
}

std::pair<arma::mat, arma::mat> simulateTrajectory(const RatData& ratdata, int session, Strategy& strategy)
{
    bool debug = false;
    std::string creditAssignment = strategy.getLearningRule();
    //std::cout << "creditAssignment=" << creditAssignment << std::endl;
    std::pair<arma::mat, arma::mat> simData;
    if(creditAssignment == "aca2")
    {
       simData = simulateAca2(ratdata, session, strategy);
    }
    else if(creditAssignment == "arl")
    {
      simData = simulateAvgRwdQLearning(ratdata, session, strategy);
    }
    else if(creditAssignment == "drl")
    {
      simData = simulateDiscountedRwdQlearning(ratdata, session, strategy);
    }

    
    return(simData);
}

void printFirst5Rows(const arma::mat& matrix, std::string matname) {
  std::cout << "Printing first 5 rows of <<" << matname << std::endl;
  for (int i = 0; i < 5; ++i) {
    std::cout << "Row " << (i + 1) << ": ";
    for (int j = 0; j < matrix.n_cols; ++j) {
      std::cout << matrix(i, j) << " ";
    }
    std::cout << std::endl;
  }
}


void updateRewardFunction(const RatData& ratdata, int session, Strategy& strategy)
{
  arma::mat allpaths = ratdata.getPaths();
  std::string strategy_name = strategy.getName();
  
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);

  bool sim = ratdata.getSim();
  
  BoostGraph& S0 = strategy.getStateS0();
  BoostGraph& S1 = strategy.getStateS1();

  int sessId = uniqSessIdx(session);
  //Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
  arma::uvec sessionIdx = arma::find(sessionVec == sessId);
  arma::vec actions_sess = allpath_actions.elem(sessionIdx);
  arma::vec states_sess = allpath_states.elem(sessionIdx);
  arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);
  
  int initState = 0;
  bool changeState = false;
  bool returnToInitState = false;
  // float avg_score = 0;
  bool resetVector = true;
  int nrow = actions_sess.n_rows;
  int S;
  if(sim == 1)
  {
    S = states_sess(0); 
  }
  else
  {
    S = states_sess(0) - 1; 
  }
  int A = 0;
  std::vector<double> rewardUpdatesS0;
  std::vector<double> rewardUpdatesS1;
  
  for (int i = 0; i < nrow; i++)
  {
    
    if (resetVector)
    {
      initState = S;
      //Rcpp::Rcout <<"initState="<<initState<<std::endl;
      resetVector = false;
    }
    
    //double pathReward=0.0;
    
    if (sim == 1)
    {
      A = actions_sess(i);
    }
    else
    {
      A = actions_sess(i) - 1;
    }

    double R = rewards_sess(i);
    if(R > 0)
    {
      R = 5;
    }

    int S_prime = 0;
    if(i < (nrow-1))
    {
      if (sim == 1)
      {
        S_prime = states_sess(i + 1);
      }
      else
      {
        S_prime = states_sess(i + 1) - 1;
      }
    }
    
    if (S_prime != initState)
    {
      changeState = true;
    }
    else if (S_prime == initState && changeState)
    {
      returnToInitState = true;
    }
    
    BoostGraph::Vertex prevNode;
    BoostGraph::Vertex currNode;
    BoostGraph::Vertex rootNode;
    std::vector<double> rewardVec;
    BoostGraph* graph;
    std::vector<double> rewardsS0 = strategy.getRewardsS0();
    std::vector<double> rewardsS1 = strategy.getRewardsS1();

    if (S == 0 && strategy.getOptimal())
    {
      graph = &S0;
      rootNode = graph->findNode("E");
      rewardVec = rewardsS0;
    }else if(S == 1 && strategy.getOptimal())
    {
      graph = &S1;
      rootNode = graph->findNode("I");
      rewardVec = rewardsS1;
    }else if(S == 0 && !strategy.getOptimal())
    {
      graph = &S0;
      rootNode = graph->findNode("E");
      rewardVec = rewardsS0;
    }else if(S == 1 && !strategy.getOptimal())
    {
      graph = &S0;
      rootNode = graph->findNode("I");
      rewardVec = rewardsS0;
    }
    std::vector<std::string> turns = graph->getTurnsFromPaths(A, S, strategy.getOptimal());
    int nbOfTurns = turns.size();


    for (int j = 0; j < nbOfTurns; j++)
    {
      std::string currTurn = turns[j]; 
      currNode = graph->findNode(currTurn);
      int nodeId = graph->getNodeId(currNode);
      std::vector<double> crpPosterior = strategy.getCrpPosteriors();
      if (j == (nbOfTurns - 1))
      {
        rewardVec[nodeId] += strategy.getPhi() * crpPosterior.back() *(R-rewardVec[nodeId]);
      }
      else
      {
        rewardVec[nodeId] +=  strategy.getPhi() * crpPosterior.back() *(-rewardVec[nodeId]);;
      }
      //std::cout << "strategy=" << strategy_name << ", currNode=" << ", rewardVec=" << rewardVec[nodeId] << std::endl;

      if (S == 0 && strategy.getOptimal())
      {
        strategy.setRewardsS0(rewardVec);
      }else if(S == 1 && strategy.getOptimal())
      {
        strategy.setRewardsS1(rewardVec);
      }else if(S == 0 && !strategy.getOptimal())
      {
        strategy.setRewardsS0(rewardVec);
      }else if(S == 1 && !strategy.getOptimal())
      {
        strategy.setRewardsS0(rewardVec);
      }
     
      BoostGraph::Edge edge;
      if(j==0)
      {
        edge = graph->findEdge(rootNode, currNode);
      }
      else
      {
        edge = graph->findEdge(prevNode, currNode);
      }

 
      //Rcpp::Rcout <<"prob_a="<< prob_a << ", pathProb=" <<pathProb <<std::endl;
      
      prevNode = currNode;
    } 
      
    //Check if episode ended
    if (returnToInitState || (i==nrow-1))
    {
      //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
      changeState = false;
      returnToInitState = false;   
      
      resetVector = true;
    }
    S = S_prime;
  }

  std::vector<double> rewardsS0 = strategy.getRewardsS0();
  std::vector<double> rewardsS1 = strategy.getRewardsS1();
 
  // std::cout << "rewardsS0: ";
  // for (const auto& element : rewardsS0) {
  //   std::cout << element << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "rewardsS1: ";
  // for (const auto& element : rewardsS1) {
  //   std::cout << element << " ";
  // }
  // std::cout << std::endl;


  return ;
}


void initializeRewards(const RatData& ratdata, int session, Strategy& strategy)
{
  arma::mat allpaths = ratdata.getPaths();
  std::string strategy_name = strategy.getName();

  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);

  bool sim = ratdata.getSim();
  
  
  BoostGraph& S0 = strategy.getStateS0();
  BoostGraph& S1 = strategy.getStateS1();

  int sessId = sessionVec(session);
  //Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
  arma::uvec sessionIdx = arma::find(sessionVec == sessId);
  arma::vec actions_sess = allpath_actions.elem(sessionIdx);
  arma::vec states_sess = allpath_states.elem(sessionIdx);
  arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);

  //double phi = 0.001;
  double phi = strategy.getPhi();
  
  
  int initState = 0;
  bool changeState = false;
  bool returnToInitState = false;
  // float avg_score = 0;
  bool resetVector = true;
  int nrow = actions_sess.n_rows;
  int S;
  if(sim == 1)
  {
    S = states_sess(0); 
  }
  else
  {
    S = states_sess(0) - 1; 
  }
  int A = 0;
  std::vector<double> rewardUpdatesS0;
  std::vector<double> rewardUpdatesS1;
  
  for (int i = 0; i < nrow; i++)
  {
    
    if (resetVector)
    {
      initState = S;
      //Rcpp::Rcout <<"initState="<<initState<<std::endl;
      resetVector = false;
    }
    
    //double pathReward=0.0;
    
    if (sim == 1)
    {
      A = actions_sess(i);
    }
    else
    {
      A = actions_sess(i) - 1;
    }

    double R = rewards_sess(i);
    if(R > 0)
    {
      R = 5;
    }

    int S_prime = 0;
    if(i < (nrow-1))
    {
      if (sim == 1)
      {
        S_prime = states_sess(i + 1);
      }
      else
      {
        S_prime = states_sess(i + 1) - 1;
      }
    }
    
    if (S_prime != initState)
    {
      changeState = true;
    }
    else if (S_prime == initState && changeState)
    {
      returnToInitState = true;
    }
    
    BoostGraph::Vertex prevNode;
    BoostGraph::Vertex currNode;
    BoostGraph::Vertex rootNode;
    std::vector<double> rewardVec;
    BoostGraph* graph;
    std::vector<double> rewardsS0 = strategy.getRewardsS0();
    std::vector<double> rewardsS1 = strategy.getRewardsS1();


    if (S == 0 && strategy.getOptimal())
    {
      graph = &S0;
      rootNode = graph->findNode("E");
      rewardVec = rewardsS0;
    }else if(S == 1 && strategy.getOptimal())
    {
      graph = &S1;
      rootNode = graph->findNode("I");
      rewardVec = rewardsS1;
    }else if(S == 0 && !strategy.getOptimal())
    {
      graph = &S0;
      rootNode = graph->findNode("E");
      rewardVec = rewardsS0;
    }else if(S == 1 && !strategy.getOptimal())
    {
      graph = &S0;
      rootNode = graph->findNode("I");
      rewardVec = rewardsS0;
    }
    std::vector<std::string> turns = graph->getTurnsFromPaths(A, S, strategy.getOptimal());
    int nbOfTurns = turns.size();


    for (int j = 0; j < nbOfTurns; j++)
    {
      std::string currTurn = turns[j]; 
      currNode = graph->findNode(currTurn);
      int nodeId = graph->getNodeId(currNode);
      if (j == (nbOfTurns - 1))
      {
        rewardVec[nodeId] += phi * (R-rewardVec[nodeId]);
      }
      else
      {
        rewardVec[nodeId] +=  phi * (-rewardVec[nodeId]);;
      }

      //std::cout <<"currTurn="<< currTurn << ", R=" <<R << ", rewardVec[nodeId]=" <<rewardVec[nodeId] <<std::endl;

      if (S == 0 && strategy.getOptimal())
      {
        strategy.setRewardsS0(rewardVec);
      }else if(S == 1 && strategy.getOptimal())
      {
        strategy.setRewardsS1(rewardVec);
      }else if(S == 0 && !strategy.getOptimal())
      {
        strategy.setRewardsS0(rewardVec);
      }else if(S == 1 && !strategy.getOptimal())
      {
        strategy.setRewardsS0(rewardVec);
      }


      prevNode = currNode;
    } 
      
    //Check if episode ended
    if (returnToInitState || (i==nrow-1))
    {
      //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
      changeState = false;
      returnToInitState = false;   
      resetVector = true;
    }
    S = S_prime;
  }

  std::vector<double> rewardsS0 = strategy.getRewardsS0();
  std::vector<double> rewardsS1 = strategy.getRewardsS1();
 
  // std::cout << "rewardsS0: ";
  // for (const auto& element : rewardsS0) {
  //   std::cout << element << " ";
  // }
  // std::cout << std::endl;

  // std::cout << "rewardsS1: ";
  // for (const auto& element : rewardsS1) {
  //   std::cout << element << " ";
  // }
  // std::cout << std::endl;

  return ;
}



std::vector<std::string> generatePathTrajectory(Strategy& strategy, BoostGraph* graph, BoostGraph::Vertex rootNode)
{
  std::vector<BoostGraph::Edge> edges;
  std::vector<std::string> turns;
  BoostGraph::Vertex node;
  node = rootNode;
  edges = graph->getOutGoingEdges(node);
      
  while (!edges.empty())
  {
    BoostGraph::Vertex childSelected = graph->sampleChild(node);
    std::string turnSelected = graph->getNodeName(childSelected);
    turns.push_back(turnSelected);
      
    node = childSelected;
    edges = graph->getOutGoingEdges(node);
  }
  return(turns);
}

int getNextState(int curr_state, int action)
{
  //Rcpp::Rcout << "curr_state=" << curr_state << ", action=" << action << ", last_turn=" << last_turn << std::endl;
  int new_state = -1;
  if (action == 4 || action == 5)
  {
    new_state = curr_state;
  }
  else if (curr_state == 0)
  {
    new_state = 1;
  }
  else if (curr_state == 1)
  {
    new_state = 0;
  }
  
  //Rcpp::Rcout << "new_state=" << new_state << std::endl;
  
  return (new_state);
}

double simulateTurnDuration(arma::mat hybridTurnTimes, int hybridTurnId, int state, int session, Strategy& strategy)
{

  //std::vector<int> turnStages = {0,totalPaths/4,totalPaths};
  std::string strategy_name = strategy.getName();
  int start = -1;
  int end = 0;
  int changepoint_ses = 10; //Rough assumption that durations stabilize after 10 sessions, not related to changepoint in the EM inference.
  arma::uvec indices = arma::find(hybridTurnTimes.col(4) > changepoint_ses, 1, "first");

  if(session < changepoint_ses)
  {
    start = 0;
    end = indices(0) - 1;
  }
  else
  {
    start = indices(0);
    end = hybridTurnTimes.n_rows-1;
  }

  //Rcpp::Rcout << "start=" << start << ", end=" << end << std::endl;
  
  arma::mat turnTimesMat_stage = hybridTurnTimes.rows(start,end);
  arma::vec turnDurations_stage = turnTimesMat_stage.col(5);

  double turnId = hybridTurnId;

  if(!strategy.getOptimal())
  {
    if(hybridTurnId == 1)
    {
      state = 0;
      hybridTurnId = 1;
    }else if(hybridTurnId == 2)
    {
      state = 0;
      hybridTurnId = 2;
    }else if(hybridTurnId == 3)
    {
      state = 0;
      hybridTurnId = 3;
    }else if(hybridTurnId == 4)
    {
      state = 0;
      hybridTurnId = 5;
    }else if(hybridTurnId == 5)
    {
      state = 0;
      hybridTurnId = 6;
    }else if(hybridTurnId == 6)
    {
      state = 0;
      hybridTurnId = 7;
    }else if(hybridTurnId == 7)
    {
      state = 1;
      hybridTurnId = 8;
    }else if(hybridTurnId == 9)
    {
      state = 1;
      hybridTurnId = 1;
    }else if(hybridTurnId == 10)
    {
      state = 1;
      hybridTurnId = 2;
    }else if(hybridTurnId == 11)
    {
      state = 1;
      hybridTurnId = 3;
    }
  }


  // Get all turn ids from turnTimesMat belonging to current turnStage
  arma::uvec arma_idx = arma::find(turnTimesMat_stage.col(3) == hybridTurnId && turnTimesMat_stage.col(2) == state);
  
  double hybridTurnDuration = 0;
  
  if(arma_idx.size() < 3)
  {
    //std::cout << "hybridTurnId=" << hybridTurnId << ", state=" << state << ", arma_idx.size()=" << arma_idx.size() << std::endl;

    if(hybridTurnId == 2 && state == 0){

      hybridTurnId = 8;

    }else if((hybridTurnId == 4 || hybridTurnId == 5|| hybridTurnId == 6 ) && state == 0)
    {
      hybridTurnId = 3;
    }else if(hybridTurnId == 2 && state == 1){

      hybridTurnId = 7;

    }else if((hybridTurnId == 4 || hybridTurnId == 5|| hybridTurnId == 6 ) && state == 1)
    {
      hybridTurnId = 3;
    } 
    
    arma_idx = arma::find(turnTimesMat_stage.col(3) == hybridTurnId && turnTimesMat_stage.col(2) == state); 
    //std::cout << "New hybridTurnId=" << hybridTurnId << ", state=" << state << ", arma_idx.size()=" << arma_idx.size() << std::endl;
  }
  // else{
  //   // if(strategy_name == "aca2_Suboptimal_Hybrid3" || strategy_name == "aca2_Optimal_Hybrid3")
  //   // {
  //   //   hybridTurnDuration = 100;
  //   // }else{
  //   //   hybridTurnDuration = 50000;
  //   // }
    
  // }


    arma::vec turnDurations_stage_turnid = turnDurations_stage.rows(arma_idx);
    double mean_value = arma::mean(turnDurations_stage_turnid);
    double std_deviation = arma::stddev(turnDurations_stage_turnid);
    double lambda = 1.0 / mean_value;

    std::random_device rd;
    std::default_random_engine generator(rd());
    std::exponential_distribution<> distribution(lambda);
  // mean=0, stddev=1

    hybridTurnDuration = distribution(generator);

  
  // //If turn not present in rat data, set duration to a very low value to give it low credits (aca), high value otherwise
  // if(strategy_name == "aca2_Suboptimal_Hybrid3" || strategy_name == "aca2_Optimal_Hybrid3")
  // {
  //   if(strategy_name == "aca2_Suboptimal_Hybrid3")
  //   {
  //     if(turnId == 2||turnId == 4||turnId == 5||turnId == 10)
  //     {
  //       hybridTurnDuration = 100;
  //     }
  //   }else{
  //     if(turnId == 2||turnId == 4||turnId == 5||turnId == 6)
  //     {
  //       hybridTurnDuration = 100;
  //     }
  //   }
  // }else{
  //   if(!strategy.getOptimal())
  //   {
  //     if(turnId == 2||turnId == 4||turnId == 5||turnId == 10)
  //     {
  //       hybridTurnDuration = 50000;
  //     }
  //   }else{
  //     if(turnId == 2||turnId == 4||turnId == 5||turnId == 6)
  //     {
  //       hybridTurnDuration = 50000;
  //     }
  //   }
  // }
    

  
  return(hybridTurnDuration);
}

