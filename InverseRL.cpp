#include "InverseRL.h"

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


void updateRewardFunction(const RatData& ratdata, int session, Strategy& strategy, bool sim)
{
  arma::mat allpaths = ratdata.getPaths();
  std::string strategy_name = strategy.getName();
  
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  
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


void initializeRewards(const RatData& ratdata, int session, Strategy& strategy, bool sim)
{
  arma::mat allpaths = ratdata.getPaths();
  std::string strategy_name = strategy.getName();

  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  
  
  BoostGraph& S0 = strategy.getStateS0();
  BoostGraph& S1 = strategy.getStateS1();

  int sessId = sessionVec(session);
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
      if (j == (nbOfTurns - 1))
      {
        rewardVec[nodeId] += strategy.getPhi() * (R-rewardVec[nodeId]);
      }
      else
      {
        rewardVec[nodeId] +=  strategy.getPhi() * (-rewardVec[nodeId]);;
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


