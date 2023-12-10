#include "InverseRL.h"
// #include "utils.h"
using namespace Rcpp;

// using namespace Rcpp;

double getAvgRwdQLearningLik(const RatData& ratdata, int session, Strategy& strategy, bool sim)
{
  arma::mat allpaths = ratdata.getPaths();
  std::string strategy_name = strategy.getName();
  //std::cout << "strategy_name= "<< strategy_name << std::endl;

  //printFirst5Rows(allpaths,"allpaths");
  arma::mat turnTimes;
  
  if(strategy_name == "Paths")
  {
    turnTimes = ratdata.getPaths();
  }
  else if(strategy_name == "Turns")
  {
    turnTimes = ratdata.getTurns();
  }
  else if(strategy_name == "Hybrid1")
  {
    turnTimes = ratdata.getHybrid1();
  }
  else if(strategy_name == "Hybrid2")
  {
    turnTimes = ratdata.getHybrid2();
  }
  else if(strategy_name == "arl_Suboptimal_Hybrid3")
  {
    turnTimes = ratdata.getHybrid3();
  }
  else if(strategy_name == "arl_Optimal_Hybrid3")
  {
    turnTimes = ratdata.getHybrid3();
  }
  else if(strategy_name == "Hybrid4")
  {
    turnTimes = ratdata.getHybrid4();
  }
  
  //Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(testModel.slot("nodeGroups"));
  
  int episodeNb = 0; 
  
  double alpha = strategy.getAlpha();
  double beta = strategy.getGamma();
  double lambda = strategy.getLambda();
  
  //Rcpp::Rcout <<  "allpaths.col(4)="<<allpaths.col(4) <<std::endl;
  
  std::vector<double> mseMatrix;
  //int mseRowIdx = 0;
  
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  
  arma::vec turnTime_method;
  if (sim)
  {
    turnTime_method = turnTimes.col(3);
  }
  else if (strategy_name == "Paths")
  {
    turnTime_method = turnTimes.col(3);
  }
  else
  {
    turnTime_method = turnTimes.col(5);
  }

 
  int episode = 1;
  int pathCounter=0;
  BoostGraph& S0 = strategy.getStateS0();
  BoostGraph& S1 = strategy.getStateS1();

  
  int sessId = uniqSessIdx(session);
  //Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
  arma::uvec sessionIdx = arma::find(sessionVec == sessId);
  arma::vec actions_sess = allpath_actions.elem(sessionIdx);
  arma::vec states_sess = allpath_states.elem(sessionIdx);
  arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);

  if (sim != 1)
  {
    states_sess = states_sess - 1;
    actions_sess = actions_sess - 1;
  }

  
  arma::uvec turnTimes_idx; 
  if (strategy_name == "Paths")
  {
    turnTimes_idx = arma::find(sessionVec == sessId); ;
  }
  else
  {
    turnTimes_idx = arma::find(turnTimes.col(4) == sessId); 
  }

  
  arma::vec turn_times_session = turnTime_method.elem(turnTimes_idx);
  arma::uword session_turn_count = 0;


  //std::cout <<"sessId=" << sessId << ", strategy_name= "<< strategy_name << ", sessionVec.size()=" << sessionVec.n_rows << ", turnTime_method.size()=" << turnTime_method.n_rows << ", turn_times_session.size=" << turn_times_session.n_rows << std::endl;
  
  int initState = 0;
  bool changeState = false;
  bool returnToInitState = false;
  int score_episode = 0;
  float avg_score = 0;
  bool resetVector = true;
  int nrow = actions_sess.n_rows;
  double averageReward = strategy.getAverageReward();
  int S = states_sess(0); 
  int A = 0;
  std::vector<std::string> episodeTurns;
  std::vector<int> episodeTurnStates;
  std::vector<double> episodeTurnTimes;

  BoostGraph::Vertex prevNode;
  BoostGraph::Vertex currNode;
  BoostGraph::Vertex rootNode;
  std::vector<double> rewardVec;
  BoostGraph* graph;
  std::vector<double> rewardsS0 = strategy.getRewardsS0();
  std::vector<double> rewardsS1 = strategy.getRewardsS1();

  std::map<std::pair<BoostGraph::Vertex, int>, double> eligibilityTraces;


  
  for (int i = 0; i < nrow; i++)
  {
    
    if (resetVector)
    {
      initState = S;
      //Rcpp::Rcout <<"initState="<<initState<<std::endl;
      resetVector = false;
    }
    
    A = actions_sess(i);
    
    int S_prime = 0;
    if(i < (nrow-1))
    {
      S_prime = states_sess(i + 1);
    }
    
    if (S_prime != initState)
    {
      changeState = true;
    }
    else if (S_prime == initState && changeState)
    {
      returnToInitState = true;
    }
    
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
    double logpathProb = 0;
    double currTurnReward = 0;
    for (int j = 0; j < nbOfTurns; j++)
    {
      
      std::string currTurn = turns[j]; 
      currNode = graph->findNode(currTurn);
      int nodeId = graph->getNodeId(currNode);
      currTurnReward = rewardVec[nodeId];

      double turntime = turn_times_session(session_turn_count);

      
      BoostGraph::Edge edge;
      if(j==0)
      {
        edge = graph->findEdge(rootNode, currNode);
      }
      else
      {
        edge = graph->findEdge(prevNode, currNode);
      }

      double prob_a = graph->getEdgeProbability(edge);
      logpathProb = logpathProb + prob_a;

      
      double qMax = -100000;
      // CASE1: IF CURR TURN IS AN INERMEDIATE TURN IN THE MAZE, DETERMINE QMAX USING EDGES
      if (!graph->isTerminalVertex(currTurn))
      {
        BoostGraph::Vertex maxChild = graph->getChildWithMaxCredit(currTurn);
        qMax = graph->getNodeCredits(maxChild);
      }
      else if(i != (nrow - 1)) // CASE2: If curr turn leads to next box, then select qmax using actions from next box
      {
        int S_prime = states_sess(i + 1);
        BoostGraph *newGraph;
        BoostGraph::Vertex newRootNode;
        if (S_prime == 0 && strategy.getOptimal())
        {
          newGraph = &S0;
          newRootNode = newGraph->findNode("E");
        }else if(S_prime == 1 && strategy.getOptimal())
        {
          newGraph = &S1;
          newRootNode = newGraph->findNode("I");
        }else if(S_prime == 0 && !strategy.getOptimal())
        {
          newGraph = &S0;
          newRootNode = graph->findNode("E");
        }else if(S_prime == 1 && !strategy.getOptimal())
        {
          newGraph = &S0;
          newRootNode = graph->findNode("I");
        }

        std::string rootNode = newGraph->getNodeName(newRootNode);
        BoostGraph::Vertex maxChild = newGraph->getChildWithMaxCredit(rootNode);
        qMax = newGraph->getNodeCredits(maxChild);
      }
      else // CASE3: i is final turn of the session
      {
        qMax = 0;
      }
      //Update eligibility trace
      double decay_factor = lambda;
      //std::cout << "exp(-beta*turntime) = " << exp(-beta*turntime) << ", lambda=" << lambda << ", decay_factor=" <<decay_factor << "\n";
      S0.updateAllEligibilityTraces(decay_factor);
      if(strategy.getOptimal())
      {
        S1.updateAllEligibilityTraces(decay_factor);
      }

      double etrace_currNode = graph->getEligibilityTrace(currNode);
      etrace_currNode = etrace_currNode + 1;
      graph->setEligibilityTrace(currNode,etrace_currNode);

      double currNode_credit = graph->getNodeCredits(currNode);
      double td_err = currTurnReward - (averageReward * turntime) + qMax - currNode_credit;

      //std::cout << "S=" <<S << ", A=" << A << ", i=" << i << ", j=" << j << ", currTurn=" << currTurn << ", currTurnReward=" << currTurnReward << ", td_err=" << td_err << ", nodeCredits=" << graph->getNodeCredits(currNode) << ", etrace=" << graph->getEligibilityTrace(currNode) << ", qMax=" << qMax << ", turntime=" << turntime << ", averageReward="<< averageReward << std::endl;


      averageReward = averageReward + (beta * td_err);

      //double nodeCredits = graph->getNodeCredits(currNode); // for debug, comment if not debugging

      
      S0.tdUpdateAllVertexCredits(alpha, td_err);
      if(strategy.getOptimal())
      {
        S1.tdUpdateAllVertexCredits(alpha, td_err);
      }

     //std::cout << "S=" <<S << ", A=" << A << ", i=" << i << ", j=" << j << ", currTurn=" << currTurn << ", updated_nodeCredits=" << graph->getNodeCredits(currNode) << ", updated_averageReward="<< averageReward << std::endl;



    //  S0.printNodeCredits();
    //   if(strategy.getOptimal())
    //   {
    //     S1.printNodeCredits();
    //   }

      // S0.printNodeEligibilityTraces();
      // if(strategy.getOptimal())
      // {
      //   S1.printNodeEligibilityTraces();
      // }


    
      S0.updateEdgeProbabilitiesSoftmax();
      if(strategy.getOptimal())
      {
        S1.updateEdgeProbabilitiesSoftmax();
      }
      
      session_turn_count++;
      prevNode = currNode;
    }
    if(A != 6)
    {
      mseMatrix.push_back(logpathProb);
    }
    else
    {
      mseMatrix.push_back(0);
    }
    pathCounter = pathCounter+1;
    
       
    //Check if episode ended
    if (returnToInitState || (i==nrow-1))
    {
      //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
      // if(debug)
      // {
      //   Rcpp::Rcout <<  "End of episode"<<std::endl;
      // }
      changeState = false;
      returnToInitState = false;
       
      episode = episode + 1;
      resetVector = true;

      // Reset eligibility trace to 0 at the end of an episode
      double decay_factor = 0;
      S0.updateAllEligibilityTraces(decay_factor);
      if(strategy.getOptimal())
      {
        S1.updateAllEligibilityTraces(decay_factor);
      }
    }
    
    S = S_prime;
    //trial=trial+1;

    strategy.updatePathProbMat();
    
  }

  strategy.setAverageReward(averageReward);
    
  double result = std::accumulate(mseMatrix.begin(), mseMatrix.end(), 0.0);
  

  return (result);
}

