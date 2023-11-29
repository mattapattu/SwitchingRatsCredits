#include "InverseRL.h"



void acaCreditUpdate(std::vector<std::string> episodeTurns, std::vector<int> episodeTurnStates, std::vector<double> episodeTurnTimes, double score_episode, Strategy& strategy)
{

  arma::vec episodeTurnStates_arma = arma::conv_to<arma::vec>::from(episodeTurnStates);
  arma::vec episodeTurnTimes_arma(episodeTurnTimes);

  std::unordered_map<int, std::unordered_map<std::string, double>> totalDurationByStateAndAction;

  for (size_t i = 0; i < episodeTurnStates_arma.n_elem; ++i) {
    int state = episodeTurnStates_arma(i);
    const std::string& action = episodeTurns[i];
    double duration = episodeTurnTimes_arma(i);

    // Insert the duration into the map for the corresponding state and action
    totalDurationByStateAndAction[state][action] += duration;
  }

   
  //Compute activity for each unque (S,A) in the episode
  for (const auto& stateEntry : totalDurationByStateAndAction) {
    int state = stateEntry.first;
    const auto& actionDurations = stateEntry.second;
    
    BoostGraph* graph = nullptr;
    if (state == 0 && strategy.getOptimal())
    {
      graph = &strategy.getStateS0();
    }else if(state == 1 && strategy.getOptimal())
    {
      graph = &strategy.getStateS1();
    }else if(state == 0 && !strategy.getOptimal())
    {
      graph = &strategy.getStateS0();
    }else if(state == 1 && !strategy.getOptimal())
    {
      graph = &strategy.getStateS0();
    }          



    //std::cout << "State " << state << ":" << std::endl;
    for (const auto& actionEntry : actionDurations) {
        const std::string& action = actionEntry.first;
        double totalDuration = actionEntry.second;

        BoostGraph::Vertex currNode = graph->findNode(action); 
        double activity = totalDuration / arma::accu(episodeTurnTimes_arma);  

        double old_credits = graph->getNodeCredits(currNode);
        double new_credits = old_credits + (strategy.getAlpha()*score_episode*activity);
        graph->setNodeCredits(currNode,new_credits);   

        //std::cout << "  Action " << action << ": " << totalDuration << " seconds" << std::endl;
    }
  }
}



double getAca2SessionLikelihood(const RatData& ratdata, int session, Strategy& strategy, bool sim)
{
  arma::mat allpaths = ratdata.getPaths();
  //printFirst5Rows(allpaths,"allpaths");
  std::string strategy_name = strategy.getName();
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
  else if(strategy_name == "aca2_Suboptimal_Hybrid3")
  {
    turnTimes = ratdata.getHybrid3();
  }
  else if(strategy_name == "aca2_Optimal_Hybrid3")
  {
    turnTimes = ratdata.getHybrid3();
  }
  else if(strategy_name == "Hybrid4")
  {
    turnTimes = ratdata.getHybrid4();
  }
    
  //printFirst5Rows(turnTimes,"turnTimes");

  //Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(testModel.slot("nodeGroups"));
  
  double alpha = strategy.getAlpha();
  double gamma = strategy.getGamma();
  std::vector<double> rewardsS0 = strategy.getRewardsS0();
  std::vector<double> rewardsS1 = strategy.getRewardsS1();
  

  int episodeNb = 0; 
  
 
  std::vector<double> mseMatrix;
  //int mseRowIdx = 0;
  
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  
  arma::vec turnTime_method;
  if (sim == 1)
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
  
  BoostGraph& S0 = strategy.getStateS0();
  BoostGraph& S1 = strategy.getStateS1();

  // if(strategy.getOptimal())
  // {
  //   S1 = strategy.getStateS1();
  // }

  int sessId = uniqSessIdx(session);
  //Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
  arma::uvec sessionIdx = arma::find(sessionVec == sessId);
  arma::vec actions_sess = allpath_actions.elem(sessionIdx);
  arma::vec states_sess = allpath_states.elem(sessionIdx);
  arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);
  
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
  double score_episode = 0;
  // float avg_score = 0;
  bool resetVector = true;
  int nrow = actions_sess.n_rows;
   
   //std::cout << "Number of paths in session "<< sessId  << " = " << nrow << std::endl;

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
  std::vector<std::string> episodeTurns;
  std::vector<int> episodeTurnStates;
  std::vector<double> episodeTurnTimes;
  
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
    BoostGraph* graph=nullptr;
    std::vector<double> rewardVec;

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


    double pathProb = 0;
    for (int j = 0; j < nbOfTurns; j++)
    {
      std::string currTurn = turns[j]; 
      currNode = graph->findNode(currTurn);
      int nodeId = graph->getNodeId(currNode);
      score_episode = score_episode + rewardVec[nodeId];
      //currNode->credit = currNode->credit + 1; //Test
      //Rcpp::Rcout <<"currNode="<< currNode->node<<std::endl;
      episodeTurns.push_back(currTurn);
      episodeTurnStates.push_back(S);
      episodeTurnTimes.push_back(turn_times_session(session_turn_count));
      
      //std::cout << "S=" <<S << ", A=" << A << ", i=" << i << ", j=" << j <<  ", currTurn=" << currTurn << ", session_turn_count="  << session_turn_count <<std::endl;


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
      pathProb = pathProb + prob_a;

      //double nodeCredits = graph->getNodeCredits(currNode); // for debug, comment if not debugging
      //std::cout << "S=" <<S << ", A=" << A << ", currTurn=" << currTurn << ", nodeCredits=" << nodeCredits << ", prob_a="<< exp(prob_a) << ", pathProb=" << exp(pathProb) <<std::endl;
      
      session_turn_count++;
      prevNode = currNode;
    }
    if(A != 6)
    {
      mseMatrix.push_back(pathProb);
    }
    else
    {
      mseMatrix.push_back(1);
    }
    
      
    //Check if episode ended
    if (returnToInitState || (i==nrow-1))
    {
      //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
      changeState = false;
      returnToInitState = false;   
      
      episodeNb = episodeNb+1;
      
      acaCreditUpdate(episodeTurns, episodeTurnStates, episodeTurnTimes, score_episode, strategy);
      S0.updateEdgeProbabilitiesSoftmax();
      if(strategy.getOptimal())
      {
        S1.updateEdgeProbabilitiesSoftmax();
      }
      
      score_episode = 0;
      episode = episode + 1;
      resetVector = true;
      episodeTurns.clear();
      episodeTurnStates.clear();
      episodeTurnTimes.clear();
    }
    S = S_prime;
    strategy.updatePathProbMat();
  }
  S0.decayCredits(gamma);
  S0.updateEdgeProbabilitiesSoftmax();
  if(strategy.getOptimal())
  {
    S1.decayCredits(gamma);
    S1.updateEdgeProbabilitiesSoftmax();
  }


  double result = std::accumulate(mseMatrix.begin(), mseMatrix.end(), 0.0);

  return (result);
}



// void updateAca2Rewards(const Rcpp::S4& ratdata, int session, Strategy& strategy, bool sim)
// {
//   arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
//   std::string strategy_name = strategy.getName();
//   arma::mat turnTimes;
  
//   if(strategy_name == "Paths")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
//   }
//   else if(strategy_name == "Turns")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
//   }
//   else if(strategy_name == "Hybrid1")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel1"));
//   }
//   else if(strategy_name == "Hybrid2")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel2"));
//   }
//   else if(strategy_name == "aca2_Suboptimal_Hybrid3")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
//   }
//   else if(strategy_name == "aca2_Optimal_Hybrid3")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
//   }
//   else if(strategy_name == "Hybrid4")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel4"));
//   }

  
//   int episodeNb = 0; 
  
 
//   std::vector<double> mseMatrix;
//   //int mseRowIdx = 0;
  
//   arma::vec allpath_actions = allpaths.col(0);
//   arma::vec allpath_states = allpaths.col(1);
//   arma::vec allpath_rewards = allpaths.col(2);
//   arma::vec sessionVec = allpaths.col(4);
//   arma::vec uniqSessIdx = arma::unique(sessionVec);
  
//   arma::vec turnTime_method;
//   if (sim == 1)
//   {
//     turnTime_method = turnTimes.col(3);
//   }
//   else if (strategy_name == "Paths")
//   {
//     turnTime_method = turnTimes.col(3);
//   }
//   else
//   {
//     turnTime_method = turnTimes.col(5);
//   }
  
//   int episode = 1;
  
//   BoostGraph& S0 = strategy.getStateS0();
//   BoostGraph& S1 = strategy.getStateS1();

//   int sessId = uniqSessIdx(session);
//   //Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
//   arma::uvec sessionIdx = arma::find(sessionVec == sessId);
//   arma::vec actions_sess = allpath_actions.elem(sessionIdx);
//   arma::vec states_sess = allpath_states.elem(sessionIdx);
//   arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);
  
//   arma::uvec turnTimes_idx; 
//   if (strategy_name == "Paths")
//   {
//     turnTimes_idx = arma::find(sessionVec == sessId); ;
//   }
//   else
//   {
//     turnTimes_idx = arma::find(turnTimes.col(4) == sessId); 
//   }
//   arma::vec turn_times_session = turnTime_method.elem(turnTimes_idx);
//   arma::uword session_turn_count = 0;
  
//   int initState = 0;
//   bool changeState = false;
//   bool returnToInitState = false;
//   // float avg_score = 0;
//   bool resetVector = true;
//   int nrow = actions_sess.n_rows;
//   int S;
//   if(sim == 1)
//   {
//     S = states_sess(0); 
//   }
//   else
//   {
//     S = states_sess(0) - 1; 
//   }
//   int A = 0;
//   std::vector<std::string> episodeTurns;
//   std::vector<int> episodeTurnStates;
//   std::vector<double> episodeTurnTimes;
//   std::vector<double> rewardUpdatesS0;
//   std::vector<double> rewardUpdatesS1;
  
//   for (int i = 0; i < nrow; i++)
//   {
    
//     if (resetVector)
//     {
//       initState = S;
//       //Rcpp::Rcout <<"initState="<<initState<<std::endl;
//       resetVector = false;
//     }
    
//     //double pathReward=0.0;
    
//     if (sim == 1)
//     {
//       A = actions_sess(i);
//     }
//     else
//     {
//       A = actions_sess(i) - 1;
//     }

//     double R = rewards_sess(i);
//     if(R > 0)
//     {
//       R = 5;
//     }

//     int S_prime = 0;
//     if(i < (nrow-1))
//     {
//       if (sim == 1)
//       {
//         S_prime = states_sess(i + 1);
//       }
//       else
//       {
//         S_prime = states_sess(i + 1) - 1;
//       }
//     }
    
//     if (S_prime != initState)
//     {
//       changeState = true;
//     }
//     else if (S_prime == initState && changeState)
//     {
//       returnToInitState = true;
//     }
    
//     BoostGraph::Vertex prevNode;
//     BoostGraph::Vertex currNode;
//     BoostGraph::Vertex rootNode;
//     std::vector<double> rewardVec;
//     BoostGraph* graph;
//     std::vector<double> rewardsS0 = strategy.getRewardsS0();
//     std::vector<double> rewardsS1 = strategy.getRewardsS1();

//     if (S == 0 && strategy.getOptimal())
//     {
//       graph = &S0;
//       rootNode = graph->findNode("E");
//       rewardVec = rewardsS0;
//     }else if(S == 1 && strategy.getOptimal())
//     {
//       graph = &S1;
//       rootNode = graph->findNode("I");
//       rewardVec = rewardsS1;
//     }else if(S == 0 && !strategy.getOptimal())
//     {
//       graph = &S0;
//       rootNode = graph->findNode("E");
//       rewardVec = rewardsS0;
//     }else if(S == 1 && !strategy.getOptimal())
//     {
//       graph = &S0;
//       rootNode = graph->findNode("I");
//       rewardVec = rewardsS0;
//     }
//     std::vector<std::string> turns = graph->getTurnsFromPaths(A, S, strategy.getOptimal());
//     int nbOfTurns = turns.size();


//     for (int j = 0; j < nbOfTurns; j++)
//     {
//       std::string currTurn = turns[j]; 
//       currNode = graph->findNode(currTurn);
//       int nodeId = graph->getNodeId(currNode);
//       std::vector<double> crpPosterior = strategy.getCrpPrior();
//       if (j == (nbOfTurns - 1))
//       {
//         rewardVec[nodeId] += strategy.getPhi() * crpPosterior.back() *(R-rewardVec[nodeId]);
//       }
//       else
//       {
//         rewardVec[nodeId] +=  strategy.getPhi() * crpPosterior.back() *(-rewardVec[nodeId]);;
//       }

//       if (S == 0 && strategy.getOptimal())
//       {
//         strategy.setRewardsS0(rewardVec);
//       }else if(S == 1 && strategy.getOptimal())
//       {
//         strategy.setRewardsS1(rewardVec);
//       }else if(S == 0 && !strategy.getOptimal())
//       {
//         strategy.setRewardsS0(rewardVec);
//       }else if(S == 1 && !strategy.getOptimal())
//       {
//         strategy.setRewardsS0(rewardVec);
//       }

//       //currNode->credit = currNode->credit + 1; //Test
//       //Rcpp::Rcout <<"currNode="<< currNode->node<<std::endl;
//       episodeTurns.push_back(currTurn);
//       episodeTurnStates.push_back(S);
//       episodeTurnTimes.push_back(turn_times_session(session_turn_count));
      
//       BoostGraph::Edge edge;
//       if(j==0)
//       {
//         edge = graph->findEdge(rootNode, currNode);
//       }
//       else
//       {
//         edge = graph->findEdge(prevNode, currNode);
//       }

 
//       //Rcpp::Rcout <<"prob_a="<< prob_a << ", pathProb=" <<pathProb <<std::endl;
      
//       session_turn_count++;
//       prevNode = currNode;
//     } 
      
//     //Check if episode ended
//     if (returnToInitState || (i==nrow-1))
//     {
//       //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
//       changeState = false;
//       returnToInitState = false;   
      
//       episodeNb = episodeNb+1;
           
//       episode = episode + 1;
//       resetVector = true;
//       episodeTurns.clear();
//       episodeTurnStates.clear();
//       episodeTurnTimes.clear();
//     }
//     S = S_prime;
//   }

//   std::vector<double> rewardsS0 = strategy.getRewardsS0();
//   std::vector<double> rewardsS1 = strategy.getRewardsS1();
 
//   std::cout << "rewardsS0: ";
//   for (const auto& element : rewardsS0) {
//     std::cout << element << " ";
//   }
//   std::cout << std::endl;

//   std::cout << "rewardsS1: ";
//   for (const auto& element : rewardsS1) {
//     std::cout << element << " ";
//   }
//   std::cout << std::endl;


//   return ;
// }



// void initializeAca2Rewards(const Rcpp::S4& ratdata, int session, Strategy& strategy, bool sim)
// {
//   arma::mat allpaths = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
//   std::string strategy_name = strategy.getName();
//   arma::mat turnTimes;
  
//     if(strategy_name == "Paths")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("allpaths"));
//   }
//   else if(strategy_name == "Turns")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("turnTimes"));
//   }
//   else if(strategy_name == "Hybrid1")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel1"));
//   }
//   else if(strategy_name == "Hybrid2")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel2"));
//   }
//   else if(strategy_name == "aca2_Suboptimal_Hybrid3")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
//   }
//   else if(strategy_name == "aca2_Optimal_Hybrid3")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel3"));
//   }
//   else if(strategy_name == "Hybrid4")
//   {
//     turnTimes = Rcpp::as<arma::mat>(ratdata.slot("hybridModel4"));
//   }

  
//   //Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(testModel.slot("nodeGroups"));
  
//   int episodeNb = 0; 
  
 
//   std::vector<double> mseMatrix;
//   //int mseRowIdx = 0;
  
//   arma::vec allpath_actions = allpaths.col(0);
//   arma::vec allpath_states = allpaths.col(1);
//   arma::vec allpath_rewards = allpaths.col(2);
//   arma::vec sessionVec = allpaths.col(4);
//   arma::vec uniqSessIdx = arma::unique(sessionVec);
  
//   arma::vec turnTime_method;
//   if (sim == 1)
//   {
//     turnTime_method = turnTimes.col(3);
//   }
//   else if (strategy_name == "Paths")
//   {
//     turnTime_method = turnTimes.col(3);
//   }
//   else
//   {
//     turnTime_method = turnTimes.col(5);
//   }
  
//   int episode = 1;
  
//   BoostGraph& S0 = strategy.getStateS0();
//   BoostGraph& S1 = strategy.getStateS1();

//   int sessId = sessionVec(session);
//   //Rcpp::Rcout <<"sessId="<<sessId<<std::endl;
//   arma::uvec sessionIdx = arma::find(sessionVec == sessId);
//   arma::vec actions_sess = allpath_actions.elem(sessionIdx);
//   arma::vec states_sess = allpath_states.elem(sessionIdx);
//   arma::vec rewards_sess = allpath_rewards.elem(sessionIdx);
  
//   arma::uvec turnTimes_idx; 
//   if (strategy_name == "Paths")
//   {
//     turnTimes_idx = arma::find(sessionVec == sessId); ;
//   }
//   else
//   {
//     turnTimes_idx = arma::find(turnTimes.col(4) == sessId); 
//   }
//   arma::vec turn_times_session = turnTime_method.elem(turnTimes_idx);
//   arma::uword session_turn_count = 0;
  
//   int initState = 0;
//   bool changeState = false;
//   bool returnToInitState = false;
//   // float avg_score = 0;
//   bool resetVector = true;
//   int nrow = actions_sess.n_rows;
//   int S;
//   if(sim == 1)
//   {
//     S = states_sess(0); 
//   }
//   else
//   {
//     S = states_sess(0) - 1; 
//   }
//   int A = 0;
//   std::vector<std::string> episodeTurns;
//   std::vector<int> episodeTurnStates;
//   std::vector<double> episodeTurnTimes;
//   std::vector<double> rewardUpdatesS0;
//   std::vector<double> rewardUpdatesS1;
  
//   for (int i = 0; i < nrow; i++)
//   {
    
//     if (resetVector)
//     {
//       initState = S;
//       //Rcpp::Rcout <<"initState="<<initState<<std::endl;
//       resetVector = false;
//     }
    
//     //double pathReward=0.0;
    
//     if (sim == 1)
//     {
//       A = actions_sess(i);
//     }
//     else
//     {
//       A = actions_sess(i) - 1;
//     }

//     double R = rewards_sess(i);
//     if(R > 0)
//     {
//       R = 5;
//     }

//     int S_prime = 0;
//     if(i < (nrow-1))
//     {
//       if (sim == 1)
//       {
//         S_prime = states_sess(i + 1);
//       }
//       else
//       {
//         S_prime = states_sess(i + 1) - 1;
//       }
//     }
    
//     if (S_prime != initState)
//     {
//       changeState = true;
//     }
//     else if (S_prime == initState && changeState)
//     {
//       returnToInitState = true;
//     }
    
//     BoostGraph::Vertex prevNode;
//     BoostGraph::Vertex currNode;
//     BoostGraph::Vertex rootNode;
//     std::vector<double> rewardVec;
//     BoostGraph* graph;
//     std::vector<double> rewardsS0 = strategy.getRewardsS0();
//     std::vector<double> rewardsS1 = strategy.getRewardsS1();


//     if (S == 0 && strategy.getOptimal())
//     {
//       graph = &S0;
//       rootNode = graph->findNode("E");
//       rewardVec = rewardsS0;
//     }else if(S == 1 && strategy.getOptimal())
//     {
//       graph = &S1;
//       rootNode = graph->findNode("I");
//       rewardVec = rewardsS1;
//     }else if(S == 0 && !strategy.getOptimal())
//     {
//       graph = &S0;
//       rootNode = graph->findNode("E");
//       rewardVec = rewardsS0;
//     }else if(S == 1 && !strategy.getOptimal())
//     {
//       graph = &S0;
//       rootNode = graph->findNode("I");
//       rewardVec = rewardsS0;
//     }
//     std::vector<std::string> turns = graph->getTurnsFromPaths(A, S, strategy.getOptimal());
//     int nbOfTurns = turns.size();


//     for (int j = 0; j < nbOfTurns; j++)
//     {
//       std::string currTurn = turns[j]; 
//       currNode = graph->findNode(currTurn);
//       int nodeId = graph->getNodeId(currNode);
//       std::vector<double> crpPosterior = strategy.getCrpPrior();
//       if (j == (nbOfTurns - 1))
//       {
//         rewardVec[nodeId] += strategy.getPhi() * (R-rewardVec[nodeId]);
//       }
//       else
//       {
//         rewardVec[nodeId] +=  strategy.getPhi() * (-rewardVec[nodeId]);;
//       }

//       //std::cout <<"currTurn="<< currTurn << ", R=" <<R << ", rewardVec[nodeId]=" <<rewardVec[nodeId] <<std::endl;

//       if (S == 0 && strategy.getOptimal())
//       {
//         strategy.setRewardsS0(rewardVec);
//       }else if(S == 1 && strategy.getOptimal())
//       {
//         strategy.setRewardsS1(rewardVec);
//       }else if(S == 0 && !strategy.getOptimal())
//       {
//         strategy.setRewardsS0(rewardVec);
//       }else if(S == 1 && !strategy.getOptimal())
//       {
//         strategy.setRewardsS0(rewardVec);
//       }

//       //currNode->credit = currNode->credit + 1; //Test
//       //Rcpp::Rcout <<"currNode="<< currNode->node<<std::endl;
//       episodeTurns.push_back(currTurn);
//       episodeTurnStates.push_back(S);
//       episodeTurnTimes.push_back(turn_times_session(session_turn_count));
      
//       BoostGraph::Edge edge;
//       if(j==0)
//       {
//         edge = graph->findEdge(rootNode, currNode);
//       }
//       else
//       {
//         edge = graph->findEdge(prevNode, currNode);
//       }

 
//       //Rcpp::Rcout <<"prob_a="<< prob_a << ", pathProb=" <<pathProb <<std::endl;
      
//       session_turn_count++;
//       prevNode = currNode;
//     } 
      
//     //Check if episode ended
//     if (returnToInitState || (i==nrow-1))
//     {
//       //Rcpp::Rcout <<  "Inside end episode"<<std::endl;
//       changeState = false;
//       returnToInitState = false;   
      
//       episodeNb = episodeNb+1;
           
//       episode = episode + 1;
//       resetVector = true;
//       episodeTurns.clear();
//       episodeTurnStates.clear();
//       episodeTurnTimes.clear();
//     }
//     S = S_prime;
//   }

//   std::vector<double> rewardsS0 = strategy.getRewardsS0();
//   std::vector<double> rewardsS1 = strategy.getRewardsS1();
 
//   std::cout << "rewardsS0: ";
//   for (const auto& element : rewardsS0) {
//     std::cout << element << " ";
//   }
//   std::cout << std::endl;

//   std::cout << "rewardsS1: ";
//   for (const auto& element : rewardsS1) {
//     std::cout << element << " ";
//   }
//   std::cout << std::endl;



//   return ;
// }
