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



double getAca2SessionLikelihood(const RatData& ratdata, int session, Strategy& strategy)
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

  bool sim = ratdata.getSim();
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
  // std::cout << "Inside getAca2SessionLikelihood" <<", strategy=" << strategy.getName() <<", session=" << session << ", uniqSessIdx.size=" << uniqSessIdx.size() << std::endl;
  
  // std::cout << "uniqSessIdx:\n";
  // for (arma::uword i = 0; i < uniqSessIdx.n_elem; ++i) {
  //     std::cout << uniqSessIdx(i) << " ";
  // }
  // std::cout << std::endl;

  int uniqSessIdx_size = uniqSessIdx.size();
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
      
      //std::cout << "S=" <<S << ", A=" << A << ", ses="<< session << ", i=" << i << ", j=" << j << ", currTurn=" << currTurn << ", currTurnDuration=" << turn_times_session(session_turn_count) << ", nodeCredits=" << graph->getNodeCredits(currNode)  <<std::endl;


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
      //std::cout << "S=" <<S << ", A=" << A << ", i=" << i << ", j=" << j << ", currTurn=" << currTurn << ", nodeCredits=" << nodeCredits << ", prob_a="<< exp(prob_a) << ", pathProb=" << exp(pathProb) <<std::endl;
      
      session_turn_count++;
      prevNode = currNode;
    }

    //std::cout << "S=" << S << ", A=" << A << ", i=" << i << ", pathProb=" << pathProb <<std::endl;

    if(A != 6)
    {
      mseMatrix.push_back(pathProb);
    }
    else
    {
      mseMatrix.push_back(0);
    }
    
      
    //Check if episode ended
    if (returnToInitState || (i==nrow-1))
    {
      //std::cout <<  "Inside end episode"<<std::endl;
      changeState = false;
      returnToInitState = false;   
      
      episodeNb = episodeNb+1;
      
      acaCreditUpdate(episodeTurns, episodeTurnStates, episodeTurnTimes, score_episode, strategy);
      S0.updateEdgeProbabilitiesSoftmax();
      if(strategy.getOptimal())
      {
        S1.updateEdgeProbabilitiesSoftmax();
      }

      // std::cout << "S0 credits:";
      // S0.printNodeCredits();
      // std::cout << "S1 credits:";
      // S1.printNodeCredits();
      
      score_episode = 0;
      episode = episode + 1;
      resetVector = true;
      episodeTurns.clear();
      episodeTurnStates.clear();
      episodeTurnTimes.clear();
    }
    S = S_prime;
    strategy.updatePathProbMat(session);
  }
  //double decay_factor = 1-(gamma/std::pow(session+1, 0.5));
  double decay_factor = gamma;
  S0.decayCredits(decay_factor);
  S0.updateEdgeProbabilitiesSoftmax();
  if(strategy.getOptimal())
  {
    S1.decayCredits(decay_factor);
    S1.updateEdgeProbabilitiesSoftmax();
  }

//  if(!strategy.getOptimal())
//  {
//     std::cout << "ses=" << session << ", likelihood = ";
//     for (auto const& i : mseMatrix)
//       std::cout << exp(i) << ", ";
//     std::cout << "\n" ;

//  }

  double result = std::accumulate(mseMatrix.begin(), mseMatrix.end(), 0.0);

  //std::cout << "strategy=" << strategy.getName() << ", alpha=" <<strategy.getAlpha() << ", gamma=" << strategy.getGamma() << ", lambda=" << strategy.getLambda() << ", ses=" << session << ", loglikelihood=" << result << std::endl;


  return (result);
}


//Not Turn models
std::pair<arma::mat, arma::mat> simulateAca2(const RatData& ratdata, int session, Strategy& strategy)
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


  //std::vector<std::string> nodeListS0 = {"E","dc1","fga1","dc1.c2h","c2ba1","a2bc1","a2bc1.c2h","fga1.a2kj","c2ba1.a2kj","a2gf","c2d"};
  //std::vector<std::string> nodeListS1 = {"I","hc1","jka1","hc1.c2d","c2ba1","a2bc1","c2h","a2kj","c2ba1.a2gf","jka1.a2gf","a2bc1.c2d"};

    
  //printFirst5Rows(turnTimes,"turnTimes");

  //Rcpp::List nodeGroups = Rcpp::as<Rcpp::List>(testModel.slot("nodeGroups"));
  
  double alpha = strategy.getAlpha();
  double gamma = strategy.getGamma();

  //std::cout << strategy.getName() << ", session=" << session << ", alpha=" <<alpha << ", gamma=" <<gamma << std::endl;

  std::vector<double> rewardsS0; 
  std::vector<double> rewardsS1; 

  if(strategy.getOptimal())
    {
        rewardsS0 = {0,0,0,0,0,0,0,5,0};
        rewardsS1 = {0,0,0,0,0,0,0,0,5};
    }else{
        rewardsS0 = {0,0,0,0,0,0,5,0,0,0,0,0};
    }

  

  int episodeNb = 0; 
 
 
  std::vector<double> mseMatrix;
  //int mseRowIdx = 0;
  
  arma::vec allpath_actions = allpaths.col(0);
  arma::vec allpath_states = allpaths.col(1);
  arma::vec allpath_rewards = allpaths.col(2);
  arma::vec sessionVec = allpaths.col(4);

  // std::cout << "sessionVec Elements:\n";
  // for (arma::uword i = 0; i < sessionVec.n_elem; ++i) {
  //     std::cout << sessionVec(i) << " ";
  // }

  arma::vec uniqSessIdx = arma::unique(sessionVec);

  //std::cout << "strategy=" << strategy.getName() <<", session=" << session << ", uniqSessIdx.size=" << uniqSessIdx.size() << std::endl;

  
  arma::vec turnTime_method;
  
  if (strategy_name == "Paths")
  {
    turnTime_method = turnTimes.col(3);
  }
  else
  {
    turnTime_method = turnTimes.col(5);
  }
  
  int episode = 1;

  arma::mat R = arma::zeros(2, 6);
  R(0, 3) = 5;
  R(1, 3) = 5;

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
  int totalPaths = allpath_actions.n_rows;

  arma::mat generated_PathData_sess(nrow, 7);
  arma::mat generated_TurnsData_sess((nrow * 3), 7);
  generated_PathData_sess.fill(-1);
  generated_TurnsData_sess.fill(-1);
  unsigned int turnIdx = 0; // counter for turn model
  int actionNb = 0;
   
   //std::cout << "Number of paths in session "<< sessId  << " = " << nrow << std::endl;

  int S = states_sess(0) - 1; // start from yhe same state as the session
  int A = -1;
  std::vector<std::string> episodeTurns;
  std::vector<int> episodeTurnStates;
  std::vector<double> episodeTurnTimes;
  
  for (int i = 0; i < nrow; i++)
  {
    actionNb++;
    double pathDuration = 0;

    if (resetVector)
    {
      initState = S;
      //Rcpp::Rcout <<"initState="<<initState<<std::endl;
      resetVector = false;
    }
    
    //double pathReward=0.0;
    
    
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
    // std::vector<std::string> turns = graph->getTurnsFromPaths(A, S, strategy.getOptimal());
    
    // generate a trajectory (Path)
    std::vector<std::string> hybridTurns = generatePathTrajectory(strategy, graph, rootNode);
    int A = graph->getPathFromTurns(hybridTurns, rootNode, strategy.getOptimal());
    int nbOfTurns = hybridTurns.size();
    int S_prime = getNextState(S,A);
        
    if (S_prime != initState)
    {
      changeState = true;
    }
    else if (S_prime == initState && changeState)
    {
      returnToInitState = true;
    }

    //std::cout << "Generated action A=" << A << ", nbOfTurns=" << nbOfTurns << ", i=" << i  << std::endl;
      
      // Based on the TurnModel durations, the durations of the testModel components 
      // are determined.
      
    for(int k=0; k < hybridTurns.size(); k++ )
    {
      std::string node = hybridTurns[k];
      BoostGraph::Vertex v = graph->findNode(node);
      double nodeCredits = graph->getNodeCredits(v);

      int hybridNodeId = graph->getNodeId(v);

      score_episode = score_episode + rewardVec[hybridNodeId];

      double hybridNodeDuration = 0;
      hybridNodeDuration = simulateTurnDuration(turnTimes, hybridNodeId, S, session, strategy);
      pathDuration = pathDuration + hybridNodeDuration;

      //std::cout << "Generated action A=" << A << ", hybridNodeDuration=" << hybridNodeDuration << std::endl;

      generated_TurnsData_sess(turnIdx, 0) = hybridNodeId;
      generated_TurnsData_sess(turnIdx, 1) = S;
      generated_TurnsData_sess(turnIdx, 2) = 0;
      generated_TurnsData_sess(turnIdx, 3) = hybridNodeDuration;
      //Rcpp::Rcout << "Turn=" << turnName1 <<", turnDuration="<< turnTime<<std::endl;
      generated_TurnsData_sess(turnIdx, 4) = sessId;
      generated_TurnsData_sess(turnIdx, 5) = actionNb;
      generated_TurnsData_sess(turnIdx, 6) = 0;

      // std::cout << "S=" <<S << ", A=" << A << ", ses="<< session << ", i=" << i << ", k=" << k << ", currTurn=" << node << ", currTurnDuration=" << hybridNodeDuration << ", nodeCredits=" << nodeCredits  <<std::endl;
      
      episodeTurns.push_back(node);
      episodeTurnStates.push_back(S);
      episodeTurnTimes.push_back(hybridNodeDuration);
      turnIdx++;
    }    
    
    //arma::mat durationMat = simulatePathTime(turnTimes, allpaths, actionNb, A, pathStages,nodeGroups);
    
    //std::cout <<"A=" << A << ", S=" << S << ", sessId=" <<sessId<< std::endl;
    generated_PathData_sess(i, 0) = A;
    generated_PathData_sess(i, 1) = S;
    //Rcpp::Rcout <<"R(S, A)=" <<R(S, A)<< std::endl;
    generated_PathData_sess(i, 2) = R(S, A);
    generated_PathData_sess(i, 3) = pathDuration;
    generated_PathData_sess(i, 4) = sessId;
    generated_PathData_sess(i, 5) = actionNb;
    
    
    if (R(S, A) > 0)
    {
      //std::cout << "turnNb=" << generated_TurnsData_sess((turnIdx - 1), 0) << ", receives reward"<< std::endl;
      generated_TurnsData_sess((turnIdx - 1), 2) = 5;
      //score_episode = score_episode + 5;
    }

    //std::cout << "S=" << S << ", A=" << A << ", i=" << i << ", pathProb=" << pathProb <<std::endl;

  
    //Check if episode ended
    if (returnToInitState || (i==nrow-1))
    {
      // std::cout <<  "Episode end, score_episode=" << score_episode <<std::endl;
      changeState = false;
      returnToInitState = false;   
      
      episodeNb = episodeNb+1;
      
      acaCreditUpdate(episodeTurns, episodeTurnStates, episodeTurnTimes, score_episode, strategy);
      S0.updateEdgeProbabilitiesSoftmax();
      if(strategy.getOptimal())
      {
        S1.updateEdgeProbabilitiesSoftmax();
      }

      // std::cout << "S0 credits:";
      // S0.printNodeCredits();
      // std::cout << "S1 credits:";
      // S1.printNodeCredits();

      // std::cout << "S0 probs:";
      // S0.printNodeProbabilities();
      // std::cout << "S1 probs:";
      // S1.printNodeProbabilities();


      
      score_episode = 0;
      episode = episode + 1;
      resetVector = true;
      episodeTurns.clear();
      episodeTurnStates.clear();
      episodeTurnTimes.clear();
    }
    S = S_prime;
    strategy.updatePathProbMat(session);
  }
  
  //double decay_factor = 1-(gamma/std::pow(session+1, 0.5));
  double decay_factor = gamma;
  S0.decayCredits(decay_factor);
  S0.updateEdgeProbabilitiesSoftmax();
  if(strategy.getOptimal())
  {
    S1.decayCredits(decay_factor);
    S1.updateEdgeProbabilitiesSoftmax();
  }

  
  return std::make_pair(generated_PathData_sess, generated_TurnsData_sess);

}





