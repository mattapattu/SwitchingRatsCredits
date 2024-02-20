// Particle filter algorithm for expert assignment problem
// Assume that there are N particles, M experts, and T trials
// Each particle m is a vector of length T, where m[t] is the index of the expert assigned to trial t
// Each expert i has a probability distribution p_i over the possible outcomes of each trial
// The likelihood of a particle m given the observed outcomes y is the product of p_mt for all t
// The posterior probability of a particle m is proportional to its prior probability times its likelihood
// The prior probability of a particle m is assumed to be uniform over all possible assignments
// The resampling step is done using multinomial resampling with replacement

#include "ParticleFilter.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
using namespace std;

// A function that returns a random integer in the range [0, n-1]
int randint(int n) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, n-1);
  return dis(gen);
}

// A function that returns a random sample from a discrete probability distribution
int sample(vector<double> p) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dis(p.begin(), p.end());
  return dis(gen);
}

// A function that normalizes a vector of probabilities to sum to one
void normalizeRows(std::vector<std::vector<double>>& matrix) {
    for (auto& row : matrix) {
        // Calculate the sum of elements in the current row
        double rowSum = 0.0;
        for (double element : row) {
            rowSum += element;
        }

        // Normalize the elements in the current row
        for (double& element : row) {
            element /= rowSum;
        }
    }
}

// A function that normalizes a vector of probabilities to sum to one
void normalize(vector<double>& p) {
  double sum = 0;
  for (double x : p) {
    sum += x;
  }
  for (double& x : p) {
    x /= sum;
  }
  if (sum == 0) {
                    
        throw std::runtime_error("Error weight vec is zero");
    }
}

std::vector<double> stratifiedResampling(const std::vector<double>& particleWeights) {
    std::vector<double> resampledParticles;
    int numParticles = particleWeights.size();

    // Compute cumulative weights
    std::vector<double> cumulativeWeights(numParticles);
    double totalWeight = 0.0;
    for (int i = 0; i < numParticles; ++i) {
        totalWeight += particleWeights[i];
        cumulativeWeights[i] = totalWeight;
    }

    // Generate random numbers in each subinterval
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniformDist(0.0, 1.0);

    for (int i = 0; i < numParticles; ++i) {
        double randomValue = (i + uniformDist(gen)) / numParticles;
        // Find the particle corresponding to the random value
        for (int j = 0; j < numParticles; ++j) {
            if (randomValue <= cumulativeWeights[j]) {
                resampledParticles.push_back(j);
                break;
            }
        }
    }

    return resampledParticles;
}

std::pair<std::vector<std::vector<double>>, double> particle_filter(int N, const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3,  vector<double> v) {

  
  arma::mat allpaths = ratdata.getPaths();
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  int sessions = uniqSessIdx.n_elem;

  // Initialize the particles with random assignments
  //vector<vector<int>> m(N, vector<int>(T));
  std::vector<ParticleFilter>  particleFilterVec;
  for (int i = 0; i < N; i++) {
   auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, v, i, 1.0);
   particleFilterVec.push_back(pf);
   //std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;

  }

  vector<double> w(N,1.0/N);

  std::vector<std::vector<double>> filteredWeights;
  std::vector<std::vector<double>> crpPriors;

  double loglik = 0;
  // Iterate over the trials
  for (int ses = 0; ses < sessions; ses++) {
    //   // Initialize the weights with uniform probabilities
    double ses_lik = 0;
    // Update the weights with the likelihood of the current outcome
    for (int i = 0; i < N; i++) {
        vector<double> crp_i = particleFilterVec[i].crpPrior(ses);
        crpPriors.push_back(crp_i);
        // std::vector<int> particleAssignments = particleFilterVec[i].getChosenStratgies();
        // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << " , assignments=[";

        // for(int j = 0; j < ses; j++)
        // {
        //     std::cout << particleAssignments[j] << ", ";
        // }
        // std::cout << "]\n" ;
        
        // std::cout << ", crp_priors = ";
        // for (auto const& i : crp_i)
        //     std::cout << i << ", ";
        // std::cout << "\n" ;


        int sampled_strat = sample(crp_i); 
        particleFilterVec[i].addAssignment(ses,sampled_strat);
        double lik = particleFilterVec[i].getSesLikelihood(sampled_strat,ses);
        w[i] *= crp_i[sampled_strat]*lik;
        ses_lik = ses_lik + lik;
        //std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", sampled_strat=" << sampled_strat << ", lik=" << lik << ", w[i]=" << w[i] << std::endl;

    }

    ses_lik = ses_lik/N;

    loglik = loglik + log(ses_lik);

    // Normalize the weights
    normalize(w);
    
    // std::cout << "ses=" << ses << ", normalied w=";
    // for (auto const& i : w)
    //     std::cout << i << ", ";
    // std::cout << "\n" ;

    // Resample the particles with replacement according to the weights
    // vector<vector<int>> m_new(N, vector<int>(T));
    // for (int i = 0; i < N; i++) {
    //   int j = sample(w);
    //   //particleFilterVec[i] = std::make_shared<ParticleFilter>(particleFilterVec[j]);
    //   ParticleFilter pf(particleFilterVec[j]);
    //   int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
    //   particleFilterVec[i].setChosenStrategies(particleFilterVec[j].getChosenStratgies());
    //   particleFilterVec[i].setStrategies(particleFilterVec[j].getStrategies());
    //   int updatedChosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
    //   std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[j].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;

    // }

    double weightSq = 0;
    for(int k=0; k<N; k++)
    {
        weightSq = weightSq + std::pow(w[k], 2);
    }
    double n_eff = 1/weightSq;
    if(n_eff < N/2)
    {
        //std::cout << "ses=" <<ses <<", n_eff=" << n_eff << ", performing resampling" << std::endl;
        std::vector<double>resampledIndices =  stratifiedResampling(w);

        for (int i = 0; i < N; i++) {
            int newIndex = resampledIndices[i];
            ParticleFilter pf(particleFilterVec[newIndex]);
            int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
            particleFilterVec[i].setChosenStrategies(particleFilterVec[newIndex].getChosenStratgies());
            particleFilterVec[i].setStrategies(particleFilterVec[newIndex].getStrategies());
            int updatedChosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
            // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[newIndex].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;

        }

        std::fill(w.begin(), w.end(), 1.0/N);

    }

    filteredWeights.push_back(w);

    
  }

  // Compute the posterior probabilities for each expert by counting the occurrences of each expert in the last trial
  vector<vector<double>> postProbsOfExperts(sessions, vector<double>(4) );
  for(int ses=0; ses<sessions; ses++)
  {
    for (int i = 0; i < N; i++) {
        std::vector<int> chosenStrategy_pf = particleFilterVec[i].getChosenStratgies();
        // std::cout << "ses=" <<ses << ", particleId=" <<i << ", chosenStrat=" << chosenStrategy_pf[ses] << std::endl;
        postProbsOfExperts[ses][chosenStrategy_pf[ses]]++;
    }
  }

//   std::cout << "posterior probs=" <<  std::endl;
//     for (const auto& row : postProbsOfExperts) {
//         for (double num : row) {
//             std::cout << num << " ";
//         }
//         std::cout << std::endl;
//     }

  

  // Normalize the posterior probabilities
  //normalizeRows(q);

    std::cout << "posterior probs=" <<  std::endl;
    for (const auto& row : postProbsOfExperts) {
        for (double num : row) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
    }
 
  // Return the posterior probabilities
  return std::make_pair(postProbsOfExperts, loglik);
}




// A function that implements the particle filter algorithm
// Input: the number of particles N, the number of experts M, the number of trials T, the expert distributions p, and the observed outcomes y
// Output: a vector of posterior probabilities for each expert
std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> particle_filter_new(int N, std::vector<ParticleFilter>&  particleFilterVec, const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3) {

  
  arma::mat allpaths = ratdata.getPaths();
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  int sessions = uniqSessIdx.n_elem;

  // Initialize the particles with random assignments
  //vector<vector<int>> m(N, vector<int>(T));
  

  vector<double> w(N,1.0/N);

  // w[t][i]
  std::vector<std::vector<double>> filteredWeights;
  std::vector<std::vector<double>> crpPriors;

  double loglik = 0;
  // Iterate over the trials
  for (int ses = 0; ses < sessions; ses++) {
    //   // Initialize the weights with uniform probabilities
    double ses_lik = 0;
    // Update the weights with the likelihood of the current outcome
    for (int i = 0; i < N; i++) {
        vector<double> crp_i = particleFilterVec[i].crpPrior(ses);
        particleFilterVec[i].setCrpPriors(crp_i);
        
        int sampled_strat = sample(crp_i); 
        particleFilterVec[i].addAssignment(ses,sampled_strat);
        double lik = particleFilterVec[i].getSesLikelihood(sampled_strat,ses);
        particleFilterVec[i].setLikelihood(lik);
        w[i] *= crp_i[sampled_strat]*lik;
        ses_lik = ses_lik + lik;
        //std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", sampled_strat=" << sampled_strat << ", lik=" << lik << ", w[i]=" << w[i] << std::endl;

    }

    ses_lik = ses_lik/N;

    loglik = loglik + log(ses_lik);

    // Normalize the weights
    normalize(w);
    
    double weightSq = 0;
    for(int k=0; k<N; k++)
    {
        weightSq = weightSq + std::pow(w[k], 2);
    }
    double n_eff = 1/weightSq;
    if(n_eff < N/2)
    {
        //std::cout << "ses=" <<ses <<", n_eff=" << n_eff << ", performing resampling" << std::endl;
        std::vector<double>resampledIndices =  stratifiedResampling(w);

        for (int i = 0; i < N; i++) {
            int newIndex = resampledIndices[i];
            ParticleFilter pf(particleFilterVec[newIndex]);
            int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
            particleFilterVec[i].setChosenStrategies(particleFilterVec[newIndex].getChosenStratgies());
            particleFilterVec[i].setStrategies(particleFilterVec[newIndex].getStrategies());
            int updatedChosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
            // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[newIndex].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;

        }

        std::fill(w.begin(), w.end(), 1.0/N);

    }

    filteredWeights.push_back(w);
    
  }

  // Compute the posterior probabilities for each expert by counting the occurrences of each expert in the last trial
  vector<vector<double>> postProbsOfExperts(sessions, vector<double>(4) );
  for(int ses=0; ses<sessions; ses++)
  {
    for (int i = 0; i < N; i++) {
        std::vector<int> chosenStrategy_pf = particleFilterVec[i].getChosenStratgies();
        // std::cout << "ses=" <<ses << ", particleId=" <<i << ", chosenStrat=" << chosenStrategy_pf[ses] << std::endl;
        postProbsOfExperts[ses][chosenStrategy_pf[ses]]++;
    }
  }

  std::cout << "postProbsOfExperts=" <<  std::endl;
    for (const auto& row : postProbsOfExperts) {
        for (double num : row) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
    }

  

  // Normalize the posterior probabilities
  //normalizeRows(q);

    std::cout << "normalized q=" <<  std::endl;
    for (const auto& row : postProbsOfExperts) {
        for (double num : row) {
            std::cout << num << " ";
        }
        std::cout << std::endl;
    }



  
  // Return the posterior probabilities
  return std::make_tuple(filteredWeights, postProbsOfExperts, crpPriors);
}

// A function that prints a vector of doubles
void print_vector(vector<double> v) {
  cout << "[";
  for (int i = 0; i < v.size(); i++) {
    cout << v[i];
    if (i < v.size() - 1) {
      cout << ", ";
    }
  }
  cout << "]" << endl;
}

std::vector<std::vector<double>> w_smoothed(std::vector<std::vector<double>> filteredWeights, std::vector<ParticleFilter>  particleFilterVec,  int N)
{
    //w_smoothed[t][i]
    std::vector<std::vector<double>> smoothedWeights;
    int n_rows = filteredWeights.size();
    for(int t = n_rows-1; t >=0; t--)
    {
        if(t==n_rows-1)
        {
            for(int i=0; i<N ; i++)
            {
                smoothedWeights[t][i] = filteredWeights[n_rows-1][i];
            }
            
        }else{
            
            //v_t = \sum_j (w[t][j]*crpPriors_t)
            

            for(int i=0; i<N ; i++)
            {
                std::vector<std::vector<double>> crpPriors_i = particleFilterVec[i].getCrpPriors();
                std::vector<double> crpPriors_i_t =  crpPriors_i[t];

                double weightedSum = 0;
                for(int k=0; k<N;k++)
                {
                    std::vector<int> assignments_k = particleFilterVec[k].getChosenStratgies();
                    std::vector<int> assignments_i = particleFilterVec[i].getChosenStratgies();
                    int X_k_tplus1 = assignments_k[t+1];
                    int X_i_t = assignments_i[t];

                    double v_t_k = 0;
                    for(int j=0; j<N; j++)
                    {
                        std::vector<std::vector<double>> crpPriors_j = particleFilterVec[j].getCrpPriors();
                        std::vector<double> crpPriors_j_t =  crpPriors_j[t];

                        v_t_k = v_t_k + (filteredWeights[t][j]*crpPriors_j_t[X_k_tplus1]);
                    }

                    weightedSum = weightedSum + (smoothedWeights[t+1][k] * crpPriors_i_t[X_k_tplus1]/v_t_k);
                }
                smoothedWeights[t][i] = filteredWeights[t][i];
            }
        }
        
    }

    return(smoothedWeights);
}

std::vector<std::vector<std::vector<double>>> wij_smoothed(std::vector<std::vector<double>> filteredWeights, std::vector<ParticleFilter>  particleFilterVec, std::vector<std::vector<double>> w_smoothed, int N)
{
    std::vector<std::vector<std::vector<double>>> wijSmoothed;
    int n_rows = filteredWeights.size();


    for(int t=0; t< n_rows-2;t++)
    {
        std::vector<std::vector<double>> wij_smoothed_t(N, std::vector<double>(N, 0.0));
        for(int i=0; i< N; i++)
        {
            for(int j=0; j<N;j++)
            {
                
                
                double w_t_i = filteredWeights[t][i];
                double w_smoothed_tplus1_j = w_smoothed[t+1][j];

                std::vector<std::vector<double>> crpPriors_i = particleFilterVec[i].getCrpPriors();
                std::vector<double> crpPriors_i_t = crpPriors_i[t];
                std::vector<int> assignments_j = particleFilterVec[j].getChosenStratgies();
                int x_j_tplus1 = assignments_j[t+1];
                
                double denom=0;
                for(int l=0; l<N; l++)
                {
                    double w_t_l = filteredWeights[t][l];
                    std::vector<std::vector<double>> crpPriors_l = particleFilterVec[l].getCrpPriors();
                    std::vector<double> crpPriors_l_t = crpPriors_l[t];
                    denom = denom + crpPriors_l_t[x_j_tplus1] * w_t_l;

                }
                wij_smoothed_t[i][j] = w_t_i*w_smoothed_tplus1_j*crpPriors_i_t[x_j_tplus1]/denom;
            }

        }
        wijSmoothed.push_back(wij_smoothed_t);
        
    }

    //wij_smoothed[t][i]
    
    return(wijSmoothed);
}

std::pair<std::vector<std::vector<double>>,std::vector<std::vector<std::vector<double>>>>  E_step(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, int N, std::vector<double> params)
{
    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    int n_rows = allpaths.n_rows;



    std::vector<ParticleFilter>  particleFilterVec;
    for (int i = 0; i < N; i++) {
        auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, i, 1.0);
        particleFilterVec.push_back(pf);
        //std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;

    }

    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> resTuple = particle_filter_new(N,particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3);
    std::vector<std::vector<double>> filteredWeights = std::get<0>(resTuple);


    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> resTuple = particle_filter_new(N,particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3);
    std::vector<std::vector<double>> filteredWeights = std::get<0>(resTuple);
    std::vector<std::vector<double>> smoothedWeights = w_smoothed(filteredWeights, particleFilterVec,N);
    
    //wijSmoothed[t][i,j]
    std::vector<std::vector<std::vector<double>>> wijSmoothed = wij_smoothed(filteredWeights, particleFilterVec,smoothedWeights, N);


    return std::make_pair(smoothedWeights, wijSmoothed);

}



double M_step(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, int N, std::pair<std::vector<std::vector<double>>,std::vector<std::vector<std::vector<double>>>> smoothed_w, std::vector<double> params)
{
    
    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    int n_rows = allpaths.n_rows;


    std::vector<ParticleFilter>  particleFilterVec;
    for (int i = 0; i < N; i++) {
        auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, i, 1.0);
        particleFilterVec.push_back(pf);
        //std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;

    }

    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<double>>, std::vector<std::vector<double>>> resTuple = particle_filter_new(N,particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3);
    std::vector<std::vector<double>> filteredWeights = std::get<0>(resTuple);

    std::vector<std::vector<double>> smoothedWeights = smoothed_w.first;
    std::vector<std::vector<std::vector<double>>> wijSmoothed = smoothed_w.second;

    double I_1=0;
    double I_2=0;
    double I_3=0;

    for(int i=0; i<N; i++)
    {
        std::vector<std::vector<double>> crpPriors_i = particleFilterVec[i].getCrpPriors();
        std::vector<double> crpPriors_i_0 = crpPriors_i[0];
        std::vector<int> assignments_i = particleFilterVec[i].getChosenStratgies();
        int X_i_0 = assignments_i[0];

        I_1 = I_1+(smoothedWeights[0][i]*log(crpPriors_i_0[X_i_0]));
    }

    for(int t=0; t<n_rows-1; t++)
    {
        for(int i=0; i<N; i++)
        {
            std::vector<double> liks_i = particleFilterVec[i].getLikelihoods();
            I_3 = I_3+(smoothedWeights[t][i]*log(liks_i[t]));
        }
    }

    for(int t=0; t<n_rows-2; t++)
    {
        for(int i=0; i<N; i++)
        {
            for(int j=0; j<N; j++)
            {
                std::vector<std::vector<double>> crpPriors_i = particleFilterVec[i].getCrpPriors();
                std::vector<double> crpPriors_i_0 = crpPriors_i[0];
                std::vector<int> assignments_j = particleFilterVec[j].getChosenStratgies();
                int X_j_tplus1 = assignments_j[t+1];

                double w_ij_smoothed = wijSmoothed[t][i][j];
                I_2 = I_2 + (log(crpPriors_i_0[X_j_tplus1])*w_ij_smoothed);
            }            
        }
    }

    return(I_1+I_2+I_3);
}

void EM(const RatData& ratdata, const MazeGraph& Suboptimal_Hybrid3, const MazeGraph& Optimal_Hybrid3, int N, std::pair<std::vector<std::vector<double>>,std::vector<std::vector<std::vector<double>>>> smoothed_w)
{
    std::vector<double> params;
    
    for(int i=0; i< 10; i++)
    {
        std::pair<std::vector<std::vector<double>>,std::vector<std::vector<std::vector<double>>>> res =   E_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, params);
        double val =  M_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, res, params);
    }

    

}