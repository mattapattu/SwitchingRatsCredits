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
  random_device rd;
  mt19937 gen(rd());
  uniform_int_distribution<> dis(0, n-1);
  return dis(gen);
}

// A function that returns a random sample from a discrete probability distribution
int sample(vector<double> p) {
  random_device rd;
  mt19937 gen(rd());
  discrete_distribution<> dis(p.begin(), p.end());
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
}



// A function that implements the particle filter algorithm
// Input: the number of particles N, the number of experts M, the number of trials T, the expert distributions p, and the observed outcomes y
// Output: a vector of posterior probabilities for each expert
vector<double> particle_filter(int N, RatData& ratdata, MazeGraph& Suboptimal_Hybrid3, MazeGraph& Optimal_Hybrid3,  vector<double> v) {

  
  arma::mat allpaths = ratdata.getPaths();
  arma::vec sessionVec = allpaths.col(4);
  arma::vec uniqSessIdx = arma::unique(sessionVec);
  int sessions = uniqSessIdx.n_elem;

  // Initialize the particles with random assignments
  //vector<vector<int>> m(N, vector<int>(T));
  vector<ParticleFilter> particleFilterVec;
  for (int i = 0; i < N; i++) {
   particleFilterVec.push_back(ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, v, i, 1.0/N));
  }


  // Iterate over the trials
  for (int ses = 0; ses < sessions; ses++) {
    //   // Initialize the weights with uniform probabilities
    vector<double> w(N,1.0);

    // Update the weights with the likelihood of the current outcome
    for (int i = 0; i < N; i++) {
        vector<double> q = particleFilterVec[i].getCrpPrior(ses);
        int sampled_strat = particleFilterVec[i].sample_crp(q); 
         particleFilterVec[i].addAssignment(ses,sampled_strat);
        double lik = particleFilterVec[i].getSesLikelihood(sampled_strat,ses);
        w[i] *= q[sampled_strat]*lik;
    }

    // Normalize the weights
    normalize(w);

    // Resample the particles with replacement according to the weights
    vector<vector<int>> m_new(N, vector<int>(T));
    for (int i = 0; i < N; i++) {
      int j = sample(w);
      particleFilterVec[i] = particleFilterVec[j];
    }
    // m = m_new;
  }

  // Compute the posterior probabilities for each expert by counting the occurrences of each expert in the last trial
  vector<vector<double>> q(sessions, vector<double>(4) );
  for(int ses=0; ses<sessions; ses++)
  {
    for (int i = 0; i < N; i++) {
        std::vector<int> chosenStrategy_pf = particleFilterVec[i].getChosenStratgies();
        q[[ses]chosenStrategy_pf[ses]]++;
    }
  }
  

  // Normalize the posterior probabilities
  normalizeRows(q);

  // Return the posterior probabilities
  return q;
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

// // A main function that tests the particle filter algorithm with some example data
// int main() {
//   // Define the number of particles, experts, and trials
//   int N = 100;

//   // Run the particle filter algorithm
//   vector<double> q = particle_filter(N, RatData& ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, v );


//   // Print the posterior probabilities
//   print_vector(q);

//   return 0;
// }
