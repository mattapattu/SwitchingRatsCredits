// Particle filter algorithm for expert assignment problem
// Assume that there are N particles, M experts, and T trials
// Each particle m is a vector of length T, where m[t] is the index of the expert assigned to trial t
// Each expert i has a probability distribution p_i over the possible outcomes of each trial
// The likelihood of a particle m given the observed outcomes y is the product of p_mt for all t
// The posterior probability of a particle m is proportional to its prior probability times its likelihood
// The prior probability of a particle m is assumed to be uniform over all possible assignments
// The resampling step is done using multinomial resampling with replacement

#include "ParticleFilter.h"
#include "Pagmoprob.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <omp.h>
#include <pagmo/batch_evaluators/thread_bfe.hpp>
#include <pagmo/algorithms/pso_gen.hpp>
#include <pagmo/algorithms/sade.hpp>

using namespace std;

// A function that returns a random integer in the range [0, n-1]
int randint(int n)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, n - 1);
    return dis(gen);
}

// A function that returns a random sample from a discrete probability distribution
int sample(vector<double> p)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dis(p.begin(), p.end());
    return dis(gen);
}

// A function that normalizes a vector of probabilities to sum to one
void normalizeRows(std::vector<std::vector<double>> &matrix)
{
    for (auto &row : matrix)
    {
        // Calculate the sum of elements in the current row
        double rowSum = 0.0;
        for (double element : row)
        {
            rowSum += element;
        }

        // Normalize the elements in the current row
        for (double &element : row)
        {
            element /= rowSum;
        }
    }
}

void normalize(vector<double> &p)
{

    bool hasNaN = std::any_of(p.begin(), p.end(),
                              [](double value)
                              { return std::isnan(value); });

    if (hasNaN)
    {
        throw std::runtime_error("nan weights before normalizing. Check");
    }
    double sum = 0;
    for (double x : p)
    {
        sum += x;
    }
    if (sum == 0)
    {

        // throw std::runtime_error("Error weight vec is zero");
        int N = p.size();
        double initWeight = 1.0 / (double)N;
        std::fill(p.begin(), p.end(), initWeight);
    }else
    {
        for (double &x : p)
        {
            x /= sum;
        }
    }
    

    hasNaN = std::any_of(p.begin(), p.end(),
                         [](double value)
                         { return std::isnan(value); });

    if (hasNaN)
    {
        throw std::runtime_error("nan weights after normalizing. Check");
    }
}

std::vector<double> systematicResampling(const std::vector<double> &particleWeights)
{
    int numParticles = particleWeights.size();
    std::vector<double> resampledParticles(numParticles, -1);

    // Compute cumulative weights
    std::vector<double> cumulativeWeights(numParticles);
    double totalWeight = 0.0;
    for (int i = 0; i < numParticles; ++i)
    {
        totalWeight += particleWeights[i];
        cumulativeWeights[i] = totalWeight;
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> uniformDist(0.0, 1.0 / numParticles);
    double u = uniformDist(gen);

    // For each point, find the corresponding particle and select it
    for (int j = 0; j < numParticles; j++)
    {
        auto it = std::lower_bound(cumulativeWeights.begin(), cumulativeWeights.end(), u);
        int index = std::distance(cumulativeWeights.begin(), it);

        resampledParticles[j] = index;
        u += 1.0 / numParticles;
    }

    return resampledParticles;
}

std::pair<std::vector<std::vector<double>>, double> particle_filter(int N, const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, vector<double> v)
{

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    // Initialize the particles with random assignments
    // vector<vector<int>> m(N, vector<int>(T));
    std::vector<ParticleFilter> particleFilterVec;
    for (int i = 0; i < N; i++)
    {
        auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, v, i, 1.0);
        particleFilterVec.push_back(pf);
        // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
    }

    vector<double> w(N, 1.0 / N);

    std::vector<std::vector<double>> filteredWeights;
    // std::vector<std::vector<double>> crpPriors(N, std::vector<double>(4, 0.0));

    double loglik = 0;
    // Iterate over the trials
    for (int ses = 0; ses < sessions; ses++)
    {
        //   // Initialize the weights with uniform probabilities
        double ses_lik = 0;
// Update the weights with the likelihood of the current outcome
#pragma omp parallel for shared(w, particleFilterVec, ses)
        for (int i = 0; i < N; i++)
        {
            particleFilterVec[i].updateStratCounts(ses);
            vector<double> crp_i = particleFilterVec[i].crpPrior(ses);
            // crpPriors.push_back(crp_i);

            int sampled_strat = sample(crp_i);
            particleFilterVec[i].addAssignment(ses, sampled_strat);
            double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses);
            w[i] *= lik;
            // ses_lik = ses_lik + log(lik);
            // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", sampled_strat=" << sampled_strat << ", lik=" << lik << ", w[i]=" << w[i] << std::endl;
            if (std::isinf(w[i]))
            {
                throw std::runtime_error("weight w[i] is infty after sampling. Check");
            }
        }

        // ses_lik = ses_lik/N;

        bool hasInf = std::any_of(w.begin(), w.end(),
                                  [](double value)
                                  { return std::isinf(value); });

        if (hasInf)
        {
            throw std::runtime_error("inf weights before normalizing. Check");
        }

        ses_lik = std::accumulate(w.begin(), w.end(), 0.0);
        ses_lik = ses_lik / N;
        if (ses_lik == 0)
        {
            ses_lik = std::numeric_limits<double>::min();
        }
        loglik = loglik + log(ses_lik);

        if (std::isinf(loglik))
        {
            throw std::runtime_error("ses loglik is inf. Check");
        }

        // Normalize the weights
        normalize(w);

        filteredWeights.push_back(w);

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
        for (int k = 0; k < N; k++)
        {
            weightSq = weightSq + std::pow(w[k], 2);
        }
        double n_eff = 1 / weightSq;
        if (n_eff < N / 2)
        {
            // std::cout << "ses=" <<ses <<", n_eff=" << n_eff << ", performing resampling" << std::endl;
            std::vector<double> resampledIndices = systematicResampling(w);
#pragma omp parallel for shared(particleFilterVec, resampledIndices, ses)
            for (int i = 0; i < N; i++)
            {
                int newIndex = resampledIndices[i];
                ParticleFilter pf(particleFilterVec[newIndex]);
                int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
                particleFilterVec[i].setChosenStrategies(particleFilterVec[newIndex].getChosenStratgies());
                particleFilterVec[i].setStrategies(particleFilterVec[newIndex].getStrategies());
                int updatedChosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
                // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[newIndex].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;
            }

            double initWeight = 1.0 / (double)N;
            std::fill(w.begin(), w.end(), initWeight);
        }
    }

    // Compute the posterior probabilities for each expert by counting the occurrences of each expert in the last trial
    vector<vector<double>> postProbsOfExperts(sessions, vector<double>(4));
    for (int ses = 0; ses < sessions; ses++)
    {
        for (int i = 0; i < N; i++)
        {
            std::vector<int> chosenStrategy_pf = particleFilterVec[i].getChosenStratgies();

            // std::cout << "ses=" <<ses << ", particleId=" <<i << ", chosenStrat=" << chosenStrategy_pf[ses] << std::endl;
            postProbsOfExperts[ses][chosenStrategy_pf[ses]] = postProbsOfExperts[ses][chosenStrategy_pf[ses]] + filteredWeights[ses][i];
            // postProbsOfExperts[ses][chosenStrategy_pf[ses]] = std::round(postProbsOfExperts[ses][chosenStrategy_pf[ses]] * 100.0) / 100.0;
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
    // normalizeRows(q);

    std::cout << "posterior probs=" << std::endl;
    for (const auto &row : postProbsOfExperts)
    {
        for (double num : row)
        {
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
std::tuple<std::vector<std::vector<double>>, double, std::vector<std::vector<double>>> particle_filter_new(int N, std::vector<ParticleFilter> &particleFilterVec, const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3)
{

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    // Initialize the particles with random assignments
    // vector<vector<int>> m(N, vector<int>(T));

    vector<double> w(N, 1.0 / N);

    // w[t][i]
    std::vector<std::vector<double>> filteredWeights;
    std::vector<std::vector<double>> crpPriors;

    double loglik = 0;
    // Iterate over the trials
    for (int ses = 0; ses < sessions; ses++)
    {
        //   // Initialize the weights with uniform probabilities
        double ses_lik = 0;
// std::cout << "ses=" << ses << std::endl;
// Update the weights with the likelihood of the current outcome
#pragma omp parallel for shared(w, particleFilterVec, ses)
        for (int i = 0; i < N; i++)
        {
            particleFilterVec[i].updateStratCounts(ses);
            vector<double> crp_i = particleFilterVec[i].crpPrior(ses);
            particleFilterVec[i].addCrpPrior(crp_i);

            int sampled_strat = sample(crp_i);
            particleFilterVec[i].addAssignment(ses, sampled_strat);
            particleFilterVec[i].addOriginalSampledStrat(ses, sampled_strat);
            double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses);
            particleFilterVec[i].setLikelihood(lik);
            w[i] *= lik;
            // ses_lik = ses_lik + lik;
            //   std::cout << "ses=" << ses << ", crpPrior=";
            //  for (auto const& i : crp_i)
            //      std::cout << i << ", ";
            //  std::cout << "\n" ;

            // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", sampled_strat=" << sampled_strat << ", lik=" << lik << ", w[i]=" << w[i] << std::endl;
            if (std::isnan(w[i]))
            {
                throw std::runtime_error("weight w[i] is nan after sampling. Check");
            }

            if (std::isinf(w[i]))
            {
                throw std::runtime_error("weight w[i] is infty after sampling. Check");
            }
        }

        ses_lik = std::accumulate(w.begin(), w.end(), 0.0);
        ses_lik = ses_lik / N;
        if (ses_lik == 0)
        {
            ses_lik = std::numeric_limits<double>::min();
        }
        loglik = loglik + log(ses_lik);

        if (std::isinf(loglik))
        {
            throw std::runtime_error("ses loglik is inf. Check");
        }
        // Normalize the weights
        normalize(w);

        filteredWeights.push_back(w);

        // #####################

        double weightSq = 0;
        for (int k = 0; k < N; k++)
        {
            weightSq = weightSq + std::pow(w[k], 2);
        }
        double n_eff = 1 / weightSq;
        if (1)
        {
            // std::cout << "ses=" <<ses <<", n_eff=" << n_eff << ", performing resampling" << std::endl;
            std::vector<double> resampledIndices = systematicResampling(w);
// std::cout << "ses=" <<ses <<", updating particles" << std::endl;
#pragma omp parallel for shared(resampledIndices, particleFilterVec, ses)
            for (int i = 0; i < N; i++)
            {
                int newIndex = resampledIndices[i];
                ParticleFilter pf(particleFilterVec[newIndex]);
                int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
                particleFilterVec[i].setChosenStrategies(particleFilterVec[newIndex].getChosenStratgies());
                particleFilterVec[i].setStrategies(particleFilterVec[newIndex].getStrategies());
                int updatedChosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
                // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[newIndex].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;
            }
            // std::cout << "ses=" <<ses <<", updating particles completed." << std::endl;
            double initWeight = 1.0 / (double)N;
            std::fill(w.begin(), w.end(), initWeight);
        }
    }

    // Compute the posterior probabilities for each expert by counting the occurrences of each expert in the last trial
    // vector<vector<double>> postProbsOfExperts(sessions, vector<double>(4) );

    // Return the posterior probabilities
    return std::make_tuple(filteredWeights, loglik, crpPriors);
}

// A function that prints a vector of doubles
void print_vector(vector<double> v)
{
    cout << "[";
    for (int i = 0; i < v.size(); i++)
    {
        cout << v[i];
        if (i < v.size() - 1)
        {
            cout << ", ";
        }
    }
    cout << "]" << endl;
}

std::vector<std::vector<double>> w_smoothed(std::vector<std::vector<double>> filteredWeights, std::vector<ParticleFilter> particleFilterVec, int N)
{
    int n_rows = filteredWeights.size();
    // w_smoothed[t][i]

    std::vector<std::vector<double>> v(n_rows, std::vector<double>(N, 0.0));

#pragma omp parallel for collapse(2)
    for (int t = 0; t < n_rows - 1; t++)
    {
        for (int k = 0; k < N; k++)
        {
            for (int j = 0; j < N; j++)
            {
                std::vector<std::vector<double>> crpPriors_j = particleFilterVec[j].getCrpPriors();
                std::vector<double> crpPriors_j_t = crpPriors_j[t];

                std::vector<int> assignments_k = particleFilterVec[k].getOriginalSampledStrats();
                int X_k_tplus1 = assignments_k[t + 1];

                v[t][k] = v[t][k] + (filteredWeights[t][j] * crpPriors_j_t[X_k_tplus1]);

                if (std::isnan(v[t][k]))
                {
                    throw std::runtime_error("v[t][k] is nan. Check");
                }
                if (std::isinf(v[t][k]))
                {
                    throw std::runtime_error("v[t][k] is infinity. Check");
                }
            }
            if (v[t][k] == 0)
            {
                v[t][k] = 1e-6;
            }
        }
    }

    // std::cout << "v:";
    // for (const auto& row : v) {
    //     for (const auto& element : row) {
    //         std::cout << element << " ";
    //     }
    //     std::cout << std::endl;  // Start a new line for each row
    // }

    std::vector<std::vector<double>> smoothedWeights(n_rows, std::vector<double>(N, 0.0));

    for (int t = n_rows - 1; t >= 0; t--)
    {
// std::cout << "ses=" << t <<", w_smoothed." << std::endl;
#pragma omp parallel for
        for (int i = 0; i < N; i++)
        {
            if (t == n_rows - 1)
            {
                smoothedWeights[t][i] = filteredWeights[n_rows - 1][i];
            }
            else
            {
                std::vector<std::vector<double>> crpPriors_i = particleFilterVec[i].getCrpPriors();
                std::vector<double> crpPriors_i_t = crpPriors_i[t];

                double weightedSum = 0;
#pragma omp parallel for reduction(+ : weightedSum)
                for (int k = 0; k < N; k++)
                {
                    std::vector<int> assignments_k = particleFilterVec[k].getOriginalSampledStrats();
                    int X_k_tplus1 = assignments_k[t + 1];
                    weightedSum = weightedSum + (smoothedWeights[t + 1][k] * crpPriors_i_t[X_k_tplus1] / v[t][k]);
                    if (std::isnan(weightedSum))
                    {
                        throw std::runtime_error("weightedSum is nan. Check");
                    }

                    if (std::isinf(weightedSum))
                    {
                        throw std::runtime_error("weightedSum is infinity. Check");
                    }
                }
                smoothedWeights[t][i] = weightedSum * filteredWeights[t][i];
                if (smoothedWeights[t][i] > 1)
                {
                    throw std::runtime_error("weights greater than 1. Check");
                }
            }
        }

        bool sumIsZero = std::accumulate(smoothedWeights[t].begin(), smoothedWeights[t].end(), 0.0,
                                         [](double sum, double element)
                                         {
                                             return sum + element;
                                         }) == 0.0;
    }

    // bool allZero = std::all_of(smoothedWeights.begin(), smoothedWeights.end(),
    //     [](const std::vector<double>& row) {
    //         return std::all_of(row.begin(), row.end(),
    //             [](double element) {
    //                 return element == 0.0;
    //             });
    // });

    // if(allZero)
    // {
    //     throw std::runtime_error("smoothedWeights are zero. Check");
    // }

    return (smoothedWeights);
}

std::vector<std::vector<std::vector<double>>> wij_smoothed(std::vector<std::vector<double>> filteredWeights, std::vector<ParticleFilter> particleFilterVec, std::vector<std::vector<double>> w_smoothed, int N)
{
    int n_rows = filteredWeights.size();
    std::vector<std::vector<std::vector<double>>> wijSmoothed(n_rows - 2, std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0)));

    std::vector<std::vector<double>> v(n_rows, std::vector<double>(N, 0.0));
#pragma omp parallel for collapse(2)
    for (int t = 0; t < n_rows - 2; t++)
    {
        for (int j = 0; j < N; j++)
        {
            for (int l = 0; l < N; l++)
            {
                double w_t_l = filteredWeights[t][l];
                std::vector<std::vector<double>> crpPriors_l = particleFilterVec[l].getCrpPriors();
                std::vector<int> assignments_j = particleFilterVec[j].getOriginalSampledStrats();
                int x_j_tplus1 = assignments_j[t + 1];

                std::vector<double> crpPriors_l_t = crpPriors_l[t];
                v[t][j] = v[t][j] + crpPriors_l_t[x_j_tplus1] * w_t_l;
            }
            if (v[t][j] == 0)
            {
                v[t][j] = 1e-6;
            }
        }
    }

    for (int t = 0; t < n_rows - 2; t++)
    {
        // std::cout << "wij_smoothed, t=" << t << std::endl;
        std::vector<std::vector<double>> &wij_smoothed_t = wijSmoothed[t];
#pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {

                double w_t_i = filteredWeights[t][i];
                double w_smoothed_tplus1_j = w_smoothed[t + 1][j];

                std::vector<std::vector<double>> crpPriors_i = particleFilterVec[i].getCrpPriors();
                std::vector<double> crpPriors_i_t = crpPriors_i[t];
                std::vector<int> assignments_j = particleFilterVec[j].getOriginalSampledStrats();
                int x_j_tplus1 = assignments_j[t + 1];

                // double denom=0;
                // #pragma omp parallel for reduction(+:denom)
                // for(int l=0; l<N; l++)
                // {
                //     double w_t_l = filteredWeights[t][l];
                //     std::vector<std::vector<double>> crpPriors_l = particleFilterVec[l].getCrpPriors();
                //     std::vector<double> crpPriors_l_t = crpPriors_l[t];
                //     denom = denom + crpPriors_l_t[x_j_tplus1] * w_t_l;

                // }
                wij_smoothed_t[i][j] = w_t_i * w_smoothed_tplus1_j * crpPriors_i_t[x_j_tplus1] / v[t][j];
            }
        }
    }

    // std::cout << "wij_smoothed completed" << std::endl;

    // wij_smoothed[t][i]

    return (wijSmoothed);
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> E_step(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::vector<double> params)
{
    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    std::vector<ParticleFilter> particleFilterVec;
    for (int i = 0; i < N; i++)
    {
        auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, i, 1.0);
        particleFilterVec.push_back(pf);
        // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
    }

    std::tuple<std::vector<std::vector<double>>, double, std::vector<std::vector<double>>> resTuple = particle_filter_new(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3);
    std::vector<std::vector<double>> filteredWeights = std::get<0>(resTuple);
    std::vector<std::vector<double>> smoothedWeights = w_smoothed(filteredWeights, particleFilterVec, N);
    
    std::cout << "filteredWeights:";
    for (const auto& row : filteredWeights) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;  // Start a new line for each row
    }

    
    std::cout << "smoothedWeights:";
    for (const auto& row : smoothedWeights) {
        for (const auto& element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;  // Start a new line for each row
    }

    std::vector<std::vector<std::vector<double>>> wijSmoothed = wij_smoothed(filteredWeights, particleFilterVec, smoothedWeights, N);
    for (const auto& m : wijSmoothed) {
        std::cout << "[\n"; // Start of a matrix
        // Loop over the rows
        for (const auto& r : m) {
            std::cout << "  ["; // Start of a row
            // Loop over the columns
            for (const auto& c : r) {
                std::cout << c << " "; // Print an element
            }
        std::cout << "]\n"; // End of a row
        }
    std::cout << "]\n"; // End of a matrix
  }

    return std::make_tuple(smoothedWeights, wijSmoothed, particleFilterVec);
}

double M_step(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> smoothed_w, std::vector<double> params)
{

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    std::vector<std::vector<double>> smoothedWeights = std::get<0>(smoothed_w);
    std::vector<std::vector<std::vector<double>>> wijSmoothed = std::get<1>(smoothed_w);
    std::vector<ParticleFilter> truePfVec = std::get<2>(smoothed_w);

    std::vector<ParticleFilter> particleFilterVec;
    for (int i = 0; i < N; i++)
    {
        auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, i, 1.0);
        pf.setStratCounts(truePfVec[i].getStratCounts());
        pf.setChosenStrategies(truePfVec[i].getChosenStratgies());
        pf.setOriginalSampledStrats(truePfVec[i].getOriginalSampledStrats());
        pf.setCrpPriors(truePfVec[i].getCrpPriors());
        particleFilterVec.push_back(pf);
        // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
    }

    double I_1 = 0;
    double I_2 = 0;
    double I_3 = 0;

    for (int t = 0; t < sessions; t++)
    {
// std::cout << "I_3 loop, t=" << t << std::endl;
#pragma omp parallel for reduction(+ : I_3)
        for (int i = 0; i < N; i++)
        {
            std::vector<int> originalSampledStrats = particleFilterVec[i].getOriginalSampledStrats();
            double lik_i = particleFilterVec[i].getSesLikelihood(originalSampledStrats[t], t);
            if (lik_i == 0)
            {
                lik_i = 1e-6;
            }
            I_3 = I_3 + (smoothedWeights[t][i] * log(lik_i));

            if (std::isnan(I_3))
            {
                throw std::runtime_error("Error nan I_3 value");
            }

            if (std::isinf(I_3))
            {
                throw std::runtime_error("Error inf I_3 value");
            }
        }
    }

    // for(int t=0; t<sessions-2; t++)
    // {
    //     // std::cout << "I_2 loop, t=" << t << std::endl;
    //     #pragma omp parallel for collapse(2) reduction(+:I_2)
    //     for(int i=0; i<N; i++)
    //     {
    //         for(int j=0; j<N; j++)
    //         {
    //             std::vector<double> crpPriors_i_t = particleFilterVec[i].crpPrior(t);
    //             std::vector<int> assignments_j = particleFilterVec[j].getOriginalSampledStrats();
    //             int X_j_tplus1 = assignments_j[t+1];

    //             double w_ij_smoothed = wijSmoothed[t][i][j];
    //             I_2 = I_2 + (log(crpPriors_i_t[X_j_tplus1])*w_ij_smoothed);

    //             if (std::isnan(I_2)) {
    //                 throw std::runtime_error("Error nan I_2 value");
    //             }

    //             if (std::isinf(I_2)) {
    //                 throw std::runtime_error("Error inf I_2 value");
    //             }
    //         }
    //     }
    // }

    for (int i = 0; i < N; i++)
    {
        std::vector<std::vector<double>> crpPriors_i = particleFilterVec[i].getCrpPriors();
        std::vector<double> crpPriors_i_0 = crpPriors_i[0];
        std::vector<int> assignments_i = particleFilterVec[i].getChosenStratgies();
        int X_i_0 = assignments_i[0];

        I_1 = I_1 + (smoothedWeights[0][i] * log(crpPriors_i_0[X_i_0]));
    }

    // for(int t=0; t<sessions-1; t++)
    // {
    //     for(int i=0; i<N; i++)
    //     {
    //         std::vector<double> liks_i = particleFilterVec[i].getLikelihoods();
    //         I_3 = I_3+(smoothedWeights[t][i]*log(liks_i[t]));
    //     }
    // }

    if (I_1 + I_2 + I_3 == 0)
    {
        return -1e6;
    }

    std::cout << "I_1=" << I_1 << ", I_3=" << I_3 << std::endl;

    return (I_1 + I_2 + I_3);
}

std::vector<double> EM(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N)
{
    double finalChamp = 1e8;
    std::vector<double> paramVec;
    for(int k = 0; k< 5; k++)
    {
        std::pair<std::vector<double>, std::vector<double>> bounds = {
        {0.1, 0.7, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8},
        {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

        // Initialize a random number generator
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist;

        // Generate a vector of 8 doubles within the specified range
        std::vector<double> params;
        for (int i = 0; i < 8; ++i) {
            double randomValue = bounds.first[i] + dist(gen) * (bounds.second[i] - bounds.first[i]);
            params.push_back(randomValue);
        }

        std::cout << "k=" <<k << ", params: ";
        for (const auto &x : params)
        {
            std::cout << x << " ";
        }
        std::cout << "\n";


        std::vector<double> QFuncVals;
        double prevChampion = 1e6;
        double champion = 1e8;
        bool terminate = false;

        for (int i = 0; i < 10; i++)
        {

            std::cout << "k=" <<k <<  ", i=" << i << ", E-step" << std::endl;
            std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> res = E_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, params);
            std::vector<std::vector<double>> smoothedWeights = std::get<0>(res);
            std::vector<std::vector<std::vector<double>>> wijSmoothed = std::get<1>(res);

            // std::cout << "smoothedWeights:";
            // for (const auto& row : smoothedWeights) {
            //     for (const auto& element : row) {
            //         std::cout << element << " ";
            //     }
            //     std::cout << std::endl;  // Start a new line for each row
            // }

            // std::vector<std::vector<double>> wijSmoothed_0 = wijSmoothed[0];
            // std::cout << "wijSmoothed[0]:";
            // for (const auto& row : wijSmoothed_0) {
            //     for (const auto& element : row) {
            //         std::cout << element << " ";
            //     }
            //     std::cout << std::endl;  // Start a new line for each row
            // }

            // std::vector<std::vector<double>> wijSmoothed_5 = wijSmoothed[5];
            // std::cout << "wijSmoothed[5]:";
            // for (const auto& row : wijSmoothed_5) {
            //     for (const auto& element : row) {
            //         std::cout << element << " ";
            //     }
            //     std::cout << std::endl;  // Start a new line for each row
            // }

            std::cout << "i=" << i << ", starting M-step" << std::endl;

            PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, res);
            // PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
            std::cout << "Initialized problem class" << std::endl;

            // Create a problem using Pagmo
            problem prob{pagmoprob};
            int count = 0;
            std::vector<double> dec_vec_champion;
            while (prevChampion - champion <= 0)
            {
                pagmo::nlopt method("sbplx");
                method.set_maxeval(20);
                pagmo::algorithm algo = pagmo::algorithm{method};
                pagmo::population pop(prob, 10);
                pop = algo.evolve(pop);

                dec_vec_champion = pop.champion_x();
                champion = pop.champion_f()[0];
                std::cout << "Final champion = " << champion << std::endl;

                std::cout << "dec_vec_champion: ";
                for (const auto &x : dec_vec_champion)
                {
                    std::cout << x << " ";
                }
                std::cout << "\n";
                count++;
                if (count == 2)
                {
                    std::cout << "Exiting, not able to improve Qfunction" << std::endl;
                    terminate = true;
                    break;
                }
            }
            if (terminate)
            {
                break;
            }
            params = dec_vec_champion;
            std::pair<std::vector<std::vector<double>>, double> q = particle_filter(N, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params);
            std::cout << "loglikelihood=" << q.second << std::endl;
            QFuncVals.push_back(q.second);

            if(champion<finalChamp)
            {
                paramVec = params;
            }

            prevChampion = champion;
            champion = 1e8;
        }

        std::cout << "Likelihoods=";
        for (auto const &i : QFuncVals)
            std::cout << i << ", ";
        std::cout << "\n";

    }

    return (paramVec);
}

std::vector<double> Mle(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N)
{
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> res;

    PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, res);
    // PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
    std::cout << "Initialized problem class" << std::endl;

    // Create a problem using Pagmo
    problem prob{pagmoprob};

    // pagmo::unconstrain unprob{prob, "kuri"};

    // pagmo::thread_bfe thread_bfe;
    // pagmo::pso_gen method ( 5 );
    // method.set_bfe ( pagmo::bfe { thread_bfe } );
    // pagmo::algorithm algo = pagmo::algorithm { method };
    // pagmo::population pop { prob, thread_bfe,100 };
    // for ( auto evolution = 0; evolution < 3; evolution++ ) {
    //     pop = algo.evolve(pop);
    // }

    // pagmo::nlopt method("sbplx");
    // method.set_maxeval(10);
    // pagmo::algorithm algo = pagmo::algorithm {method};
    // pagmo::population pop(prob, 10);
    // pop = algo.evolve(pop);

    // std::vector<double> dec_vec_champion = pop.champion_x();
    // std::cout << "Final champion = " << pop.champion_f()[0] << std::endl;

    // std::cout << "dec_vec_champion: ";
    // for (const auto &x : dec_vec_champion) {
    //     std::cout << x << " ";
    // }
    // std::cout << "\n";

    pagmo::algorithm algo{sade(10, 2, 2)};
    archipelago archi{5u, algo, prob, 10u};
    archi.evolve(5);
    archi.wait_check();

    double champion_score = 1e8;
    std::vector<double> dec_vec_champion;
    for (const auto &isl : archi)
    {
        std::vector<double> dec_vec = isl.get_population().champion_x();

        // std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
        // for (auto const& i : dec_vec)
        //     std::cout << i << ", ";
        // std::cout << "\n" ;

        double champion_isl = isl.get_population().champion_f()[0];
        if (champion_isl < champion_score)
        {
            champion_score = champion_isl;
            dec_vec_champion = dec_vec;
        }
    }

    std::cout << "Final champion = " << champion_score << std::endl;
    for (auto const &i : dec_vec_champion)
        std::cout << i << ", ";
    std::cout << "\n";

    // const auto fv = prob.fitness(dec_vec_champion);
    // std::cout << "Value of the objfun in dec_vec_champion: " << fv[0] << '\n';

    std::vector<double> params = dec_vec_champion;
    std::pair<std::vector<std::vector<double>>, double> q = particle_filter(N, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params);
    std::cout << "loglikelihood=" << q.second << std::endl;

    return (params);
}


void testQFunc(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N)
{
    
    for(int i=0; i<1; i++)
    {
        std::vector<double> params = {0.228439, 0.921127, 0.0429102, 0.575078, 0.755174, 0.0658294, 0.09602, 0.179565};
        std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> res = E_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, params);
        double Q = M_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, res, params);
        std::cout << "i=" <<i << ", Q=" << Q << std::endl;
    }
}