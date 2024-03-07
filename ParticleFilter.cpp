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
#include "BS_thread_pool.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <pagmo/archipelago.hpp>
#include <pagmo/algorithms/de.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/problems/schwefel.hpp>



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

std::vector<double> colwise_mean(const std::vector<std::vector<double>>& matrix) {
    std::size_t rows = matrix.size();
    std::size_t cols = matrix[0].size();  // Assuming all rows have the same number of columns

    std::vector<double> means(cols, 0.0);

    for (std::size_t j = 0; j < cols; ++j) {
        double col_sum = 0.0;

        for (std::size_t i = 0; i < rows; ++i) {
            col_sum += matrix[i][j];
        }

        means[j] = col_sum / rows;
    }

    return means;
}


std::pair<std::vector<std::vector<double>>, double> particle_filter(int N, const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, vector<double> v,BS::thread_pool& pool)
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
    vector<double> log_w(N, log(1.0 / N));

    std::vector<std::vector<double>> filteredWeights;
    // std::vector<std::vector<double>> crpPriors(N, std::vector<double>(4, 0.0));

    double loglik = 0;
    // Iterate over the trials
    for (int ses = 0; ses < sessions; ses++)
    {
        //   // Initialize the weights with uniform probabilities
        double ses_lik = 0;
        // Update the weights with the likelihood of the current outcome
        // std::mutex mutex;
        std::vector<std::vector<double>> samplingDistributionVec(N, std::vector<double>(4, 0.0));
        auto generate_sampling_distribution = [&particleFilterVec,&ses,&samplingDistributionVec](int i) {

            particleFilterVec[i].updateStratCounts(ses);
            std::vector<double> crp_i = particleFilterVec[i].crpPrior(ses);

            std::vector<std::shared_ptr<Strategy>> strategies = particleFilterVec[i].getStrategies();
            particleFilterVec[i].backUpStratCredits();
            std::vector<double> likelihoods;
            for(int k=0; k<strategies.size();k++)
            {
                double lik = particleFilterVec[i].getSesLikelihood(k, ses);
                if(lik==0)
                {
                    lik = std::numeric_limits<double>::min();
                }
                likelihoods.push_back(lik);
            }

            double sum = 0;
            for(int k=0; k<strategies.size();k++)
            {
                sum += crp_i[k]*likelihoods[k];
            }    

            std::vector<double> p;
            for(int k=0; k<strategies.size();k++)
            {
                p.push_back(crp_i[k]*likelihoods[k]/sum);
            }

            if(sum==0)
            {
                std::cout << "Err in generate_sampling_distribution. Check" << std::endl;
                std::cout << "ses=" << ses << " i=" << i <<  ", crp_i = ";
                for (auto const& w : crp_i)
                    std::cout << w << ", ";
                std::cout << "\n" ;
                std::cout << "ses=" << ses << " i=" << i <<  ", likelihoods = ";
                for (auto const& w : likelihoods)
                    std::cout << w << ", ";
                std::cout << "\n" ;

                // throw std::runtime_error("Err in generate_sampling_distribution. Check");
            }

            particleFilterVec[i].rollBackCredits();

            samplingDistributionVec[i] = p;

        };

        for (int i = 0; i < N; i++) {
            auto bound_lambda = std::bind(generate_sampling_distribution, i);
            std::future<void> my_future = pool.submit_task(bound_lambda);
        }

        pool.wait();

        // std::vector<double> samplingDistribution_ses = colwise_mean(samplingDistributionVec);

        auto sample_particle_filter = [&particleFilterVec,&ses,&w](int i, std::vector<double> p) {
            // Use lock_guard to ensure mutual exclusion for particleFilterVec and w updates
            // std::lock_guard<std::mutex> lock(mutex);

            //particleFilterVec[i].updateStratCounts(ses);
            //std::vector<double> crp_i = particleFilterVec[i].crpPrior(ses);
            particleFilterVec[i].addCrpPrior(p);

            int sampled_strat = sample(p);
            if(p[sampled_strat] == 0)
            {
                std::cout << "Sample not matching distribution. Check" << std::endl;
                std::cout << "ses=" << ses << ", sample=" << sampled_strat;
                for (auto const& n : p)
                    std::cout << n << ", ";
                std::cout << "\n" ;

                // throw std::runtime_error("Sample not matching distribution. Check");
            }
            particleFilterVec[i].addAssignment(ses, sampled_strat);
            particleFilterVec[i].addOriginalSampledStrat(ses, sampled_strat);
            double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses);
            // double loglik = particleFilterVec[i].getSesLogLikelihood(sampled_strat, ses);
            particleFilterVec[i].setLikelihood(lik);
            w[i] *= lik;
            // log_w[i] += loglik;

            particleFilterVec[i].backUpStratCredits(); 

        };

        // ThreadPool pool(16);
        // std::vector<std::future<void>> futures;

        for (int i = 0; i < N; i++) {
            auto bound_lambda = std::bind(sample_particle_filter, i, samplingDistributionVec[i]);
            std::future<void> my_future = pool.submit_task(bound_lambda);
        }

        pool.wait();

        // // Wait for all tasks to complete
        // for (auto& future : futures) {
        //     future.wait();
        // }

        // double max_log_w = *std::max_element(log_w.begin(), log_w.end());

        // for (std::size_t i = 0; i < log_w.size(); ++i) {
        //         w[i] = std::exp(log_w[i] - max_log_w);
        //     }


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
        if (n_eff < N/5)
        {
            // std::cout << "ses=" <<ses <<", n_eff=" << n_eff << ", performing resampling" << std::endl;
            std::vector<double> resampledIndices = systematicResampling(w);
            //std::mutex mutex;

            
            for(int j=0; j<N; j++)
            {
                particleFilterVec[j].backUpChosenStrategies();
                particleFilterVec[j].backUpStratCredits();
            }
           
            auto resample_particle_filter = [&particleFilterVec,&ses](int i, std::vector<double> resampledIndices_) {
                int newIndex = resampledIndices_[i];
                //ParticleFilter pf(particleFilterVec[newIndex]);

                // Use lock_guard to ensure mutual exclusion for particleFilterVec updates
                std::vector<std::pair<std::vector<double>,std::vector<double>>>& stratBackUps = particleFilterVec[newIndex].getStratCreditBackUps();
                std::vector<signed int> & chosenStrategyBackUp =  particleFilterVec[newIndex].getChosenStrategyBackups();
                int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
                particleFilterVec[i].setChosenStrategies(chosenStrategyBackUp);
                particleFilterVec[i].setStratBackups(stratBackUps);
                int updatedChosenStrat = particleFilterVec[i].getChosenStratgies()[ses];

                // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[newIndex].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;
            };


            for (int i = 0; i < N; i++) {
                auto bound_lambda = std::bind(resample_particle_filter, i,resampledIndices);
                std::future<void> my_future = pool.submit_task(bound_lambda);

            }

            pool.wait();
            // // Wait for all tasks to complete
            // for (auto& future : futures) {
            //     future.wait();
            // }

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
std::tuple<std::vector<std::vector<double>>, double, std::vector<std::vector<double>>> particle_filter_new(int N, std::vector<ParticleFilter> &particleFilterVec, const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, BS::thread_pool& pool)
{

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    // Initialize the particles with random assignments
    // vector<vector<int>> m(N, vector<int>(T));

    vector<double> w(N, 1.0 /(double) N);
    // vector<double> log_w(N, log(1.0 / N));

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
        
        
        std::vector<std::vector<double>> samplingDistributionVec(N, std::vector<double>(4, 0.0));
        auto generate_sampling_distribution = [&particleFilterVec,&ses,&samplingDistributionVec](int i) {

            particleFilterVec[i].updateStratCounts(ses);
            std::vector<double> crp_i = particleFilterVec[i].crpPrior(ses);

            std::vector<std::shared_ptr<Strategy>> strategies = particleFilterVec[i].getStrategies();
            particleFilterVec[i].backUpStratCredits();
            std::vector<double> likelihoods;
            for(int k=0; k<strategies.size();k++)
            {
                double lik = particleFilterVec[i].getSesLikelihood(k, ses);
                if(lik==0)
                {
                    lik = std::numeric_limits<double>::min();
                }
                likelihoods.push_back(lik);
            }

            double sum = 0;
            for(int k=0; k<strategies.size();k++)
            {
                sum += crp_i[k]*likelihoods[k];
            }    

            std::vector<double> p;
            for(int k=0; k<strategies.size();k++)
            {
                p.push_back(crp_i[k]*likelihoods[k]/sum);
            }

            if(sum==0)
            {
                std::cout << "Err in generate_sampling_distribution. Check" << std::endl;
                std::cout << "ses=" << ses << " i=" << i <<  ", crp_i = ";
                for (auto const& w : crp_i)
                    std::cout << w << ", ";
                std::cout << "\n" ;
                std::cout << "ses=" << ses << " i=" << i <<  ", likelihoods = ";
                for (auto const& w : likelihoods)
                    std::cout << w << ", ";
                std::cout << "\n" ;

                // throw std::runtime_error("Err in generate_sampling_distribution. Check");
            }

            particleFilterVec[i].rollBackCredits();

            samplingDistributionVec[i] = p;

        };

        for (int i = 0; i < N; i++) {
            auto bound_lambda = std::bind(generate_sampling_distribution, i);
            std::future<void> my_future = pool.submit_task(bound_lambda);
        }

        pool.wait();

        // std::vector<double> samplingDistribution_ses = colwise_mean(samplingDistributionVec);

        auto sample_particle_filter = [&particleFilterVec,&ses,&w](int i, std::vector<double> p) {
            // Use lock_guard to ensure mutual exclusion for particleFilterVec and w updates
            // std::lock_guard<std::mutex> lock(mutex);

            //particleFilterVec[i].updateStratCounts(ses);
            //std::vector<double> crp_i = particleFilterVec[i].crpPrior(ses);
            particleFilterVec[i].addCrpPrior(p);

            int sampled_strat = sample(p);
            if(p[sampled_strat] == 0)
            {
                std::cout << "Sample not matching distribution. Check" << std::endl;
                std::cout << "ses=" << ses << ", sample=" << sampled_strat;
                for (auto const& n : p)
                    std::cout << n << ", ";
                std::cout << "\n" ;

                // throw std::runtime_error("Sample not matching distribution. Check");
            }
            particleFilterVec[i].addAssignment(ses, sampled_strat);
            particleFilterVec[i].addOriginalSampledStrat(ses, sampled_strat);
            double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses);
            // double loglik = particleFilterVec[i].getSesLogLikelihood(sampled_strat, ses);
            particleFilterVec[i].setLikelihood(lik);
            w[i] *= lik;
            // log_w[i] += loglik;

            particleFilterVec[i].backUpStratCredits(); 

            // Add any other logic specific to your application

            // if (std::isnan(w[i])) {
            //     throw std::runtime_error("weight w[i] is nan after sampling. Check");
            // }

            // if (std::isinf(w[i])) {
            //     throw std::runtime_error("weight w[i] is infty after sampling. Check");
            // }
        };

        // ThreadPool pool(16);
        // std::vector<std::future<void>> futures;

        for (int i = 0; i < N; i++) {
            auto bound_lambda = std::bind(sample_particle_filter, i, samplingDistributionVec[i]);
            std::future<void> my_future = pool.submit_task(bound_lambda);
        }

        pool.wait();


        // double max_log_w = *std::max_element(log_w.begin(), log_w.end());

        // for (std::size_t i = 0; i < log_w.size(); ++i) {
        //         w[i] = std::exp(log_w[i] - max_log_w);
        //     }

        // std::cout << "ses=" << ses << ", particle weights log_w = ";
        // for (auto const& i : log_w)
        //     std::cout << i << ", ";
        // std::cout << "\n" ;
    

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
       


        // std::cout << "ses=" << ses << ", particle w = ";
        // for (auto const& i : w)
        //     std::cout << i << ", ";
        // std::cout << "\n" ;

        // for(int l=0; l<N;l++)
        // {
        //     std::vector<double> likelioods = particleFilterVec[l].getLikelihoods();
        //     std::vector<int> history = particleFilterVec[l].getChosenStratgies();
        //     std::cout << "ses=" << ses << ", particle=" << l <<", history=";
        //     for (int n=0; n<=ses; n++)
        //         std::cout << history[n] << ", ";
        //     std::cout << "\n" ;


        //     std::cout << "ses=" << ses << ", particle=" << l <<", likelihoods=";
        //     for (auto const& m : likelioods)
        //         std::cout << m << ", ";
        //     std::cout << "\n" ;

        // }

        // std::cout << "ses=" << ses << ", samplingDistributionVec:";
        // for (const auto& row : samplingDistributionVec) {
        //     for (const auto& element : row) {
        //         std::cout << element << " ";
        //     }
        //     std::cout << std::endl;
        // }



        // #####################

        double weightSq = 0;
        for (int k = 0; k < N; k++)
        {
            weightSq = weightSq + std::pow(w[k], 2);
        }
        double n_eff = 1 / weightSq;
        if (n_eff < N/5)
        {
            // std::cout << "ses=" <<ses <<", n_eff=" << n_eff << ", performing resampling" << std::endl;
            std::vector<double> resampledIndices = systematicResampling(w);
            // std::cout << "ses=" <<ses <<", updating particles" << std::endl;

            // std::cout << "ses=" << ses << ", resampledIndices = ";
            // for (auto const& i : resampledIndices)
            //     std::cout << i << ", ";
            // std::cout << "\n" ;

            for(int j=0; j<N; j++)
            {
                particleFilterVec[j].backUpChosenStrategies();
                particleFilterVec[j].backUpStratCredits();
            }
           
            auto resample_particle_filter = [&particleFilterVec,&ses](int i, std::vector<double> resampledIndices_) {
                int newIndex = resampledIndices_[i];
                //ParticleFilter pf(particleFilterVec[newIndex]);

                // Use lock_guard to ensure mutual exclusion for particleFilterVec updates
                std::vector<std::pair<std::vector<double>,std::vector<double>>>& stratBackUps = particleFilterVec[newIndex].getStratCreditBackUps();
                std::vector<signed int> & chosenStrategyBackUp =  particleFilterVec[newIndex].getChosenStrategyBackups();
                int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
                particleFilterVec[i].setChosenStrategies(chosenStrategyBackUp);
                particleFilterVec[i].setStratBackups(stratBackUps);
                int updatedChosenStrat = particleFilterVec[i].getChosenStratgies()[ses];

                // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[newIndex].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;
            };


            for (int i = 0; i < N; i++) {
                auto bound_lambda = std::bind(resample_particle_filter, i,resampledIndices);
                std::future<void> my_future = pool.submit_task(bound_lambda);

            }

            pool.wait();

            // std::cout << "ses=" <<ses <<", updating particles completed." << std::endl;
            double initWeight = (1.0 / (double)N);
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

std::vector<std::vector<double>> w_smoothed(std::vector<std::vector<double>> filteredWeights, std::vector<ParticleFilter>& particleFilterVec, int N,BS::thread_pool& pool)
{
    std::cout << "Inside w_smoothed" << std::endl;
    int n_rows = filteredWeights.size();
    // w_smoothed[t][i]

    std::vector<std::vector<double>> v(n_rows, std::vector<double>(N, 0.0));

    for (int t = 0; t < n_rows - 1; t++) {
        // for (int k = 0; k < N; k++) {
        //     auto bound_lambda = std::bind(compute_v_element, t, k);
        //     std::future<void> my_future = pool.submit_task(bound_lambda);

        // }

        BS::multi_future<void> loop_future = pool.submit_sequence(0,N,[&t,&particleFilterVec,&filteredWeights, &N, &v](int k)
        {
            for (int j = 0; j < N; j++) {
                std::vector<std::vector<double>> crpPriors_j = particleFilterVec[j].getCrpPriors();
                std::vector<double> crpPriors_j_t = crpPriors_j[t];

                std::vector<int> assignments_k = particleFilterVec[k].getOriginalSampledStrats();
                int X_k_tplus1 = assignments_k[t + 1];

                double contribution = filteredWeights[t][j] * crpPriors_j_t[X_k_tplus1];

                // Use lock_guard to ensure mutual exclusion for v[t][k] updates

                v[t][k] += contribution;

                if (std::isnan(v[t][k])) {
                    throw std::runtime_error("v[t][k] is nan. Check");
                }
                if (std::isinf(v[t][k])) {
                    throw std::runtime_error("v[t][k] is infinity. Check");
                }
                
            }
            if (v[t][k] == 0) {
                v[t][k] = 1e-6;
            }
        });

        loop_future.wait();
        
    }



    // // Wait for all tasks to complete
    // for (auto& future : futures) {
    //     future.wait();
    // }


    // std::cout << "v:";
    // for (const auto& row : v) {
    //     for (const auto& element : row) {
    //         std::cout << element << " ";
    //     }
    //     std::cout << std::endl;  // Start a new line for each row
    // }

    std::vector<std::vector<double>> smoothedWeights(n_rows, std::vector<double>(N, 0.0));

    auto compute_smoothed_weight = [&particleFilterVec,&filteredWeights,&n_rows,&smoothedWeights,&N,&v](int t, int i) {
        if (t == n_rows - 1) {
            smoothedWeights[t][i] = filteredWeights[n_rows - 1][i];
        } else {
            std::vector<std::vector<double>> crpPriors_i = particleFilterVec[i].getCrpPriors();
            std::vector<double> crpPriors_i_t = crpPriors_i[t];

            // std::cout << "ses=" << t << ", i=" << i << ", crpPriors_i_t = ";
            // for (auto const& i : crpPriors_i_t)
            //     std::cout << i << ", ";
            // std::cout << "\n" ;

            double weightedSum = 0;
            for (int k = 0; k < N; k++) {
                std::vector<int> assignments_k = particleFilterVec[k].getOriginalSampledStrats();
                int X_k_tplus1 = assignments_k[t + 1];
                weightedSum += (smoothedWeights[t + 1][k] * crpPriors_i_t[X_k_tplus1] / v[t][k]);
                // Add appropriate error checking here if needed
                // if(k==N-1 && weightedSum==0)
                // {
                //     std::cout << "ses=" << t << ", X_k_tplus1 = ";
                //     for (auto const& i : origSampledStrat_tplus)
                //         std::cout << i << ", ";
                //     std::cout << "\n" ;


                // }

            }
            smoothedWeights[t][i] = weightedSum * filteredWeights[t][i];
            if(std::isnan(smoothedWeights[t][i]))
            {
                std::cout << "v[t]=";
                for (auto const& y : v[t])
                    std::cout << y << ", ";
                std::cout << "\n" ;

                throw std::runtime_error("smoothedWeights[t][i] is nan");
            }
        }
    };

    // std::vector<std::thread> threads_smoothedWeights;

    for (int t = n_rows - 1; t >= 0; t--) {
        for (int i = 0; i < N; i++) {
            auto bound_lambda = std::bind(compute_smoothed_weight, t, i);
            std::future<void> my_future = pool.submit_task(bound_lambda);
        }

        pool.wait();

        if(std::accumulate(smoothedWeights[t].begin(), smoothedWeights[t].end(), 0.0)==0)
        {
            std::cout << "smoothedWeights[t], t=" << t;
            for (auto const& y : smoothedWeights[t])
                        std::cout << y << ", ";
                    std::cout << "\n" ;


            for(int j=0;j<N;j++)
            {
                std::vector<std::vector<double>> crpPriors_j = particleFilterVec[j].getCrpPriors();
                std::vector<double> crpPriors_j_t = crpPriors_j[t];
                std::cout << "ses=" << t << ", j=" << j << ", crpPriors_j_t = ";
                for (auto const& i : crpPriors_j_t)
                    std::cout << i << ", ";
                std::cout << "\n" ;

                double weightedSum = 0;
                for (int k = 0; k < N; k++) {
                    std::vector<int> assignments_k = particleFilterVec[k].getOriginalSampledStrats();
                    int X_k_tplus1 = assignments_k[t + 1];
                    weightedSum += (smoothedWeights[t + 1][k] * crpPriors_j_t[X_k_tplus1] / v[t][k]);
                    // Add appropriate error checking here if needed
                    std::cout << "t=" << t << ", k=" <<k << ", X_k_tplus1=" << X_k_tplus1 << ", smoothedWeights[t + 1][k]=" << smoothedWeights[t + 1][k] << ", crpPriors_j_t[X_k_tplus1]=" << crpPriors_j_t[X_k_tplus1]  << std::endl;
                }
            }
        

            std::string err = "smoothedWeights[t] vec is zero for t=" + t;        
            throw std::runtime_error(err);
        }

        bool containsNaN = false;
        for (const auto& element : smoothedWeights[t]) {
            if (std::isnan(element)) {
                containsNaN = true;
                break;  // No need to continue checking once a NaN is found
            }
        }

        if(containsNaN)
        {
            std::cout << "v[t], t=" << t;
            for (auto const& y : v[t])
                std::cout << y << ", ";
            std::cout << "\n" ;
            break;
        }



    }
    
    return (smoothedWeights);
}

std::vector<std::vector<std::vector<double>>> wij_smoothed(std::vector<std::vector<double>> filteredWeights, std::vector<ParticleFilter>& particleFilterVec, std::vector<std::vector<double>> w_smoothed, int N,BS::thread_pool& pool)
{
    // std::cout << "Inside wij_smoothed" << std::endl;
    int n_rows = filteredWeights.size();
    std::vector<std::vector<std::vector<double>>> wijSmoothed(n_rows - 2, std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0)));

    std::vector<std::vector<double>> v(n_rows, std::vector<double>(N, 0.0));

    auto compute_v_element = [&particleFilterVec,&filteredWeights,&N,&v](int t, int j) {
        for (int l = 0; l < N; l++) {
            double w_t_l = filteredWeights[t][l];
            std::vector<std::vector<double>> crpPriors_l = particleFilterVec[l].getCrpPriors();
            std::vector<int> assignments_j = particleFilterVec[j].getOriginalSampledStrats();
            int x_j_tplus1 = assignments_j[t + 1];

            std::vector<double> crpPriors_l_t = crpPriors_l[t];
            v[t][j] += crpPriors_l_t[x_j_tplus1] * w_t_l;
        }
        if (v[t][j] == 0) {
            v[t][j] = 1e-6;
        }
    };



    for (int t = 0; t < n_rows - 2; t++) {
        for (int j = 0; j < N; j++) {
            auto bound_lambda = std::bind(compute_v_element, t, j);
            std::future<void> my_future = pool.submit_task(bound_lambda);

        }
    }
    pool.wait();
    // // Wait for all tasks to complete
    // for (auto& future : futures) {
    //     future.wait();
    // }


    auto compute_ij_element = [&filteredWeights,&particleFilterVec,&w_smoothed,&wijSmoothed,&v](int t, int i, int j) {
        double w_t_i = filteredWeights[t][i];
        double w_smoothed_tplus1_j = w_smoothed[t + 1][j];

        std::vector<double> crpPriors_i_t = particleFilterVec[i].getCrpPriors()[t];
        int x_j_tplus1 = particleFilterVec[j].getOriginalSampledStrats()[t + 1];

        wijSmoothed[t][i][j] = w_t_i * w_smoothed_tplus1_j * crpPriors_i_t[x_j_tplus1] / v[t][j];
    };

    for (int t = 0; t < n_rows - 2; t++)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                auto bound_lambda = std::bind(compute_ij_element, t, i, j);
                std::future<void> my_future = pool.submit_task(bound_lambda);
            }
        }
    }
    pool.wait();

    return (wijSmoothed);
}

std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>, std::vector<std::vector<double>>> E_step(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::vector<double> params, BS::thread_pool& pool)
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

    auto [filteredWeights, loglik, crpPriors] = particle_filter_new(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, pool);
    std::vector<std::vector<double>> smoothedWeights = w_smoothed(filteredWeights, particleFilterVec, N, pool);
    // wijSmoothed[t][i,j]
    std::vector<std::vector<std::vector<double>>> wijSmoothed = wij_smoothed(filteredWeights, particleFilterVec, smoothedWeights, N,pool);

    return std::make_tuple(smoothedWeights, wijSmoothed, particleFilterVec, filteredWeights);
}

double M_step(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> smoothed_w, std::vector<double> params, BS::thread_pool& pool)
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


    for (int t = 0; t < sessions; t++) {
        
        BS::multi_future<double> loop_future = pool.submit_sequence<double>(0,N,[&t,&particleFilterVec,&smoothedWeights](int i)
        {
            double local_I_3 = 0.0;

            
            std::vector<int> originalSampledStrats = particleFilterVec[i].getOriginalSampledStrats();
            // std::cout << "ses=" << t << ", i=" << i << ", originalSampledStrats=" << originalSampledStrats[t] << std::endl;

            double lik_i = particleFilterVec[i].getSesLikelihood(originalSampledStrats[t], t);

            // std::cout << "ses=" << t << ", i=" << i << ", originalSampledStrats=" << originalSampledStrats[t] << ", lik_i=" << lik_i << std::endl;

            if (lik_i == 0) {
                lik_i = 1e-6;
            }

            local_I_3 += (smoothedWeights[t][i] * log(lik_i));

            if (std::isnan(local_I_3) || std::isinf(local_I_3)) {
                std::cout << "ses=" << t<< ", particle=" << i << ", lik_i=" << lik_i << std::endl;
                std::cout << "particle original history:";
                for (auto const& n : originalSampledStrats)
                    std::cout << n << ", ";
                std::cout << "\n" ;

                // std::cout << "params:";
                // for (auto const& n : params)
                //     std::cout << n << ", ";
                // std::cout << "\n" ;

                throw std::runtime_error("Error in local_I_3 value");
            }
            
            return local_I_3;
        });
        loop_future.wait();
        for (auto& future : loop_future) {
           I_3+= future.get();
        }

    }

    // // Wait for all tasks to complete
    // for (auto& future : futures) {
    //     future.wait();
    // }

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

    // std::cout << "I_1=" << I_1 << ", I_3=" << I_3 << std::endl;

    return (I_1 + I_2 + I_3);
}

// std::vector<double> EM(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, BS::thread_pool& pool)
// {
//     double finalChamp = 1e8;
//     std::vector<double> paramVec;
//     for(int k = 0; k< 5; k++)
//     {
//         std::pair<std::vector<double>, std::vector<double>> bounds = {
//         {0.1, 0.7, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8, 1e-8},
//         {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}};

//         // Initialize a random number generator
//         std::random_device rd;
//         std::mt19937 gen(rd());
//         std::uniform_real_distribution<double> dist;

//         // Generate a vector of 8 doubles within the specified range
//         std::vector<double> params;
//         for (int i = 0; i < 8; ++i) {
//             double randomValue = bounds.first[i] + dist(gen) * (bounds.second[i] - bounds.first[i]);
//             params.push_back(randomValue);
//         }

//         std::cout << "k=" <<k << ", params: ";
//         for (const auto &x : params)
//         {
//             std::cout << x << " ";
//         }
//         std::cout << "\n";


//         std::vector<double> QFuncVals;
//         double prevChampion = 1e6;
//         double champion = 1e8;
//         bool terminate = false;

//         for (int i = 0; i < 10; i++)
//         {

//             std::cout << "k=" <<k <<  ", i=" << i << ", E-step" << std::endl;
//             std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>, std::vector<std::vector<double>>> res = E_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, params, pool);
//             std::vector<std::vector<double>> smoothedWeights = std::get<0>(res);
//             std::vector<std::vector<std::vector<double>>> wijSmoothed = std::get<1>(res);

//             // std::cout << "smoothedWeights:";
//             // for (const auto& row : smoothedWeights) {
//             //     for (const auto& element : row) {
//             //         std::cout << element << " ";
//             //     }
//             //     std::cout << std::endl;  // Start a new line for each row
//             // }

//             // std::vector<std::vector<double>> wijSmoothed_0 = wijSmoothed[0];
//             // std::cout << "wijSmoothed[0]:";
//             // for (const auto& row : wijSmoothed_0) {
//             //     for (const auto& element : row) {
//             //         std::cout << element << " ";
//             //     }
//             //     std::cout << std::endl;  // Start a new line for each row
//             // }

//             // std::vector<std::vector<double>> wijSmoothed_5 = wijSmoothed[5];
//             // std::cout << "wijSmoothed[5]:";
//             // for (const auto& row : wijSmoothed_5) {
//             //     for (const auto& element : row) {
//             //         std::cout << element << " ";
//             //     }
//             //     std::cout << std::endl;  // Start a new line for each row
//             // }

//             std::cout << "i=" << i << ", starting M-step" << std::endl;
//             std::vector<ParticleFilter> particleFilterVec =  std::get<2>(res);
//             std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> resTuple = std::make_tuple(smoothedWeights, wijSmoothed, particleFilterVec);
//             PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, resTuple, pool);
//             // PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
//             std::cout << "Initialized problem class" << std::endl;

//             // Create a problem using Pagmo
//             problem prob{pagmoprob};
//             int count = 0;
//             std::vector<double> dec_vec_champion;
//             while (prevChampion - champion <= 0)
//             {
//                 pagmo::nlopt method("sbplx");
//                 method.set_maxeval(20);
//                 pagmo::algorithm algo = pagmo::algorithm{method};
//                 pagmo::population pop(prob, 10);
//                 pop = algo.evolve(pop);

//                 dec_vec_champion = pop.champion_x();
//                 champion = pop.champion_f()[0];
//                 std::cout << "Final champion = " << champion << std::endl;

//                 std::cout << "dec_vec_champion: ";
//                 for (const auto &x : dec_vec_champion)
//                 {
//                     std::cout << x << " ";
//                 }
//                 std::cout << "\n";
//                 count++;
//                 if (count == 2)
//                 {
//                     std::cout << "Exiting, not able to improve Qfunction" << std::endl;
//                     terminate = true;
//                     break;
//                 }
//             }
//             if (terminate)
//             {
//                 break;
//             }
//             params = dec_vec_champion;
//             std::pair<std::vector<std::vector<double>>, double> q = particle_filter(N, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params,pool);
//             std::cout << "loglikelihood=" << q.second << std::endl;
//             QFuncVals.push_back(q.second);

//             if(champion<finalChamp)
//             {
//                 paramVec = params;
//             }

//             prevChampion = champion;
//             champion = 1e8;
//         }

//         std::cout << "Likelihoods=";
//         for (auto const &i : QFuncVals)
//             std::cout << i << ", ";
//         std::cout << "\n";

//     }

//     return (paramVec);
// }


std::vector<double> EM(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, BS::thread_pool& pool)
{
    std::vector<double> params = {228439, 0.921127, 0.0429102, 0.575078};
    std::vector<double> QFuncVals;

    for (int i = 0; i < 10; i++)
    {

        std::cout << "i=" << i << ", E-step" << std::endl;
        auto [smoothedWeights, wijSmoothed, particleFilterVec, filteredWeights] = E_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, params, pool);

        std::cout << "i=" << i << ", starting M-step" << std::endl;
        PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, std::make_tuple(smoothedWeights, wijSmoothed, particleFilterVec), pool);
        // PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, resTuple, pool);
        // PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
        std::cout << "Initialized problem class" << std::endl;

        // Create a problem using Pagmo
        problem prob{pagmoprob};
        int count = 0;
        
        // pagmo::nlopt method("sbplx");
        pagmo::de method;
        //method.set_maxeval(20);
        pagmo::algorithm algo = pagmo::algorithm{method};
        pagmo::population pop(prob, 10);
        pop = algo.evolve(pop);

        std::vector<double> dec_vec_champion = pop.champion_x();
        double champion = pop.champion_f()[0];
        std::cout << "Final champion = " << champion << std::endl;

        std::cout << "dec_vec_champion: ";
        for (const auto &x : dec_vec_champion)
        {
            std::cout << x << " ";
        }
        std::cout << "\n";
        params = dec_vec_champion;
        // std::pair<std::vector<std::vector<double>>, double> q = particle_filter(N, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params,pool);
        // std::cout << "loglikelihood=" << q.second << std::endl;
        // QFuncVals.push_back(q.second);

    }

    // std::cout << "Likelihoods=";
    // for (auto const &i : QFuncVals)
    //     std::cout << i << ", ";
    // std::cout << "\n";


    return (params);
}


std::vector<double> Mle(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N,BS::thread_pool& pool)
{
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>, std::vector<std::vector<double>>> res;
    std::vector<std::vector<double>> smoothedWeights = std::get<0>(res);
    std::vector<std::vector<std::vector<double>>> wijSmoothed = std::get<1>(res);
    std::vector<ParticleFilter> particleFilterVec =  std::get<2>(res);
    std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> resTuple = std::make_tuple(smoothedWeights, wijSmoothed, particleFilterVec);
    PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, resTuple, pool);

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
    // //method.set_maxeval(10);
    // pagmo::algorithm algo = pagmo::algorithm {method};
    // pagmo::population pop(prob, 20);
    // pop = algo.evolve(pop);

        pagmo::de method (5);
        //method.set_maxeval(10);
        pagmo::algorithm algo = pagmo::algorithm {method};
        pagmo::population pop(prob, 20);
        for(int k=0; k<3;k++)
        {
            pop = algo.evolve(pop);
        }
        

    std::vector<double> dec_vec_champion = pop.champion_x();
    std::cout << "Final champion = " << pop.champion_f()[0] << std::endl;

    // std::cout << "dec_vec_champion: ";
    // for (const auto &x : dec_vec_champion) {
    //     std::cout << x << " ";
    // }
    // std::cout << "\n";


    // double champion_score = 1e8;
    // std::vector<double> dec_vec_champion;
    // for (const auto &isl : archi)
    // {
    //     std::vector<double> dec_vec = isl.get_population().champion_x();

    //     // std::cout << "champion:" <<isl.get_population().champion_f()[0] << '\n';
    //     // for (auto const& i : dec_vec)
    //     //     std::cout << i << ", ";
    //     // std::cout << "\n" ;

    //     double champion_isl = isl.get_population().champion_f()[0];
    //     if (champion_isl < champion_score)
    //     {
    //         champion_score = champion_isl;
    //         dec_vec_champion = dec_vec;
    //     }
    // }

    // std::cout << "Final champion = " << champion_score << std::endl;
    for (auto const &i : dec_vec_champion)
        std::cout << i << ", ";
    std::cout << "\n";

    // const auto fv = prob.fitness(dec_vec_champion);
    // std::cout << "Value of the objfun in dec_vec_champion: " << fv[0] << '\n';

    std::vector<double> params = dec_vec_champion;
    std::pair<std::vector<std::vector<double>>, double> q = particle_filter(N, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params,pool);
    std::cout << "loglikelihood=" << q.second << std::endl;

    return (params);
}


void testQFunc(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, BS::thread_pool& pool, RInside & R)
{

    unsigned int numThreads = std::thread::hardware_concurrency();
    std::vector<double> params = {0.215106, 0.872738, 0.0288027, 0.544798};


    // Print the result
    std::cout << "Number of threads available: " << numThreads << "\n";
    for(int i=0; i<5; i++)
    {
        
    //     {
    //         std::cout << "i=" <<i << ", performing E-step" << std::endl;
    //         auto [smoothedWeights, wijSmoothed, particleFilterVec, filteredWeights] = E_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, params, pool);

    //         // std::vector<std::vector<double>> smoothedWeights = std::get<0>(res);
    //         // std::vector<ParticleFilter> particleFilterVec = std::get<2>(res);
    //         // std::vector<std::vector<double>> filteredWeights = std::get<3>(res);

    //         // std::vector<std::vector<std::vector<double>>> wijSmoothed = std::get<1>(res);
    //         // std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> resTuple = std::make_tuple(smoothedWeights, wijSmoothed, particleFilterVec);


    //         R["smoothedWeights"] = smoothedWeights;
    //         std::string file = "smoothedWeights_"+std::to_string(i)+".Rdata";
    //         std::string rCode = "save(smoothedWeights, file='" + file + "')";
    //         R.parseEvalQ(rCode.c_str());

    //         Rcpp::List crpPriorList;

    //         std::vector<std::vector<int>> chosenStrats;
    //         std::vector<std::vector<int>> origChosenStrats;
    //         std::vector<std::vector<double>> likelihoods;

    //         for(int k=0; k<particleFilterVec.size();k++)
    //         {
    //             std::vector<std::vector<double>> crpPriors_k = particleFilterVec[k].getCrpPriors();
    //             // Rcpp::DataFrame df_crpPriors_k = Rcpp::wrap(crpPriors_k);
    //             crpPriorList.push_back(crpPriors_k);

    //             std::vector<int> chosenStrats_k =  particleFilterVec[k].getChosenStratgies();  
    //             chosenStrats.push_back(chosenStrats_k);

    //             std::vector<int> origChosenStrats_k =  particleFilterVec[k].getOriginalSampledStrats();  
    //             origChosenStrats.push_back(origChosenStrats_k); 


    //             std::vector<double> lik_k =  particleFilterVec[k].getLikelihoods();  
    //             likelihoods.push_back(lik_k); 
    //         }

    //         R["crpPriorList"] = crpPriorList;
    //         std::string file2 = "crpPriorList_"+std::to_string(i)+".Rdata";
    //         std::string rCode2 = "save(crpPriorList, file='" + file2 + "')";
    //         R.parseEvalQ(rCode2.c_str());

    //         R["filteredWeights"] = filteredWeights;
    //         std::string file3 = "filteredWeights_"+std::to_string(i)+".Rdata";
    //         std::string rCode3 = "save(filteredWeights, file='" + file3 + "')";
    //         R.parseEvalQ(rCode3.c_str());

    //         R["chosenStrats"] = chosenStrats;
    //         std::string file4 = "chosenStrats_"+std::to_string(i)+".Rdata";
    //         std::string rCode4 = "save(chosenStrats, file='" + file4 + "')";
    //         R.parseEvalQ(rCode4.c_str());

    //         R["likelihoods"] = likelihoods;
    //         std::string file5 = "likelihoods_"+std::to_string(i)+".Rdata";
    //         std::string rCode5 = "save(likelihoods, file='" + file5 + "')";
    //         R.parseEvalQ(rCode5.c_str());

    //         R["origChosenStrats"] = origChosenStrats;
    //         std::string file6 = "origChosenStrats_"+std::to_string(i)+".Rdata";
    //         std::string rCode6 = "save(origChosenStrats, file='" + file6 + "')";
    //         R.parseEvalQ(rCode6.c_str());
    //         double Q = M_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, std::make_tuple(smoothedWeights, wijSmoothed, particleFilterVec), params, pool);
    //         std::cout << "i=" <<i << ", Q=" << Q << std::endl;


    //     }



        std::pair<std::vector<std::vector<double>>, double> q = particle_filter(N, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params,pool);
        std::cout << "loglikelihood=" << q.second << std::endl;

    }


}