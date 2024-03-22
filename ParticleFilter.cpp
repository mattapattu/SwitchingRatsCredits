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
#include <pagmo/algorithm.hpp>
#include <pagmo/archipelago.hpp>
#include <pagmo/algorithms/nlopt.hpp>
#include <pagmo/algorithms/sade.hpp>
#include <stdexcept>




using namespace std;



// A function that returns a random integer in the range [0, n-1]
int randint()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 1);
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

void normalizeRow(std::vector<std::vector<double>>& matrix, size_t t) {
    if (t < matrix.size() && !matrix[t].empty()) {
        // Get the sum of elements in the t-th row
        double rowSum = 0.0;
        for (double element : matrix[t]) {
            rowSum += element;
        }

        // Normalize the t-th row
        for (double& element : matrix[t]) {
            element /= rowSum;
        }
    } else {
        std::cerr << "Invalid row index or empty row." << std::endl;
    }
}


void normalize(std::vector<double> &p)
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


std::tuple<std::vector<std::vector<double>>, double, std::vector<int>> particle_filter(int N, const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, std::vector<double> v,BS::thread_pool& pool)
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

    std::vector<std::vector<double>> delta(sessions, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> phi(sessions, std::vector<double>(N, 0.0));

    std::vector<ParticleFilter> particleFilterVec;
    for (int i = 0; i < N; i++)
    {
        auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, v, i, 1.0);
        particleFilterVec.push_back(pf);
        // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
    }



    double loglik = 0;
    // Iterate over the trials
    for (int ses = 0; ses < sessions; ses++)
    {
        //   // Initialize the weights with uniform probabilities
        double ses_lik = 0;
        // std::cout << "ses=" << ses << std::endl;

        if(ses==0)
        {
            BS::multi_future<void> initPf = pool.submit_sequence(0,N,[&particleFilterVec,&delta,&phi,&ses,&w](int i) {

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

                    throw std::runtime_error("Err in generate_sampling_distribution. Check");
                }
                
                particleFilterVec[i].rollBackCredits();

                particleFilterVec[i].addCrpPrior(p,ses);

                // std::cout <<"i=" <<i << ", ses=" << ses << ", p=";
                // for (auto const& n : p)
                //     std::cout << n << ", ";
                // std::cout << "\n" ;


                int sampled_strat = sample(p);
                if(p[sampled_strat] == 0)
                {
                    std::cout << "Sample not matching distribution. Check" << std::endl;
                    std::cout << "ses=" << ses << ", sample=" << sampled_strat;
                    for (auto const& n : p)
                        std::cout << n << ", ";
                    std::cout << "\n" ;

                    throw std::runtime_error("Sample not matching distribution. Check");
                }
                
                // // std::cout << "particleId=" << i <<  ", ses=" <<ses << ", particleHistory:";
                // //  for (auto const& i : particleHistory)
                // //     std::cout << i << ", ";
                // // std::cout << "\n" ; 
                particleFilterVec[i].addAssignment(ses, sampled_strat);
                particleFilterVec[i].addOriginalSampledStrat(ses, sampled_strat);
                std::vector<int> particleHistory = particleFilterVec[i].getChosenStratgies();
                particleFilterVec[i].addParticleTrajectory(particleHistory, ses);

                double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses);

                // double loglik = particleFilterVec[i].getSesLogLikelihood(sampled_strat, ses);
                // particleFilterVec[i].setLikelihood(lik);
                w[i] *= lik;
                // // log_w[i] += loglik;

                delta[ses][i] = log(0.25)+log(lik);

                particleFilterVec[i].backUpStratCredits(); 

            });

            initPf.wait();

            normalize(w);

            filteredWeights.push_back(w);

            continue;
        } // End init particles

        // Resample particles from t=1 to T-1
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

            // std::cout << "ses=" << ses << ", resampledIndices = ";
            // for (auto const& i : resampledIndices)
            //     std::cout << i << ", ";
            // std::cout << "\n" ;

            for(int j=0; j<N; j++)
            {
                particleFilterVec[j].backUpChosenStrategies();
                particleFilterVec[j].backUpStratCredits();
            }
           
            BS::multi_future<void> resample_particle_filter = pool.submit_sequence(0,N,[&particleFilterVec,&ses,&resampledIndices](int i) {
                int newIndex = resampledIndices[i];
                //ParticleFilter pf(particleFilterVec[newIndex]);

                // Use lock_guard to ensure mutual exclusion for particleFilterVec updates
                std::vector<std::pair<std::vector<double>,std::vector<double>>>& stratBackUps = particleFilterVec[newIndex].getStratCreditBackUps();
                std::vector<signed int> chosenStrategyBackUp =  particleFilterVec[newIndex].getChosenStrategyBackups();
                int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
                particleFilterVec[i].setChosenStrategies(chosenStrategyBackUp);
                particleFilterVec[i].setStratBackups(stratBackUps);
                int updatedChosenStrat = particleFilterVec[i].getChosenStratgies()[ses];

                // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[newIndex].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;
            });

            resample_particle_filter.wait();


            // std::cout << "ses=" <<ses <<", updating particles completed." << std::endl;
            double initWeight = (1.0 / (double)N);
            std::fill(w.begin(), w.end(), initWeight);
            
        }


        std::vector<double> lik_ses(N,0);
         BS::multi_future<void> propogate = pool.submit_sequence(0,N,[&particleFilterVec,&lik_ses,&ses,&w](int i) {

            // particleFilterVec[i].updateStratCounts(ses);
            // std::vector<double> crp_i = particleFilterVec[i].crpPrior(ses);
            std::vector<int> particleHistory_t_minus1 = particleFilterVec[i].getChosenStratgies();
            std::vector<double> crp_i;
            try
            {
                crp_i = particleFilterVec[i].crpPrior2(particleHistory_t_minus1,ses-1);
            }
            catch(const std::exception& e)
            {
                std::cout << "Err in propogate in PF" << std::endl;
                std::cerr << e.what() << '\n';
            }

            // std::cout << "i=" <<i << ", ses=" << ses << ", crp=";
            // for (auto const& n : crp_i)
            //     std::cout << n << ", ";
            // std::cout << "\n" ;

            
            

            // std::cout << "particleId=" << i <<  ", ses=" <<ses << ", particleHistory_t:";
            //  for (auto const& i : particleHistory_t)
            //     std::cout << i << ", ";
            // std::cout << "\n" ; 

            // std::cout << "particleId=" << i <<  ", ses=" <<ses << ", crp_i:";
            //  for (auto const& i : crp_i)
            //     std::cout << i << ", ";
            // std::cout << "\n" ; 



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

                std::cout << "particleId=" << i <<  ", ses=" <<ses << ", particleHistory_t_minus1:";
                for (auto const& i : particleHistory_t_minus1)
                    std::cout << i << ", ";
                std::cout << "\n" ; 


                

                throw std::runtime_error("Err in generate_sampling_distribution. Check");
            }
            
            particleFilterVec[i].rollBackCredits();

            // std::cout << "i=" <<i << ", ses=" << ses << ", p=";
            // for (auto const& n : p)
            //     std::cout << n << ", ";
            // std::cout << "\n" ;


            particleFilterVec[i].addCrpPrior(p,ses);

            int sampled_strat = sample(p);
            if(p[sampled_strat] == 0)
            {
                std::cout << "Sample not matching distribution. Check" << std::endl;
                std::cout << "ses=" << ses << ", sample=" << sampled_strat;
                for (auto const& n : p)
                    std::cout << n << ", ";
                std::cout << "\n" ;

                throw std::runtime_error("Sample not matching distribution. Check");
            }
            
            particleFilterVec[i].addAssignment(ses, sampled_strat);
            particleFilterVec[i].addOriginalSampledStrat(ses, sampled_strat);
            double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses);
            lik_ses[i] = lik;
            std::vector<int> particleHistory_t = particleFilterVec[i].getChosenStratgies();
            particleFilterVec[i].addParticleTrajectory(particleHistory_t, ses);


            // // double loglik = particleFilterVec[i].getSesLogLikelihood(sampled_strat, ses);
            // particleFilterVec[i].setLikelihood(lik);
            w[i] *= lik;
            // // log_w[i] += loglik;
            
            

            particleFilterVec[i].backUpStratCredits(); 

        });

        propogate.wait();

        for(int j=0; j<N; j++)
        {
            int x_t_j = particleFilterVec[j].getChosenStratgies()[ses];
            double score = -1e-6;
            double maxScore = -1e-6;
            int maxidx = 0;
            for(int k=0; k<N;k++)
            {
                std::vector<double> crp_j_t = particleFilterVec[k].getCrpPriors()[ses];
                score = delta[ses-1][k] + log(crp_j_t[x_t_j]);
                if(score > maxScore)
                {
                    maxScore = score;
                    maxidx = k;
                }
            }
            delta[ses][j]  = log(lik_ses[j])+ maxScore;
            phi[ses][j] = maxidx;
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

    }

    std::vector<int> viterbiSeq(sessions,0);
     
    auto finalMaxIndex = std::distance(delta[sessions-1].begin(), std::max_element(delta[sessions-1].begin(), delta[sessions-1].end()));
    viterbiSeq[sessions-1] = particleFilterVec[finalMaxIndex].getOriginalSampledStrats()[sessions-1];
    for(int t=sessions-2; t>=0; t--)
    {
        int maxIndex = phi[t+1][finalMaxIndex];
        viterbiSeq[t] = particleFilterVec[maxIndex].getOriginalSampledStrats()[t];
    }

    std::vector<std::vector<double>> filteringDist(4,std::vector<double>(sessions));
    for (int ses = 0; ses < sessions; ses++)
    {
        for (int i = 0; i < N; i++)
        {
            std::vector<int> chosenStrategy_pf = particleFilterVec[i].getOriginalSampledStrats();

            // std::cout << "ses=" <<ses << ", particleId=" <<i << ", chosenStrat=" << chosenStrategy_pf[ses] << std::endl;
            filteringDist[chosenStrategy_pf[ses]][ses] = filteringDist[chosenStrategy_pf[ses]][ses] + filteredWeights[ses][i];
            // postProbsOfExperts[ses][chosenStrategy_pf[ses]] = std::round(postProbsOfExperts[ses][chosenStrategy_pf[ses]] * 100.0) / 100.0;
        }
    }

      
    // Normalize the posterior probabilities
    // normalizeRows(q);

    std::cout << "posterior probs=" << std::endl;
    for (const auto &row : filteringDist)
    {
        for (double num : row)
        {
            std::cout << std::fixed << std::setprecision(2) << num << " ";
        }
        std::cout << std::endl;
    }




    // Compute the posterior probabilities for each expert by counting the occurrences of each expert in the last trial
    // vector<vector<double>> postProbsOfExperts(sessions, vector<double>(4) );

    // Return the posterior probabilities
    return std::make_tuple(filteredWeights, loglik, viterbiSeq);
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

    std::vector<std::vector<double>> delta(sessions, std::vector<double>(N, 0.0));
    std::vector<std::vector<double>> phi(sessions, std::vector<double>(N, 0.0));


    double loglik = 0;
    // Iterate over the trials
    for (int ses = 0; ses < sessions; ses++)
    {
        //   // Initialize the weights with uniform probabilities
        double ses_lik = 0;
        // std::cout << "ses=" << ses << std::endl;

        if(ses==0)
        {
            BS::multi_future<void> initPf = pool.submit_sequence(0,N,[&particleFilterVec,&delta,&phi,&ses,&w](int i) {

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

                    throw std::runtime_error("Err in generate_sampling_distribution. Check");
                }
                
                particleFilterVec[i].rollBackCredits();

                particleFilterVec[i].addCrpPrior(p,ses);

                int sampled_strat = sample(p);
                if(p[sampled_strat] == 0)
                {
                    std::cout << "Sample not matching distribution. Check" << std::endl;
                    std::cout << "ses=" << ses << ", sample=" << sampled_strat;
                    for (auto const& n : p)
                        std::cout << n << ", ";
                    std::cout << "\n" ;

                    throw std::runtime_error("Sample not matching distribution. Check");
                }
                
                // // std::cout << "particleId=" << i <<  ", ses=" <<ses << ", particleHistory:";
                // //  for (auto const& i : particleHistory)
                // //     std::cout << i << ", ";
                // // std::cout << "\n" ; 
                particleFilterVec[i].addAssignment(ses, sampled_strat);
                particleFilterVec[i].addOriginalSampledStrat(ses, sampled_strat);
                std::vector<int> particleHistory = particleFilterVec[i].getChosenStratgies();
                particleFilterVec[i].addParticleTrajectory(particleHistory, ses);

                double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses);

                // double loglik = particleFilterVec[i].getSesLogLikelihood(sampled_strat, ses);
                // particleFilterVec[i].setLikelihood(lik);
                w[i] *= lik;
                // // log_w[i] += loglik;

                delta[ses][i] = log(0.25)+log(lik);

                particleFilterVec[i].backUpStratCredits(); 

            });

            initPf.wait();

            normalize(w);

            filteredWeights.push_back(w);

            continue;
        } // End init particles

        // Resample particles from t=1 to T-1
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

            // std::cout << "ses=" << ses << ", resampledIndices = ";
            // for (auto const& i : resampledIndices)
            //     std::cout << i << ", ";
            // std::cout << "\n" ;

            for(int j=0; j<N; j++)
            {
                particleFilterVec[j].backUpChosenStrategies();
                particleFilterVec[j].backUpStratCredits();
            }
           
            BS::multi_future<void> resample_particle_filter = pool.submit_sequence(0,N,[&particleFilterVec,&ses,&resampledIndices](int i) {
                int newIndex = resampledIndices[i];
                //ParticleFilter pf(particleFilterVec[newIndex]);

                // Use lock_guard to ensure mutual exclusion for particleFilterVec updates
                std::vector<std::pair<std::vector<double>,std::vector<double>>>& stratBackUps = particleFilterVec[newIndex].getStratCreditBackUps();
                std::vector<signed int> chosenStrategyBackUp =  particleFilterVec[newIndex].getChosenStrategyBackups();
                int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
                particleFilterVec[i].setChosenStrategies(chosenStrategyBackUp);
                particleFilterVec[i].setStratBackups(stratBackUps);
                int updatedChosenStrat = particleFilterVec[i].getChosenStratgies()[ses];

                // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[newIndex].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;
            });

            resample_particle_filter.wait();


            // std::cout << "ses=" <<ses <<", updating particles completed." << std::endl;
            double initWeight = (1.0 / (double)N);
            std::fill(w.begin(), w.end(), initWeight);
            
        }


        std::vector<double> lik_ses(N,0);
         BS::multi_future<void> propogate = pool.submit_sequence(0,N,[&particleFilterVec,&lik_ses,&ses,&w](int i) {

            // particleFilterVec[i].updateStratCounts(ses);
            // std::vector<double> crp_i = particleFilterVec[i].crpPrior(ses);
            std::vector<int> particleHistory_t_minus1 = particleFilterVec[i].getChosenStratgies();
            std::vector<double> crp_i;
            try
            {
                crp_i = particleFilterVec[i].crpPrior2(particleHistory_t_minus1,ses-1);

            }
            catch(const std::exception& e)
            {
                std::cout << "Err in propogate in PF" << std::endl;
                std::cerr << e.what() << '\n';
            }

            // std::cout << "particleId=" << i <<  ", ses=" <<ses << ", particleHistory_t:";
            //  for (auto const& i : particleHistory_t)
            //     std::cout << i << ", ";
            // std::cout << "\n" ; 

            // std::cout << "particleId=" << i <<  ", ses=" <<ses << ", crp_i:";
            //  for (auto const& i : crp_i)
            //     std::cout << i << ", ";
            // std::cout << "\n" ; 



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

                std::cout << "particleId=" << i <<  ", ses=" <<ses << ", particleHistory_t_minus1:";
                for (auto const& i : particleHistory_t_minus1)
                    std::cout << i << ", ";
                std::cout << "\n" ; 


                

                throw std::runtime_error("Err in generate_sampling_distribution. Check");
            }
            
            particleFilterVec[i].rollBackCredits();

            particleFilterVec[i].addCrpPrior(p,ses);

            int sampled_strat = sample(p);
            if(p[sampled_strat] == 0)
            {
                std::cout << "Sample not matching distribution. Check" << std::endl;
                std::cout << "ses=" << ses << ", sample=" << sampled_strat;
                for (auto const& n : p)
                    std::cout << n << ", ";
                std::cout << "\n" ;

                throw std::runtime_error("Sample not matching distribution. Check");
            }
            
            particleFilterVec[i].addAssignment(ses, sampled_strat);
            particleFilterVec[i].addOriginalSampledStrat(ses, sampled_strat);
            double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses);
            lik_ses[i] = lik;
            std::vector<int> particleHistory_t = particleFilterVec[i].getChosenStratgies();
            particleFilterVec[i].addParticleTrajectory(particleHistory_t, ses);


            // // double loglik = particleFilterVec[i].getSesLogLikelihood(sampled_strat, ses);
            // particleFilterVec[i].setLikelihood(lik);
            w[i] *= lik;
            // // log_w[i] += loglik;
            
            

            particleFilterVec[i].backUpStratCredits(); 

        });

        propogate.wait();


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



std::tuple<std::vector<std::vector<double>>, std::vector<ParticleFilter>, std::vector<std::vector<int>>> E_step2(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::vector<double> params, BS::thread_pool& pool)
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

    //auto [filteredWeights, loglik, crpPriors] = particle_filter_new(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, pool);
    auto [filteredWeights, loglik] = fapf(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, pool);
    std::vector<std::vector<double>> smoothedWeightMat(sessions, std::vector<double>(N, 0.0));

    std::vector<double> smoothedWeights_T = filteredWeights.back();
    std::vector<std::vector<int>> smoothedChoices(sessions, std::vector<int>(N, 0.0));

    int T = sessions-1;
    std::vector<double> resampledParticles = stratifiedResampling(smoothedWeights_T);
    // std::cout << "t=" << T << ", resampledParticles:";
    // for (auto const& n : resampledParticles)
    //     std::cout << n << ", ";
    // std::cout << "\n" ;

    for(int i=0; i<N; i++)
    {
        int resampledParticle =  resampledParticles[i];
        std::vector<int> assignments_resampled = particleFilterVec[resampledParticle].getOriginalSampledStrats();
        int resampled_X_T = assignments_resampled[T];
        smoothedChoices[T][i] = resampled_X_T; 
        smoothedWeightMat[T][i]  = filteredWeights[T][i];
    }
    
    for(int t=T-1; t>=0; t--)
    {
        // std::cout << "t=" << t << std::endl;
        for(int j=0; j<N; j++)
        {
            std::vector<double> crpPriors_j = particleFilterVec[j].getCrpPriors()[t+1];
            int X_tplus1_smoothed = smoothedChoices[t+1][j];
            smoothedWeightMat[t][j] = filteredWeights[t][j]*crpPriors_j[X_tplus1_smoothed]; 
        }
        normalizeRow(smoothedWeightMat, t);
        resampledParticles = stratifiedResampling(smoothedWeightMat[t]);
        for(int i=0; i<N; i++)
        {
            std::vector<int> assignments_resampled = particleFilterVec[resampledParticles[i]].getOriginalSampledStrats();
            int resampled_X_T = assignments_resampled[T];
            smoothedChoices[t][i] = resampled_X_T; 
        }
        
        // std::cout << "t=" << t << ", smoothedWeightMat[t,:]:";
        // for (auto const& n : smoothedWeightMat[t])
        //     std::cout << n << ", ";
        // std::cout << "\n" ;


        // std::cout << "t=" << t << ", smoothedChoices[t,:]:";
        // for (auto const& n : smoothedChoices[t])
        //     std::cout << n << ", ";
        // std::cout << "\n" ;

        
    }

    return std::make_tuple(smoothedWeightMat, particleFilterVec, smoothedChoices);


}


double M_step2(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::tuple<std::vector<std::vector<double>>, std::vector<ParticleFilter>, std::vector<std::vector<int>>> smoothedRes, std::vector<double> params, BS::thread_pool& pool)
{
    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    
    std::vector<std::vector<double>>& smoothedWeightMat = std::get<0>(smoothedRes);
    std::vector<ParticleFilter>& particleFilterVec = std::get<1>(smoothedRes);
    std::vector<std::vector<int>>& smoothedChoices = std::get<2>(smoothedRes);

    for (int i = 0; i < N; i++)
    {
        particleFilterVec[i].resetStrategies();
        // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
    }

    double I_1 = 0;
    double I_2 = 0;
    double I_3 = 0;


        
    BS::multi_future<double> loop_future = pool.submit_sequence<double>(0,N,[&particleFilterVec,&smoothedChoices,&sessions,&smoothedWeightMat](int i)
    {
        double local_I_3 = 0.0;
        for(int t=0; t<sessions;t++)
        {
                   
            // std::vector<int> originalSampledStrats = particleFilterVec[i].getOriginalSampledStrats();
            // std::cout << "ses=" << t << ", i=" << i << ", originalSampledStrats=" << originalSampledStrats[t] << std::endl;

            double lik_i = particleFilterVec[i].getSesLikelihood(smoothedChoices[t][i], t);

            //std::cout << "ses=" << t << ", i=" << i << ", smoothedChoices[t][i]=" << smoothedChoices[t][i] << ", lik_i=" << lik_i << std::endl;       

            if (lik_i == 0) {
                lik_i = 1e-6;
            }

            local_I_3 += (smoothedWeightMat[t][i] * log(lik_i));

            if (std::isnan(local_I_3) || std::isinf(local_I_3)) {
                
                throw std::runtime_error("Error in local_I_3 value");
            }
            
            
        }
        return local_I_3;
    });

    loop_future.wait();
    for (auto& future : loop_future) {
        I_3+= future.get();
    }

    for(int t=0; t<sessions-2; t++)
    {
        BS::multi_future<double> loop_future = pool.submit_sequence<double>(0,N,[&particleFilterVec, &smoothedWeightMat, &smoothedChoices, &t](int i)
        {
            double local_I_2 = 0.0;

            std::vector<double> crpPriors_i_t = particleFilterVec[i].getCrpPriors()[t];
            int X_i_tplus1 = smoothedChoices[t+1][i];

            double p_crp_X_i_tplus1 = crpPriors_i_t[X_i_tplus1];
            if(p_crp_X_i_tplus1==0)
            {
                p_crp_X_i_tplus1 = std::numeric_limits<double>::min();
            }


            local_I_2 = local_I_2 + (log(p_crp_X_i_tplus1)*smoothedWeightMat[t+1][i]);

            if (std::isnan(local_I_2)) {
                // std::cout << "t=" << t << ", i=" << i << ", j=" << j << ", w_ij_smoothed=" << w_ij_smoothed << ", X_j_tplus1=" << X_j_tplus1 << ", crpPriors_i_t[X_j_tplus1]=" << p_crp_X_j_tplus1 << std::endl;
                throw std::runtime_error("Error nan I_2 value");
            }

            if (std::isinf(local_I_2)) {
                throw std::runtime_error("Error inf I_2 value");
            }
            

            return local_I_2;

        });

        loop_future.wait();
        for (auto& future : loop_future) {
           I_2+= future.get();
        }

    }


    return (I_1+I_2+I_3);


}

std::vector<std::vector<int>> E_step3(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, int M, std::vector<double> params, BS::thread_pool& pool)
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

    // auto [filteredWeights, loglik, crpPriors] = particle_filter_new(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, pool);
    auto [filteredWeights, loglik] = fapf(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, pool);
    // std::vector<std::vector<std::vector<double>>> smoothedWeightMat(sessions, std::vector<double>(N, 0.0));
    std::cout << "fapf completed" << std::endl;
    
    std::vector<std::vector<int>> smoothedTrajectories(M, std::vector<int>(sessions, 0.0));

    int T = sessions-1;
    // std::cout << "t=" << T << ", resampledParticles:";
    // for (auto const& n : resampledParticles)
    //     std::cout << n << ", ";
    // std::cout << "\n" ;
            
   std::vector<double> smoothedWeights_T = filteredWeights.back();

    for(int j=0; j<M;j++)
    {
        std::vector<double> resampledParticles_j = systematicResampling(smoothedWeights_T);
        for(int i=0; i<N; i++)
        {
            int resampledParticle =  resampledParticles_j[i];
            // std::vector<std::vector<int>> particleTrajectories = particleFilterVec[i].getParticleTrajectories();
            // std::vector<int> particleHistory_t = particleTrajectories[T];
            std::vector<int> assignments_resampled = particleFilterVec[resampledParticle].getOriginalSampledStrats();
            // std::vector<int> assignments_resampled = particleHistory_t;

            // std::cout << "particleId=" << i << ", ses=" << T <<  ", particleHistory_t:";
            //  for (auto const& i : particleHistory_t)
            //     std::cout << i << ", ";
            // std::cout << "\n" ; 

            

            int resampled_X_T = assignments_resampled[T];
            smoothedTrajectories[j][T] = resampled_X_T; 
            // std::cout << "particleId=" << i <<  ", ses=" <<T << ", resampled_X_T=" << resampled_X_T << std::endl;

        }  
    }
    
    
    for(int t=T-1; t>=0; t--)
    {
        // std::cout << "t=" << t << std::endl;
        for(int j=0; j<M; j++)
        {
            std::vector<double> smoothedWeights_j(N,0);
            for(int i=0; i<N; i++)
            {
                //crp(x_t+1|x_t)
                std::vector<std::vector<double>> crpPriors =   particleFilterVec[i].getCrpPriors();
                std::vector<double> crp_tplus1 = crpPriors[t+1];
                // std::vector<std::vector<int>> particleTrajectories = particleFilterVec[i].getParticleTrajectories();
                // std::vector<int> particleHistory_t = particleTrajectories[t];
                // std::vector<double> crp_tplus1;
                // try
                // {
                //     crp_tplus1 = particleFilterVec[i].crpPrior2(particleHistory_t,t+1);
                // }
                // catch(const std::exception& e)
                // {
                //     std::cout << "Err in crp in E_step3" << std::endl;
                //     std::cerr << e.what() << '\n';
                // }
                
                
                // std::cout << "particleId=" << i <<  ", ses=" <<t << ", crp_tplus1:";
                //  for (auto const& i : crp_tplus1)
                //     std::cout << i << ", ";
                // std::cout << "\n" ; 


                int X_tplus1_smoothed = smoothedTrajectories[j][t+1];
                smoothedWeights_j[i]  = filteredWeights[t][i] * crp_tplus1[X_tplus1_smoothed];
                // std::cout << "particleId=" << i <<  ", ses=" <<t << ", X_tplus1_smoothed=" << X_tplus1_smoothed << std::endl;
            }
            normalize(smoothedWeights_j);
            std::vector<double> resampledParticles_j = systematicResampling(smoothedWeights_j);
            for(int i=0; i<N; i++)
            {
                int resampledParticle =  resampledParticles_j[i];
                // std::vector<std::vector<int>> particleTrajectories = particleFilterVec[resampledParticle].getParticleTrajectories();
                // std::vector<int> particleHistory_t = particleTrajectories[t+1];
                std::vector<int> assignments_resampled = particleFilterVec[resampledParticle].getOriginalSampledStrats();
                

                // std::vector<int> assignments_resampled = particleHistory_t;
                int resampled_X_t = assignments_resampled[t];
                smoothedTrajectories[j][t] = resampled_X_t; 
            }

        }
       
    }

    // std::cout << "smoothedTrajectories:" << std::endl;
    // for (const auto& row : smoothedTrajectories) {
    //     for (int value : row) {
    //         std::cout << value << " ";
    //     }
    //     std::cout << std::endl;
    // }


    return smoothedTrajectories;


}

double M_step3(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int M, int k, double gamma, std::vector<std::vector<int>> smoothedTrajectories, std::vector<double> params, BS::thread_pool& pool)
{
    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    
    std::vector<ParticleFilter> particleFilterVec;
    for (int i = 0; i < M; i++)
    {
        auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, i, 1.0);
        particleFilterVec.push_back(pf);
        // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
    }

    double I_1 = 0;
    double I_2 = 0;
    double I_3 = 0;

    std::vector<double> I2_vec;
    std::vector<double> I3_vec;
        
    BS::multi_future<double> loop_I3 = pool.submit_sequence<double>(0,M,[&particleFilterVec,&sessions,&smoothedTrajectories](int j)
    {
        double local_I_3 = 0.0;
        for(int t=0; t<sessions;t++)
        {
                   
            // std::vector<int> originalSampledStrats = particleFilterVec[i].getOriginalSampledStrats();
            // std::cout << "ses=" << t << ", i=" << i << ", originalSampledStrats=" << originalSampledStrats[t] << std::endl;

            double lik_i = particleFilterVec[j].getSesLikelihood(smoothedTrajectories[j][t], t);

            //std::cout << "ses=" << t << ", i=" << i << ", smoothedChoices[t][i]=" << smoothedChoices[t][i] << ", lik_i=" << lik_i << std::endl;       

            if (lik_i == 0) {
                lik_i = 1e-6;
            }

            local_I_3 +=  log(lik_i);

            if (std::isnan(local_I_3) || std::isinf(local_I_3)) {
                
                throw std::runtime_error("Error in local_I_3 value");
            }
            
            
        }
        return local_I_3;
    });

    loop_I3.wait();
    for (auto& future : loop_I3) {
        I3_vec.push_back(future.get());
    }

    
    BS::multi_future<double> loop_I2 = pool.submit_sequence<double>(0,M,[&particleFilterVec,&sessions, &smoothedTrajectories](int j)
    {
        double local_I_2 = 0.0;

        std::vector<int> particleHistory_t = smoothedTrajectories[j];
        for(int t=0; t<sessions-2; t++)
        {
            
            std::vector<double> crp_t;
            try
            {
                crp_t = particleFilterVec[j].crpPrior2(particleHistory_t,t);
            }
            catch(const std::exception& e)
            {
                std::cout << "Err in loop_I2 in M_step3" << std::endl;
                std::cerr << e.what() << '\n';
            }
            

            double p_crp_X_i_tplus1 = crp_t[particleHistory_t[t+1]];
            if(p_crp_X_i_tplus1==0)
            {
                p_crp_X_i_tplus1 = std::numeric_limits<double>::min();
            }


            local_I_2 = local_I_2 + log(p_crp_X_i_tplus1);

            if (std::isnan(local_I_2)) {
                // std::cout << "t=" << t << ", i=" << i << ", j=" << j << ", w_ij_smoothed=" << w_ij_smoothed << ", X_j_tplus1=" << X_j_tplus1 << ", crpPriors_i_t[X_j_tplus1]=" << p_crp_X_j_tplus1 << std::endl;
                throw std::runtime_error("Error nan I_2 value");
            }

            if (std::isinf(local_I_2)) {
                throw std::runtime_error("Error inf I_2 value");
            }
            
        }
        
        return local_I_2;


    });
    
    
    loop_I2.wait();
    for (auto& future : loop_I2) {
        I2_vec.push_back(future.get());
    }

    // std::cout << "I2=: ";
    // for (const auto &x : I2_vec)
    // {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;


    // std::cout << "I3=: ";
    // for (const auto &x : I3_vec)
    // {
    //     std::cout << x << " ";
    // }
    // std::cout << std::endl;


    double Q_k = 0;
    for(int i=0; i < M; i++)
    {
        Q_k = Q_k+I2_vec[i]+I3_vec[i];
    }

    Q_k = Q_k/M;
    

    // double gamma_k = (double) gamma/(double) k;
    // double Q_kplus1 = (1-gamma)*Q + gamma*Q_k;
    // std::cout << "Q_k=" << Q_k << "gamma=" << gamma << ", k=" << k <<  ", gamma_k=" << gamma_k << ", Q_kplus1=" << Q_kplus1 << std::endl;
    
    return (Q_k);


}

double M_step4(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::vector<std::vector<int>> smoothedTrajectories, std::vector<std::vector<double>> filteredWeights, std::vector<double> params, BS::thread_pool& pool)
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

    double I_1 = 0;
    double I_2 = 0;
    double I_3 = 0;

    std::vector<double> I2_vec;
    std::vector<double> I3_vec;
        
    BS::multi_future<double> loop_I3 = pool.submit_sequence<double>(0,N,[&particleFilterVec,&sessions,&smoothedTrajectories, &filteredWeights](int j)
    {
        double local_I_3 = 0.0;
        for(int t=0; t<sessions;t++)
        {
                   
            // std::vector<int> originalSampledStrats = particleFilterVec[i].getOriginalSampledStrats();
            // std::cout << "ses=" << t << ", i=" << i << ", originalSampledStrats=" << originalSampledStrats[t] << std::endl;

            double lik_i = particleFilterVec[j].getSesLikelihood(smoothedTrajectories[j][t], t);

            //std::cout << "ses=" << t << ", i=" << i << ", smoothedChoices[t][i]=" << smoothedChoices[t][i] << ", lik_i=" << lik_i << std::endl;       

            if (lik_i == 0) {
                lik_i = 1e-6;
            }

            local_I_3 +=  log(lik_i);

            if (std::isnan(local_I_3) || std::isinf(local_I_3)) {
                
                throw std::runtime_error("Error in local_I_3 value");
            }
            
            
        }
        local_I_3 = filteredWeights[sessions-1][j]*local_I_3;
        return local_I_3;
    });

    loop_I3.wait();
    for (auto& future : loop_I3) {
        I3_vec.push_back(future.get());
    }

    
    BS::multi_future<double> loop_I2 = pool.submit_sequence<double>(0,N,[&particleFilterVec,&sessions, &smoothedTrajectories, &filteredWeights](int j)
    {
        double local_I_2 = 0.0;

        std::vector<int> particleHistory_t = smoothedTrajectories[j];
        for(int t=0; t<sessions-2; t++)
        {
            
            std::vector<double> crp_t;
            try
            {
                crp_t = particleFilterVec[j].crpPrior2(particleHistory_t,t);
            }
            catch(const std::exception& e)
            {
                std::cout << "Err in loop_I2 in M_step3" << std::endl;
                std::cerr << e.what() << '\n';
            }
            

            double p_crp_X_i_tplus1 = crp_t[particleHistory_t[t+1]];
            if(p_crp_X_i_tplus1==0)
            {
                p_crp_X_i_tplus1 = std::numeric_limits<double>::min();
            }


            local_I_2 = local_I_2 + log(p_crp_X_i_tplus1);

            if (std::isnan(local_I_2)) {
                // std::cout << "t=" << t << ", i=" << i << ", j=" << j << ", w_ij_smoothed=" << w_ij_smoothed << ", X_j_tplus1=" << X_j_tplus1 << ", crpPriors_i_t[X_j_tplus1]=" << p_crp_X_j_tplus1 << std::endl;
                throw std::runtime_error("Error nan I_2 value");
            }

            if (std::isinf(local_I_2)) {
                throw std::runtime_error("Error inf I_2 value");
            }
            
        }
        local_I_2 = filteredWeights[sessions-1][j]*local_I_2;
        return local_I_2;


    });
    
    
    loop_I2.wait();
    for (auto& future : loop_I2) {
        I2_vec.push_back(future.get());
    }


    double Q_k = 0;
    for(int i=0; i < N; i++)
    {
        Q_k = Q_k+I2_vec[i]+I3_vec[i];
    }

    // Q_k = Q_k/M;
    

    // double gamma_k = (double) gamma/(double) k;
    // double Q_kplus1 = (1-gamma)*Q + gamma*Q_k;
    // std::cout << "Q_k=" << Q_k << "gamma=" << gamma << ", k=" << k <<  ", gamma_k=" << gamma_k << ", Q_kplus1=" << Q_kplus1 << std::endl;
    
    return (Q_k);


}


double M_step5(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, std::vector<int> smoothedTrajectory, std::vector<double> params, BS::thread_pool& pool)
{
    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    
    auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, 1, 1.0);
    

    double I_1 = 0;
    double I_2 = 0;
    double I_3 = 0;

    std::vector<double> I2_vec;
    std::vector<double> I3_vec;
        
    double local_I_3 = 0.0;
    for(int t=0; t<sessions;t++)
    {
                
        // std::vector<int> originalSampledStrats = particleFilterVec[i].getOriginalSampledStrats();
        // std::cout << "ses=" << t << ", i=" << i << ", originalSampledStrats=" << originalSampledStrats[t] << std::endl;

        double lik_i = pf.getSesLikelihood(smoothedTrajectory[t], t);

        //std::cout << "ses=" << t << ", i=" << i << ", smoothedChoices[t][i]=" << smoothedChoices[t][i] << ", lik_i=" << lik_i << std::endl;       

        if (lik_i == 0) {
            lik_i = 1e-6;
        }

        local_I_3 +=  log(lik_i);

        if (std::isnan(local_I_3) || std::isinf(local_I_3)) {
            
            throw std::runtime_error("Error in local_I_3 value");
        }
        
        
    }


   
    double local_I_2 = 0.0;

    std::vector<int> particleHistory_t = smoothedTrajectory;
    for(int t=0; t<sessions-2; t++)
    {
        
        std::vector<double> crp_t;
        try
        {
            crp_t = pf.crpPrior2(particleHistory_t,t);
        }
        catch(const std::exception& e)
        {
            std::cout << "Err in loop_I2 in M_step3" << std::endl;
            std::cerr << e.what() << '\n';
        }
        

        double p_crp_X_i_tplus1 = crp_t[particleHistory_t[t+1]];
        if(p_crp_X_i_tplus1==0)
        {
            p_crp_X_i_tplus1 = std::numeric_limits<double>::min();
        }


        local_I_2 = local_I_2 + log(p_crp_X_i_tplus1);

        if (std::isnan(local_I_2)) {
            // std::cout << "t=" << t << ", i=" << i << ", j=" << j << ", w_ij_smoothed=" << w_ij_smoothed << ", X_j_tplus1=" << X_j_tplus1 << ", crpPriors_i_t[X_j_tplus1]=" << p_crp_X_j_tplus1 << std::endl;
            throw std::runtime_error("Error nan I_2 value");
        }

        if (std::isinf(local_I_2)) {
            throw std::runtime_error("Error inf I_2 value");
        }
        
    }
    
    

    double Q_k = local_I_3+local_I_2;
    
    // Q_k = Q_k/M;
    

    // double gamma_k = (double) gamma/(double) k;
    // double Q_kplus1 = (1-gamma)*Q + gamma*Q_k;
    // std::cout << "Q_k=" << Q_k << "gamma=" << gamma << ", k=" << k <<  ", gamma_k=" << gamma_k << ", Q_kplus1=" << Q_kplus1 << std::endl;
    
    return (Q_k);


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
    std::vector<double> params = {0.228439, 0.921127, 0.0429102, 0.575078,0.2};
    std::vector<double> QFuncVals;
    std::vector<std::vector<double>> params_iter;
    double Q_prev = 0;
    int M = 10;
    double gamma = 0.1;
    std::vector<std::vector<int>> prevSmoothedTrajectories;

    for (int i = 0; i < 100; i++)
    {

        std::cout << "i=" << i << ", E-step" << std::endl;
        // auto [smoothedWeights, wijSmoothed, particleFilterVec, filteredWeights] = E_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, params, pool);
        // auto [smoothedWeightMat, particleFilterVec, smoothedChoices] = E_step2(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, params, pool);
        auto smoothedTrajectories = E_step3(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, M, params, pool);

        std::cout << "i=" << i << ", starting M-step" << std::endl;     
        PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, M, i+1, gamma, smoothedTrajectories, prevSmoothedTrajectories,  pool);
        // PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, resTuple, pool);
        // PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
        std::cout << "Initialized problem class" << std::endl;

        // Create a problem using Pagmo
        problem prob{pagmoprob};
        int count = 0;
        
        pagmo::nlopt method("sbplx");
        method.set_xtol_abs(1e-3);
        // method.set_maxeval(50);
        //pagmo::sade method (5,2,2,1e-6, 1e-6, false, 915909831);
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
        
        double maxStopCriteria = 0.0;
        for (size_t i = 0; i < params.size(); ++i) {
            double stopCriterion = std::abs(params[i] - dec_vec_champion[i])/(params[i]+0.01);
            if (stopCriterion > maxStopCriteria) {
                maxStopCriteria = stopCriterion;
            }
        }
        std::cout << "i=" << i << ", max_stopping_criteria=" << maxStopCriteria << std::endl;
        if(maxStopCriteria < 0.05)
        {
            std::cout << "Terminate EM, parameters converged after i=" << i << std::endl;
        }

        params = dec_vec_champion;
        //std::pair<std::vector<std::vector<double>>, double> q = particle_filter(N, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params,pool);
         
        prevSmoothedTrajectories = smoothedTrajectories;

        std::cout << "smoothedTrajectories:" << std::endl;
        for (const auto& row : smoothedTrajectories) {
            for (int value : row) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }
        //auto [filteredWeights, loglik] = fapf(N, fapfVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, pool);
        auto [filteredWeights, loglik, viterbiSeq] = particle_filter(N, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params,pool);
        std::cout << "viterbiSeq: ";
        for (const auto &x : viterbiSeq)
        {
            std::cout << x << " ";
        }
        std::cout << "\n";

        std::cout << "loglikelihood=" << loglik << std::endl;
    //    / QFuncVals.push_back(loglik);
        params_iter.push_back(params);

    }

    // std::cout << "Likelihoods=";
    // for (auto const &i : QFuncVals)
    //     std::cout << i << ", ";
    // std::cout << "\n";
     auto minElementIterator = std::max_element(QFuncVals.begin(), QFuncVals.end());
    // Calculate the index of the minimum element
    int maxIndex = std::distance(QFuncVals.begin(), minElementIterator);

    std::vector<double> finalParams = params_iter[maxIndex];

    // auto [smoothedWeights, wijSmoothed, particleFilterVec, filteredWeights] = E_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, finalParams, pool);
    // int sessions = smoothedWeights.size();
    // vector<vector<double>> smoothedPosterior(sessions, vector<double>(4));
    // for (int ses = 0; ses < sessions; ses++)
    // {
    //     for (int i = 0; i < N; i++)
    //     {
    //         std::vector<int> chosenStrategy_pf = particleFilterVec[i].getOriginalSampledStrats();

    //         // std::cout << "ses=" <<ses << ", particleId=" <<i << ", chosenStrat=" << chosenStrategy_pf[ses] << std::endl;
    //         smoothedPosterior[ses][chosenStrategy_pf[ses]] = smoothedPosterior[ses][chosenStrategy_pf[ses]] + smoothedWeights[ses][i];
    //         // postProbsOfExperts[ses][chosenStrategy_pf[ses]] = std::round(postProbsOfExperts[ses][chosenStrategy_pf[ses]] * 100.0) / 100.0;
    //     }

    // }
    // std::cout << "smoothed posterior:" << std::endl;
    // for (const auto& row : smoothedPosterior) {
    //     for (const auto& elem : row) {
    //         std::cout << std::fixed << std::setprecision(2) << elem << " ";
    //     }
    //     std::cout << std::endl; // Newline for each row
    // }


    return (finalParams);
}


std::vector<double> SAEM(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, BS::thread_pool& pool)
{
    std::vector<double> params = {0.01, 0.7, 0.0429102, 0.575078};
    std::vector<double> QFuncVals;
    std::vector<std::vector<double>> params_iter;
    double Q_prev = 0;
    int M = 10;
    double gamma = 0.1;
    std::vector<std::vector<int>> prevSmoothedTrajectories;
    std::vector<std::vector<double>> prevFilteredWeights;

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    std::vector<int> x_cond(sessions, 3);

    for (int i = 0; i < 200; i++)
    {

        std::cout << "i=" << i << ", E-step" << std::endl;
        std::vector<ParticleFilter> particleFilterVec;
        for (int i = 0; i < N; i++)
        {
            auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, i, 1.0);
            particleFilterVec.push_back(pf);
            // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
        }
        auto [filteredWeights, loglik, smoothedTrajectories] = cpf_as(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, x_cond, pool);

        PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, i+1, gamma, smoothedTrajectories, filteredWeights, prevSmoothedTrajectories, prevFilteredWeights,  pool);
        // PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, resTuple, pool);
        // PagmoProb pagmoprob(ratdata,Suboptimal_Hybrid3,Optimal_Hybrid3);
        std::cout << "Initialized problem class" << std::endl;

        // Create a problem using Pagmo
        problem prob{pagmoprob};
        int count = 0;
        
        pagmo::nlopt method("sbplx");
        method.set_xtol_abs(1e-3);
        // method.set_maxeval(50);
        // pagmo::sade method (10,2,2,1e-6, 1e-6, false, 915909831);
        pagmo::algorithm algo = pagmo::algorithm{method};
        pagmo::population pop(prob, 30);
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
        
        double maxStopCriteria = 0.0;
        for (size_t j = 0; j < params.size(); ++j) {
            double stopCriterion = std::abs(params[j] - dec_vec_champion[j])/(params[j]+0.01);
            if (stopCriterion > maxStopCriteria) {
                maxStopCriteria = stopCriterion;
            }

        }
        std::cout << "i=" << i << ", max_stopping_criteria=" << maxStopCriteria << std::endl;
        
        double relLogLik = 0;
        for(int k=0; k<N;k++)
        {
            double Q_k = M_step5(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, smoothedTrajectories[k], dec_vec_champion, pool);
            double Q_k_minus1 = M_step5(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3,smoothedTrajectories[k], params, pool);
            double ratio = Q_k/Q_k_minus1;
            relLogLik = relLogLik + ratio;

        }
        relLogLik = log(relLogLik/N);
        std::cout << "relLogLik=" << std::fixed << std::setprecision(4) << relLogLik << std::endl;

        params = dec_vec_champion;


        std::cout << "smoothedTrajectories:" << std::endl;
        for (const auto& row : smoothedTrajectories) {
            for (int value : row) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }

        std::vector<std::vector<double>> filteringDist(4,std::vector<double>(sessions));
        for (int ses = 0; ses < sessions; ses++)
        {
            for (int i = 0; i < N; i++)
            {
                std::vector<int> chosenStrategy_pf = particleFilterVec[i].getOriginalSampledStrats();

                // std::cout << "ses=" <<ses << ", particleId=" <<i << ", chosenStrat=" << chosenStrategy_pf[ses] << std::endl;
                filteringDist[chosenStrategy_pf[ses]][ses] = filteringDist[chosenStrategy_pf[ses]][ses] + filteredWeights[ses][i];
                // postProbsOfExperts[ses][chosenStrategy_pf[ses]] = std::round(postProbsOfExperts[ses][chosenStrategy_pf[ses]] * 100.0) / 100.0;
            }
        }

        std::cout << "filtering Dist=" << std::endl;
        for (const auto &row : filteringDist)
        {
            for (double num : row)
            {
                std::cout << std::fixed << std::setprecision(2) << num << " ";
            }
            std::cout << std::endl;
        }

        if(maxStopCriteria < 0.05 && i > 5)
        {
            break;
            std::cout << "Terminate EM, parameters converged after i=" << i << std::endl;
        }else if(std::abs(relLogLik) < 1e-4 && i > 5)
        {
            break;
            std::cout << "Terminate EM, likelihood converged after i=" << i << std::endl;
        }

        prevSmoothedTrajectories = smoothedTrajectories;
        prevFilteredWeights = filteredWeights;

        params_iter.push_back(params);
        
        int sampled_trajectory = sample(filteredWeights[sessions-1]);
        x_cond = smoothedTrajectories[sampled_trajectory]; 

    }

    // std::cout << "Likelihoods=";
    // for (auto const &i : QFuncVals)
    //     std::cout << i << ", ";
    // std::cout << "\n";
     auto minElementIterator = std::max_element(QFuncVals.begin(), QFuncVals.end());
    // Calculate the index of the minimum element
    int maxIndex = std::distance(QFuncVals.begin(), minElementIterator);

    std::vector<double> finalParams = params_iter[maxIndex];

    // auto [smoothedWeights, wijSmoothed, particleFilterVec, filteredWeights] = E_step(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, finalParams, pool);
    // int sessions = smoothedWeights.size();
    // vector<vector<double>> smoothedPosterior(sessions, vector<double>(4));
    // for (int ses = 0; ses < sessions; ses++)
    // {
    //     for (int i = 0; i < N; i++)
    //     {
    //         std::vector<int> chosenStrategy_pf = particleFilterVec[i].getOriginalSampledStrats();

    //         // std::cout << "ses=" <<ses << ", particleId=" <<i << ", chosenStrat=" << chosenStrategy_pf[ses] << std::endl;
    //         smoothedPosterior[ses][chosenStrategy_pf[ses]] = smoothedPosterior[ses][chosenStrategy_pf[ses]] + smoothedWeights[ses][i];
    //         // postProbsOfExperts[ses][chosenStrategy_pf[ses]] = std::round(postProbsOfExperts[ses][chosenStrategy_pf[ses]] * 100.0) / 100.0;
    //     }

    // }
    // std::cout << "smoothed posterior:" << std::endl;
    // for (const auto& row : smoothedPosterior) {
    //     for (const auto& elem : row) {
    //         std::cout << std::fixed << std::setprecision(2) << elem << " ";
    //     }
    //     std::cout << std::endl; // Newline for each row
    // }


    return (finalParams);
}


std::vector<double> Mle(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N,BS::thread_pool& pool)
{
    // std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>, std::vector<std::vector<double>>> res;
    // std::vector<std::vector<double>> smoothedWeights = std::get<0>(res);
    // std::vector<std::vector<std::vector<double>>> wijSmoothed = std::get<1>(res);
    // std::vector<ParticleFilter> particleFilterVec =  std::get<2>(res);
    // std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> resTuple = std::make_tuple(smoothedWeights, wijSmoothed, particleFilterVec);
    
    
    // std::tuple<std::vector<std::vector<double>>, std::vector<std::vector<std::vector<double>>>, std::vector<ParticleFilter>> resTuple;
    std::tuple<std::vector<std::vector<double>>, std::vector<ParticleFilter>, std::vector<std::vector<int>>> resTuple;
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

        pagmo::sade method (1,2,2);
        //method.set_maxeval(10);
        pagmo::algorithm algo = pagmo::algorithm {method};
        pagmo::population pop(prob, 20);
        for(int j=0; j<10; j++)
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
    // std::pair<std::vector<std::vector<double>>, double> q = particle_filter(N, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params,pool);
    // std::cout << "loglikelihood=" << q.second << std::endl;

    return (params);
}


void testQFunc(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, BS::thread_pool& pool, RInside & R)
{

    unsigned int numThreads = std::thread::hardware_concurrency();
    std::vector<double> params = { 0.05, 0.41, 0.03, 0.90};

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    std::vector<int> x_cond(sessions, 3);



    // Print the result
    std::cout << "Number of threads available: " << numThreads << "\n";
    for(int i=0; i<10; i++)
    {
        
        std::cout << "i=" <<i << ", performing E-step" << std::endl;

        std::vector<ParticleFilter> particleFilterVec;
        for (int i = 0; i < N; i++)
        {
            auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, i, 1.0);
            particleFilterVec.push_back(pf);
            // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
        }
        std::vector<int> xcond = {0,1,0,0,1,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,1,1,1,1};
        auto [filteredWeights, loglik, smoothedTrajectories] = cpf_as(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, x_cond, pool);
        std::cout << "loglik=" << loglik << std::endl;
        std::cout << "smoothedTrajectories:" << std::endl;
        for (const auto& row : smoothedTrajectories) {
            for (int value : row) {
                std::cout << value << " ";
            }
            std::cout << std::endl;
        }

        std::vector<std::vector<double>> filteringDist(4,std::vector<double>(sessions));
        for (int ses = 0; ses < sessions; ses++)
        {
            for (int i = 0; i < N; i++)
            {
                std::vector<int> chosenStrategy_pf = particleFilterVec[i].getOriginalSampledStrats();

                // std::cout << "ses=" <<ses << ", particleId=" <<i << ", chosenStrat=" << chosenStrategy_pf[ses] << std::endl;
                filteringDist[chosenStrategy_pf[ses]][ses] = filteringDist[chosenStrategy_pf[ses]][ses] + filteredWeights[ses][i];
                // postProbsOfExperts[ses][chosenStrategy_pf[ses]] = std::round(postProbsOfExperts[ses][chosenStrategy_pf[ses]] * 100.0) / 100.0;
            }
        }

        std::cout << "filtering Dist=" << std::endl;
        for (const auto &row : filteringDist)
        {
            for (double num : row)
            {
                std::cout << std::fixed << std::setprecision(2) << num << " ";
            }
            std::cout << std::endl;
        }



    }


}