#include "ParticleFilter.h"
#include "Pagmoprob.h"
#include "BS_thread_pool.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>


std::vector<double> stratifiedResampling(std::vector<double>&particleWeights)
{
    int numParticles = particleWeights.size();
    std::vector<double> resampledParticles(numParticles);

    std::vector<double> cumSum(numParticles);
    std::partial_sum(particleWeights.begin(), particleWeights.end(), cumSum.begin());

    int n=0; 
    int m=0;   
    std::random_device rd;
    std::mt19937 gen(rd());
       
    while(n < numParticles)
    {
        std::uniform_real_distribution<double> uniformDist(0.0, 1.0 / numParticles);
        double u = uniformDist(gen) + static_cast<double>(n) / numParticles;

        while(cumSum[m] < u)
        {
            ++m;
        }
        resampledParticles[n] = m;
        ++n;
    }

    return (resampledParticles);
}

std::pair<std::vector<std::vector<double>>, double> fapf(int N, std::vector<ParticleFilter> &particleFilterVec, const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, BS::thread_pool& pool)
{

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    // Initialize the particles with random assignments
    std::vector<double> w(N, 1.0 / N);

    std::vector<std::vector<double>> filteredWeights;
    // std::vector<std::vector<double>> crpPriors(N, std::vector<double>(4, 0.0));

    double loglik = 0;
    std::vector<double> stratChoices;
    // Iterate over the trials
    for (int ses = 0; ses < sessions; ses++)
    {
        
        //Initialize choices at t=0
        if(ses==0)
        {
            auto initialize = [&particleFilterVec,&ses](int i) {
                std::vector<int> particleHistory_t = particleFilterVec[i].getOriginalSampledStrats();
                std::vector<double> crp_i = particleFilterVec[i].crpPrior2(particleHistory_t,ses);

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

                // p(y_t|x_(t-1)) = \sum_(x_t) p(y_t|x_t)*p(x_t|x_(t-1))
                double sum = 0;
                for(int k=0; k<strategies.size();k++)
                {
                    sum += crp_i[k]*likelihoods[k];
                }    
                particleFilterVec[i].rollBackCredits();

                std::vector<double> p;
                for(int k=0; k<strategies.size();k++)
                {
                    p.push_back(crp_i[k]*likelihoods[k]/sum);
                }

                int sampled_strat = sample(p);

                particleFilterVec[i].addAssignment(ses, sampled_strat);
                particleFilterVec[i].addOriginalSampledStrat(ses, sampled_strat);
                double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses);
                // double loglik = particleFilterVec[i].getSesLogLikelihood(sampled_strat, ses);
                particleFilterVec[i].setLikelihood(lik);


            };

            for(int i=0; i<N; i++)
            {
                auto bound_lambda = std::bind(initialize, i);
                std::future<void> my_future = pool.submit_task(bound_lambda);
            }   

            pool.wait();         

        }

        //Step1: Compute smoothed weights to resample x_t: w_t ~ p(y_{t+1}|x_t)
        std::vector<double> likelihoods (N, 0);
        auto computeWeights = [&particleFilterVec,&ses,&likelihoods, &w](int i) {

            // particleFilterVec[i].updateStratCounts(ses);
            // std::vector<double> crp_i = particleFilterVec[i].crpPrior(ses);

            std::vector<int> particleHistory_t = particleFilterVec[i].getOriginalSampledStrats();
            if(particleHistory_t[ses]==-1)
            {
                throw std::runtime_error("Error particle history not updated");
            }
            std::vector<double> crp_t = particleFilterVec[i].crpPrior2(particleHistory_t,ses);           

            std::vector<std::shared_ptr<Strategy>> strategies = particleFilterVec[i].getStrategies();
            particleFilterVec[i].backUpStratCredits();
            std::vector<double> likelihoods_tplus1;
            for(int k=0; k<strategies.size();k++)
            {
                double lik = particleFilterVec[i].getSesLikelihood(k, ses+1);
                if(lik==0)
                {
                    lik = std::numeric_limits<double>::min();
                }
                likelihoods_tplus1.push_back(lik);
            }

            // p(y_t|x_(t-1)) = \sum_(x_t) p(y_t|x_t)*p(x_t|x_(t-1))
            double sum = 0;
            for(int k=0; k<strategies.size();k++)
            {
                int x_tplus1 = k;
                sum += crp_t[k]*likelihoods_tplus1[k];
            }    

            if(sum==0)
            {
                std::cout << "Err in generate_sampling_distribution. Check" << std::endl;
                std::cout << "ses=" << ses << " i=" << i <<  ", crp_i = ";
                for (auto const& w : crp_t)
                    std::cout << w << ", ";
                std::cout << "\n" ;
                std::cout << "ses=" << ses << " i=" << i <<  ", likelihoods = ";
                for (auto const& w : likelihoods)
                    std::cout << w << ", ";
                std::cout << "\n" ;

                // throw std::runtime_error("Err in generate_sampling_distribution. Check");
            }

            particleFilterVec[i].rollBackCredits();

            w[i] = sum;

        };

        for (int i = 0; i < N; i++) {
            auto bound_lambda = std::bind(computeWeights, i);
            std::future<void> my_future = pool.submit_task(bound_lambda);
        }

        pool.wait();

        double lik_ses = std::accumulate(w.begin(), w.end(), 0.0);
        lik_ses = lik_ses/N;
        loglik = loglik+log(lik_ses);

        normalize(w);

        //Step2: Resample the particles based on weights computed from step1     
        std::vector<double> resampledIndices = stratifiedResampling(w);
        for(int j=0; j<N; j++)
        {
            particleFilterVec[j].backUpChosenStrategies();
            particleFilterVec[j].backUpStratCredits();
        }
        
        auto resample_x_t = [&particleFilterVec,&ses](int i, std::vector<double> resampledIndices_) {
            int newIndex = resampledIndices_[i];
            //ParticleFilter pf(particleFilterVec[newIndex]);

            // Use lock_guard to ensure mutual exclusion for particleFilterVec updates
            std::vector<std::pair<std::vector<double>,std::vector<double>>>& stratBackUps = particleFilterVec[newIndex].getStratCreditBackUps();
            std::vector<signed int> chosenStrategyBackUp =  particleFilterVec[newIndex].getChosenStrategyBackups();
            int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
            particleFilterVec[i].setChosenStrategies(chosenStrategyBackUp);
            particleFilterVec[i].setStratBackups(stratBackUps);
            //particleFilterVec[i].updateStratCounts(ses);
            int updatedChosenStrat = particleFilterVec[i].getChosenStratgies()[ses];

            // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[newIndex].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;
        };


        for (int i = 0; i < N; i++) {
            auto bound_lambda = std::bind(resample_x_t, i,resampledIndices);
            std::future<void> my_future = pool.submit_task(bound_lambda);

        }

        pool.wait();

        //Step3: Sample new state at t=t+1 by generating new state from p(x_{t+1}|x_{t},y_t)  
        std::vector<double> likelihoods_ses(N,0);
        auto sample_x_tplus1 = [&particleFilterVec,&ses,&w,&N,&likelihoods_ses](int i) {
            // Use lock_guard to ensure mutual exclusion for particleFilterVec and w updates
            // std::lock_guard<std::mutex> lock(mutex);

            // particleFilterVec[i].updateStratCounts(ses);
            //std::vector<double> crp_i = particleFilterVec[i].crpPrior(ses);
            std::vector<int> particleHistory_t = particleFilterVec[i].getChosenStratgies();
            std::vector<double> crp_t = particleFilterVec[i].crpPrior2(particleHistory_t,ses);  

            std::vector<std::shared_ptr<Strategy>> strategies = particleFilterVec[i].getStrategies();
            particleFilterVec[i].backUpStratCredits();
            std::vector<double> likelihoods_tplus1;
            for(int k=0; k<strategies.size();k++)
            {
                double lik = particleFilterVec[i].getSesLikelihood(k, ses+1);
                if(lik==0)
                {
                    lik = std::numeric_limits<double>::min();
                }
                likelihoods_tplus1.push_back(lik);
            }

            // p(y_t|x_(t-1)) = \sum_(x_t) p(y_t|x_t)*p(x_t|x_(t-1))
            double sum = 0;
            for(int k=0; k<strategies.size();k++)
            {
                int x_tplus1 = k;
                sum += crp_t[k]*likelihoods_tplus1[k];
            }    

            if(sum==0)
            {
                std::cout << "Err in generate_sampling_distribution. Check" << std::endl;
                std::cout << "ses=" << ses << " i=" << i <<  ", crp_i = ";
                for (auto const& w : crp_t)
                    std::cout << w << ", ";
                std::cout << "\n" ;
                std::cout << "ses=" << ses << " i=" << i <<  ", likelihoods = ";
                for (auto const& w : likelihoods_tplus1)
                    std::cout << w << ", ";
                std::cout << "\n" ;

                // throw std::runtime_error("Err in generate_sampling_distribution. Check");
            }
         
            particleFilterVec[i].rollBackCredits();
            std::vector<double> p;
            for(int k=0; k<strategies.size();k++)
            {
                p.push_back(crp_t[k]*likelihoods_tplus1[k]/sum);
            }


            particleFilterVec[i].addCrpPrior(p,ses);
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
            particleFilterVec[i].addAssignment(ses+1, sampled_strat);
            particleFilterVec[i].addOriginalSampledStrat(ses+1, sampled_strat);
            // std::cout << "i=" << i << ", sampled_strat=" << sampled_strat << std::endl;
            particleHistory_t = particleFilterVec[i].getOriginalSampledStrats();
            if(particleHistory_t[ses+1] == -1)
            {
                throw std::runtime_error("Error particle history not updated");
            }
            double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses+1);
            // double loglik = particleFilterVec[i].getSesLogLikelihood(sampled_strat, ses);
            particleFilterVec[i].setLikelihood(lik);
            // log_w[i] += loglik;

            // particleFilterVec[i].backUpStratCredits(); 

        };

        // ThreadPool pool(16);
        // std::vector<std::future<void>> futures;

        for (int i = 0; i < N; i++) {
            auto bound_lambda = std::bind(sample_x_tplus1, i);
            std::future<void> my_future = pool.submit_task(bound_lambda);
        }

        pool.wait();

        std::vector<double> w(N, 1.0 / N);
        filteredWeights.push_back(w);

        
    }

    // Return the posterior probabilities
    return std::make_pair(filteredWeights, loglik);
}