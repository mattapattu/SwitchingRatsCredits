#include "ParticleFilter.h"
#include "Pagmoprob.h"
#include "BS_thread_pool.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>

double log_sum_exp(const std::vector<double>& log_weights) {
    double max_log_weight = *std::max_element(log_weights.begin(), log_weights.end());
    double sum_exp_diff = 0.0;
    for (const double& lw : log_weights) {
        sum_exp_diff += std::exp(lw - max_log_weight);
    }
    return max_log_weight + std::log(sum_exp_diff);
}


std::tuple<std::vector<std::vector<double>>, double, std::vector<std::vector<int>>> cpf_as(int N, std::vector<ParticleFilter> &particleFilterVec, const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, std::vector<int> x_cond, int l_truncate ,BS::thread_pool& pool)
{

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;

    // Initialize the particles with random assignments
    // vector<vector<int>> m(N, vector<int>(T));

    std::vector<double> w(N, 1.0 /(double) N);
    // vector<double> log_w(N, log(1.0 / N));

    // w[t][i]
    std::vector<std::vector<double>> filteredWeights;
    std::vector<std::vector<int>> smoothedTrajectories;

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
            BS::multi_future<void> initPf = pool.submit_sequence(0,N,[&particleFilterVec,&delta,&phi,&ses,&N, &x_cond,&w](int i) {
              
                std::vector<double> crp_i = {0.25,0.25,0.25,0.25};
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
                // if(sum==0)
                // {
                //     std::cout << "Err in generate_sampling_distribution. Check" << std::endl;
                //     std::cout << "ses=" << ses << " i=" << i <<  ", crp_i = ";
                //     for (auto const& w : crp_i)
                //         std::cout << w << ", ";
                //     std::cout << "\n" ;
                //     std::cout << "ses=" << ses << " i=" << i <<  ", likelihoods = ";
                //     for (auto const& w : likelihoods)
                //         std::cout << w << ", ";
                //     std::cout << "\n" ;

                //     throw std::runtime_error("Err in generate_sampling_distribution. Check");
                // }
                
                particleFilterVec[i].rollBackCredits();

                particleFilterVec[i].addCrpPrior(p,ses);

                int sampled_strat = -1;
                if(i == N-1)
                {
                    sampled_strat = x_cond[0];
                }else{
                    sampled_strat = sample(p);
                }

                // if(p[sampled_strat] == 0)
                // {
                //     std::cout << "Sample not matching distribution. Check" << std::endl;
                //     std::cout << "ses=" << ses << ", sample=" << sampled_strat;
                //     for (auto const& n : p)
                //         std::cout << n << ", ";
                //     std::cout << "\n" ;

                //     throw std::runtime_error("Sample not matching distribution. Check");
                // }

                if(p[sampled_strat] == 0)
                {
                    p[sampled_strat] = std::numeric_limits<double>::min();
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
                w[i] = lik*crp_i[sampled_strat]/p[sampled_strat];
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
           
            BS::multi_future<void> resample_particle_filter = pool.submit_sequence(0,N-1,[&particleFilterVec,&ses,&resampledIndices](int i) {
                int newIndex = resampledIndices[i];
                //ParticleFilter pf(particleFilterVec[newIndex]);

                // Use lock_guard to ensure mutual exclusion for particleFilterVec updates
                std::vector<std::pair<std::vector<double>,std::vector<double>>>& stratBackUps = particleFilterVec[newIndex].getStratCreditBackUps();
                //New ancestor = particleFilterVec[newIndex].chosenStrategies
                std::vector<signed int> chosenStrategyBackUp =  particleFilterVec[newIndex].getChosenStrategyBackups();
                int chosenStrat = particleFilterVec[i].getChosenStratgies()[ses];
                particleFilterVec[i].setChosenStrategies(chosenStrategyBackUp);
                particleFilterVec[i].setStratBackups(stratBackUps);

                // std::cout << "ses=" << ses << ", particleId=" << particleFilterVec[i].getParticleId() << ", resampledParticle=" << particleFilterVec[newIndex].getParticleId() << ", chosenStrat=" << chosenStrat << ", updatedChosenStrat=" << updatedChosenStrat << std::endl;
            });

            resample_particle_filter.wait();


            // std::cout << "ses=" <<ses <<", updating particles completed." << std::endl;
            double initWeight = (1.0 / (double)N);
            std::fill(w.begin(), w.end(), initWeight);
            
        }

        //Ancestor sampling
        std::vector<double> ancestorProbs(N,0.0);
        BS::multi_future<void> ancestorSampling = pool.submit_sequence(0,N,[&particleFilterVec,&ses,&w,&x_cond,&sessions, &ancestorProbs, &l_truncate](int i) 
        {
            particleFilterVec[i].backUpStratCredits();
            std::vector<int> particleHistory_t_minus1 = particleFilterVec[i].getParticleTrajectories()[ses-1];
            
            int l = std::min(sessions, (ses-1+l_truncate));
            double prod = log(1);
            for(int s=ses; s < l; s++)
            {
                
                std::vector<double> crp_i = particleFilterVec[i].crpPrior2(particleHistory_t_minus1,s-1);
                double lik = particleFilterVec[i].getSesLikelihood(x_cond[s], s);
                prod += log(crp_i[x_cond[s]]) + log(lik);
                particleHistory_t_minus1[s] = x_cond[s];

            }
            // //Choice history of particle i before resampling
            // std::vector<double> crp_i;
            // try
            // {
            //     crp_i = particleFilterVec[i].crpPrior2(particleHistory_t_minus1,ses-1);

            // }
            // catch(const std::exception& e)
            // {
            //     std::cout << "Err in propogate in ancestorProbs PF" << std::endl;
            //     //std::cerr << e.what() << '\n';
            // }
            ancestorProbs[i] = prod + log(w[i]);

            if(ancestorProbs[i] == 0)
            {
                std::cout << "ancestorProbs[i]=0" << ", ses=" << ses << ", i=" << i << std::endl;
            }

            particleFilterVec[i].rollBackCredits();

        });
        ancestorSampling.wait();        

        //double sum = std::accumulate(ancestorProbs.begin(), ancestorProbs.end(), 0); 
        double lse = log_sum_exp(ancestorProbs);

        std::vector<double> normalized_weights;
        for (const double& lw : ancestorProbs) {
            normalized_weights.push_back(std::exp(lw - lse));
        }

        // if(std::accumulate(normalized_weights.begin(), normalized_weights.end(), 0) != 1)
        // {
        //     double weights_sum = std::accumulate(normalized_weights.begin(), normalized_weights.end(), 0);
        //     std::cout << "Prob weights not summing to one, weights_sum=" << weights_sum  << std::endl;
        // }

        int sampled_ancestor = sample(normalized_weights);
        std::vector<signed int> chosenAncestorBackUp =  particleFilterVec[sampled_ancestor].getChosenStrategyBackups();
        particleFilterVec[N-1].setChosenStrategies(chosenAncestorBackUp);


        //Propogation
        std::vector<double> lik_ses(N,0);
         BS::multi_future<void> propogate = pool.submit_sequence(0,N,[&particleFilterVec,&lik_ses,&ses,&w, &N, &x_cond](int i) {

            //Choice history with new ancestor after resampling
            std::vector<int> resampledParticleHistory_t_minus1 = particleFilterVec[i].getChosenStratgies();
            std::vector<double> crp_i;
            try
            {
                crp_i = particleFilterVec[i].crpPrior2(resampledParticleHistory_t_minus1,ses-1);

            }
            catch(const std::exception& e)
            {
                std::cout << "Err in propogate in PF" << std::endl;
                std::cerr << e.what() << '\n';
            }

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
            if(sum == 0)
            {
                sum = std::numeric_limits<double>::min();
            }    
            if(sum == 0)
            {
                sum = std::numeric_limits<double>::min();
            }
            std::vector<double> p;
            for(int k=0; k<strategies.size();k++)
            {
                p.push_back(crp_i[k]*likelihoods[k]/sum);
            }
            // if(sum==0)
            // {
            //     std::cout << "Err in generate_sampling_distribution. Check" << std::endl;
            //     std::cout << "ses=" << ses << " i=" << i <<  ", crp_i = ";
            //     for (auto const& w : crp_i)
            //         std::cout << w << ", ";
            //     std::cout << "\n" ;

            //     std::cout << "particleId=" << i <<  ", ses=" <<ses << ", resampledParticleHistory_t_minus1:";
            //     for (auto const& i : resampledParticleHistory_t_minus1)
            //         std::cout << i << ", ";
            //     std::cout << "\n" ; 


                

            //     throw std::runtime_error("Err in generate_sampling_distribution. Check");
            // }

            
            
            particleFilterVec[i].rollBackCredits();

            particleFilterVec[i].addCrpPrior(p,ses);

            int sampled_strat = -1;
            if(i == N-1)
            {
                sampled_strat = x_cond[ses];
            }else{
                sampled_strat = sample(p);
            }
            // if(p[sampled_strat] == 0)
            // {
            //     std::cout << "Sample not matching distribution. Check" << std::endl;
            //     std::cout << "ses=" << ses << ", sample=" << sampled_strat;
            //     for (auto const& n : p)
            //         std::cout << n << ", ";
            //     std::cout << "\n" ;

            //     if(i == N-1)
            //     {
            //         std::cout << "ses=" << ses << ", xcond=";
            //         for (auto const& n : x_cond)
            //             std::cout << n << ", ";
            //         std::cout << "\n" ;
            //     }

            //     throw std::runtime_error("Sample not matching distribution. Check");
            // }

            if(p[sampled_strat] == 0)
            {
                p[sampled_strat] = std::numeric_limits<double>::min();
            }
            
            particleFilterVec[i].addAssignment(ses, sampled_strat);
            particleFilterVec[i].addOriginalSampledStrat(ses, sampled_strat);
            double lik = particleFilterVec[i].getSesLikelihood(sampled_strat, ses);
            lik_ses[i] = lik;
            std::vector<int> particleHistory_t = particleFilterVec[i].getChosenStratgies();
            particleFilterVec[i].addParticleTrajectory(particleHistory_t, ses);


            // // double loglik = particleFilterVec[i].getSesLogLikelihood(sampled_strat, ses);
            // particleFilterVec[i].setLikelihood(lik);
            w[i] = lik*crp_i[sampled_strat]/p[sampled_strat];
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
       

    }

    for(int i=0; i< N; i++)
    {
        std::vector<int> particleHistory_T = particleFilterVec[i].getParticleTrajectories()[sessions-1];
        smoothedTrajectories.push_back(particleHistory_T);
    }

    // Compute the posterior probabilities for each expert by counting the occurrences of each expert in the last trial
    // vector<vector<double>> postProbsOfExperts(sessions, vector<double>(4) );

    // Return the posterior probabilities
    return std::make_tuple(filteredWeights, loglik, smoothedTrajectories);
    
}
