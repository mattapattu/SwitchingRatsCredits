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
#include <pagmo/algorithms/de.hpp>
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

void stateEstimation(const RatData &ratdata, const MazeGraph &Suboptimal_Hybrid3, const MazeGraph &Optimal_Hybrid3, int N, std::vector<double> params, int l_truncate, BS::thread_pool& pool)
{
    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    std::vector<std::vector<int>> sampledSmoothedTrajectories;
    std::vector<std::vector<double>> stratCounts(4, std::vector<double>(sessions, 0.0));
    std::vector<int> x_cond(sessions,0);

    for (int i = 0; i < 1100; i++)
    {

        // std::cout << "i=" << i << ", E-step" << std::endl;
        std::vector<ParticleFilter> particleFilterVec;
        for (int k = 0; k < N; k++)
        {
            auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, k, 1.0);
            particleFilterVec.push_back(pf);
            // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
        }
        // std::cout << "i=" << i << ", initialized pf vec" << std::endl;

        auto [filteredWeights, loglik, smoothedTrajectories] = cpf_as(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, x_cond, l_truncate, pool);

        // std::cout << "Pf sim completed" << std::endl;
 

        int sampled_trajectory = sample(filteredWeights[sessions-1]);
        x_cond = smoothedTrajectories[sampled_trajectory]; 
        if(i >= 300 && i%10==0)
        {
            sampledSmoothedTrajectories.push_back(x_cond);
        }
        
    }
    std::cout << "Generated smoothed trajectories" << std::endl;
    for(int j =0; j < sampledSmoothedTrajectories.size(); j++)
    {
       for(int t=0; t<sessions;t++)
       {
            int strat_t_j = sampledSmoothedTrajectories[j][t];
            stratCounts[strat_t_j][t]++;
       } 
    }

    for (auto& row : stratCounts) {
        for (auto& element : row) {
            element /= (double) sampledSmoothedTrajectories.size();
        }
    }

    for (const auto& row : stratCounts) {
        for (double element : row) {
            std::cout << element << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "Map sequence: ";
    for(int t=0; t<sessions;t++)
    {
        std::vector<double> stratProbs_t = {stratCounts[0][t],stratCounts[1][t],stratCounts[2][t],stratCounts[3][t]};
        auto max_it = std::max_element(stratProbs_t.begin(), stratProbs_t.end());
        size_t max_index = std::distance(stratProbs_t.begin(), max_it);

        std::vector<double> sortedVec = stratProbs_t;

        std::sort(sortedVec.begin(), sortedVec.end(), std::greater<double>());
        if(sortedVec[0] - sortedVec[1] >= 0.1)
        {
            std::cout << max_index << ", "; 
        }else{
            std::cout << " None,"; 
        }

    } 
    std::cout << std::endl;


    

    return;

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
    std::vector<int> x_cond(sessions, 0);

    int l_truncate = 5;   

    for (int i = 0; i < 300; i++)
    {

        std::cout << "i=" << i << ", E-step" << std::endl;
        std::vector<ParticleFilter> particleFilterVec;
        for (int i = 0; i < N; i++)
        {
            auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, i, 1.0);
            particleFilterVec.push_back(pf);
            // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
        }
        auto [filteredWeights, loglik, smoothedTrajectories] = cpf_as(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, x_cond, l_truncate, pool);

        if(i >= 100)
        {
            PagmoProb pagmoprob(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, i+1, gamma, smoothedTrajectories, filteredWeights, prevSmoothedTrajectories, prevFilteredWeights,  pool);
            std::cout << "Initialized problem class" << std::endl;

            // Create a problem using Pagmo
            problem prob{pagmoprob};
            int count = 0;
            
            pagmo::nlopt method("sbplx");
            method.set_xtol_abs(1e-3);
            // method.set_maxeval(50);
            //pagmo::sade method (10,2,2,1e-6, 1e-6, false, 915909831);
            // pagmo::de method(10);
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
            std::cout << "relLogLik=" << std::fixed << std::setprecision(6) << relLogLik << std::endl;

            params = dec_vec_champion;


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

            // if(maxStopCriteria < 0.05 && i > 5)
            // {
            //     std::cout << "Terminate EM, parameters converged after i=" << i << std::endl;
            //     break;
            // }else
            if(std::abs(relLogLik) < 1e-5 && i > 120)
            {
                std::cout << "Terminate EM, likelihood converged after i=" << i  << std::endl;
                std::vector<ParticleFilter> particleFilterVec_;
                for (int i = 0; i < N; i++)
                {
                    auto pf = ParticleFilter(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, params, i, 1.0);
                    particleFilterVec_.push_back(pf);
                    // std::cout << "i=" << i << ", particleId=" << particleFilterVec[i].getParticleId() << std::endl;
                }
                auto [filteredWeights_, loglik_, smoothedTrajectories_] = cpf_as(N, particleFilterVec_, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, x_cond, l_truncate ,pool);
                std::cout << "loglik=" << loglik_ << std::endl;
                std::cout << "Joint Posterior:" << std::endl;
                stateEstimation(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, params, l_truncate, pool);

                break;

            }

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
    //std::vector<double> params = {0.12, 0.92, 0.06, 0.51};
    std::vector<double> params;
    if(ratdata.getRat()=="rat_103")
    {
        params = {0.05, 1.00, 0.04, 0.79};

    }else if(ratdata.getRat()=="rat_106")
    {
        params = {0.25, 0.81, 0.05, 0.60};

    }else if(ratdata.getRat()=="rat_112")
    {
        params = {0.11, 0.72, 0.02, 0.63};

    }else if(ratdata.getRat()=="rat_113")
    {
        params = {0.80, 0.65, 0.05, 0.72};

    }else if(ratdata.getRat()=="rat_114")
    {
        params = {0.11, 0.91, 0.05, 0.52};

    }

    arma::mat allpaths = ratdata.getPaths();
    arma::vec sessionVec = allpaths.col(4);
    arma::vec uniqSessIdx = arma::unique(sessionVec);
    int sessions = uniqSessIdx.n_elem;
    std::vector<int> x_cond(sessions, 0);



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
        std::vector<int> xcond(sessions,0);
        auto [filteredWeights, loglik, smoothedTrajectories] = cpf_as(N, particleFilterVec, ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, x_cond, 2, pool);
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

        stateEstimation(ratdata, Suboptimal_Hybrid3, Optimal_Hybrid3, N, params, 1, pool);

    }


}