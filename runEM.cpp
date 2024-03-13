#include "InferStrategy.h"
#include "Simulation.h"
#include "ParticleFilter.h"
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/utility.hpp>




int main(int argc, char* argv[]) 
{
    std::cout <<"Inside main" <<std::endl;
    // Replace with the path to your Rdata file and the S4 object name
    // std::string rdataFilePath = "/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/rat114.Rdata";
    // std::string s4ObjectName = "ratdata";
    RInside R;

    //std::vector<std::string> rats = {"rat103","rat106","rat112","rat113","rat114"};

    std::vector<std::string> rats = {"rat106"};

    //std::string rat = argv[1];
    //std::vector<std::string> rats = {rat};

    //std::string run = argv[1];

    BS::thread_pool pool(18);
    std::cout << "Nb of threads in pool=" << pool.get_thread_count() << std::endl;

    for(const std::string& ratName: rats)
    {
        std::string cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/InverseRL/"+ ratName +".Rdata')";
        R.parseEvalQ(cmd);                  
        Rcpp::S4 ratdata = R.parseEval("get('ratdata')");

        cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/InverseRL/Hybrid3.Rdata')";
        R.parseEvalQ(cmd);                  
        Rcpp::S4 Optimal_Hybrid3 = R.parseEval("get('Hybrid3')"); 


        cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/InverseRL/SubOptimalHybrid3.Rdata')";
        R.parseEvalQ(cmd);                  
        Rcpp::S4 Suboptimal_Hybrid3 = R.parseEval("get('SubOptimalHybrid3')"); 

        RatData rdata(ratdata);
        MazeGraph suboptimalHybrid3(Suboptimal_Hybrid3, false);
        MazeGraph optimalHybrid3(Optimal_Hybrid3, true);

        std::string rat = rdata.getRat();
        std::cout << "rat=" << rdata.getRat() << std::endl;
    
        // Write params to file
        // findParams(rdata, suboptimalHybrid3, optimalHybrid3);    

        // ////Read the params from from rat param file, e.g rat_103.txt
        // std::string filename = rat + ".txt";
        // std::ifstream infile(filename);
        // std::map<std::pair<std::string, bool>, std::vector<double>> params;
        // boost::archive::text_iarchive ia(infile);
        // ia >> params;
        // infile.close();


        //Estimate cluster parameters and write to clusterParams.txt
        // findClusterParams(rdata, suboptimalHybrid3, optimalHybrid3);

        //findMultiObjClusterParams(rdata, suboptimalHybrid3, optimalHybrid3, params);

        //read clusterParams.txt to get the parameters for rat
        std::string filename_cluster = "clusterMLEParams.txt";
        std::ifstream cluster_infile(filename_cluster);
        std::map<std::string, std::vector<double>> clusterParams;
        boost::archive::text_iarchive ia_cluster(cluster_infile);
        ia_cluster >> clusterParams;
        cluster_infile.close();

        //std::vector<double> v = {0.1, 0.812323, 0.0703131, 0.813582}; 
        //std::pair<std::vector<std::vector<double>>, double> q = particle_filter(1000, rdata, suboptimalHybrid3, optimalHybrid3, v );

        
        //std::vector<double> params = EM(rdata, suboptimalHybrid3, optimalHybrid3, 1000, pool);
         //std::vector<double> params = Mle(rdata, suboptimalHybrid3, optimalHybrid3, 100, pool);
        testQFunc(rdata, suboptimalHybrid3, optimalHybrid3, 100,pool,R);


        //runEM(rdata, suboptimalHybrid3, optimalHybrid3, clusterParams, true);

        //runEM2(rdata, suboptimalHybrid3, optimalHybrid3, clusterParams, true);

        //testRecovery(rdata, suboptimalHybrid3, optimalHybrid3, R);
        
        //testLogLik(rdata, suboptimalHybrid3, optimalHybrid3);
    }

    for(const std::string& ratName: rats)
    {
        std::string cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/InverseRL/"+ ratName +".Rdata')";
        R.parseEvalQ(cmd);                  
        Rcpp::S4 ratdata = R.parseEval("get('ratdata')");

        cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/InverseRL/Hybrid3.Rdata')";
        R.parseEvalQ(cmd);                  
        Rcpp::S4 Optimal_Hybrid3 = R.parseEval("get('Hybrid3')"); 


        cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/InverseRL/SubOptimalHybrid3.Rdata')";
        R.parseEvalQ(cmd);                  
        Rcpp::S4 Suboptimal_Hybrid3 = R.parseEval("get('SubOptimalHybrid3')"); 

        RatData rdata(ratdata);
        MazeGraph suboptimalHybrid3(Suboptimal_Hybrid3, false);
        MazeGraph optimalHybrid3(Optimal_Hybrid3, true);

        std::cout << "rat=" << rdata.getRat() << ", starting testRecovery" << std::endl;

        //testRecovery(rdata, suboptimalHybrid3, optimalHybrid3, R, "run1");

    }

        

}
