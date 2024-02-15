#ifndef MAZEGRAPH_H
#define MAZEGRAPH_H

#include <RcppArmadillo.h>


struct MazeEdge {
    std::string src;
    std::string dest;
    double prob;
};


class MazeGraph{
    public:
        MazeGraph(const Rcpp::S4& turnModel, bool optimal_){
           
            name = Rcpp::as<std::string>(turnModel.slot("Name"));
            optimal = optimal_;

            Rcpp::S4 s4graphS0 = Rcpp::as<Rcpp::S4>(turnModel.slot("S0"));
            Rcpp::CharacterVector rcppNodeListS0 = turnModel.slot("nodes.S0");
            Rcpp::List rcppEdgeListS0 = turnModel.slot("edges.S0");
            Rcpp::List turnNodesS0 = turnModel.slot("turnNodes.S0");

            Rcpp::S4 s4graphS1 = Rcpp::as<Rcpp::S4>(turnModel.slot("S1"));
            Rcpp::CharacterVector rcppNodeListS1 = turnModel.slot("nodes.S1");
            Rcpp::List rcppEdgeListS1 = turnModel.slot("edges.S1");
            Rcpp::List turnNodesS1 = turnModel.slot("turnNodes.S1");

            Rcpp::List rcppNodeGroups =  turnModel.slot("nodeGroups");

            // for(int i=0;i<nodeGroups.size();i++)
            // {
            //     std::vector<int> intVector;
            //     for (auto str : nodeGroups[i]) {
            //         intVector.push_back(std::stoi(Rcpp::as<std::string>(str)));
            //     }

            //     nodeGroups.push_back(intVector);
            // }

            
            nodeListS0 = Rcpp::as<std::vector<std::string>>(rcppNodeListS0);
            nodeListS1 = Rcpp::as<std::vector<std::string>>(rcppNodeListS1);
           
            
            for (int i = 0; i < rcppEdgeListS0.size(); i++)
            {

                Rcpp::S4 edge = rcppEdgeListS0[i];
                SEXP edgeVec = edge.slot("edge");
                Rcpp::StringVector vec(edgeVec);
                SEXP prob = edge.slot("prob");
                Rcpp::NumericVector probVec(prob);
                edgeListS0.push_back({Rcpp::as<std::string>(vec[0]), Rcpp::as<std::string>(vec[1]), probVec[0]});

            }

            for (int i = 0; i < rcppEdgeListS1.size(); i++)
            {

                Rcpp::S4 edge = rcppEdgeListS1[i];
                SEXP edgeVec = edge.slot("edge");
                Rcpp::StringVector vec(edgeVec);
                SEXP prob = edge.slot("prob");
                Rcpp::NumericVector probVec(prob);
                edgeListS1.push_back({Rcpp::as<std::string>(vec[0]), Rcpp::as<std::string>(vec[1]), probVec[0]});

            }

            Rcpp::CharacterVector turnNodesNamesS0 = turnNodesS0.names();
            for (int i = 0; i < turnNodesNamesS0.size(); ++i) {
                // Extract the element (assuming each element is a std::vector<std::string>)
                Rcpp::CharacterVector vec = turnNodesS0[i];

                // Convert Rcpp::CharacterVector to std::vector<std::string>
                std::vector<std::string> cppVector = Rcpp::as<std::vector<std::string>>(vec);

                // Use the name as the nodeName
                std::string nodeName = Rcpp::as<std::string>(turnNodesNamesS0[i]);

                // Add to the result map
                turnNodeMapS0[nodeName] = cppVector;
            }

            Rcpp::CharacterVector turnNodesNamesS1 = turnNodesS1.names();
            for (int i = 0; i < turnNodesNamesS1.size(); ++i) {
                // Extract the element (assuming each element is a std::vector<std::string>)
                Rcpp::CharacterVector vec = turnNodesS1[i];

                // Convert Rcpp::CharacterVector to std::vector<std::string>
                std::vector<std::string> cppVector = Rcpp::as<std::vector<std::string>>(vec);

                // Use the name as the nodeName
                std::string nodeName = Rcpp::as<std::string>(turnNodesNamesS1[i]);

                // Add to the result map
                turnNodeMapS1[nodeName] = cppVector;
            }

            Path0S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path0"));
            Path1S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path1"));
            Path2S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path2"));
            Path3S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path3"));
            Path4S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path4"));
            Path5S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path5"));
            if(!optimal)
            {
                Path6S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path6"));
                Path7S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path7"));
                Path8S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path8"));
                Path9S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path9"));
                Path10S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path10"));
                Path11S0 = Rcpp::as<std::vector<std::string>>(s4graphS0.slot("Path11"));
            }
            

            Path0S1 = Rcpp::as<std::vector<std::string>>(s4graphS1.slot("Path0"));
            Path1S1 = Rcpp::as<std::vector<std::string>>(s4graphS1.slot("Path1"));
            Path2S1 = Rcpp::as<std::vector<std::string>>(s4graphS1.slot("Path2"));
            Path3S1 = Rcpp::as<std::vector<std::string>>(s4graphS1.slot("Path3"));
            Path4S1 = Rcpp::as<std::vector<std::string>>(s4graphS1.slot("Path4"));
            Path5S1 = Rcpp::as<std::vector<std::string>>(s4graphS1.slot("Path5"));

        }

        const std::string & getName() const{
            return name;
        }

        const std::vector<std::string>& getNodeListS0() const {
            return nodeListS0;
        }

        const std::vector<std::string>& getNodeListS1() const {
            return nodeListS1;
        }


        const std::vector<MazeEdge>& getEdgeListS0() const {
            return edgeListS0;
        }

        const std::vector<MazeEdge>& getEdgeListS1() const {
            return edgeListS1;
        }

        const std::unordered_map<std::string, std::vector<std::string>>& getTurnNodeMapS0() const {
            return turnNodeMapS0;
        }

        const std::unordered_map<std::string, std::vector<std::string>>& getTurnNodeMapS1() const {
            return turnNodeMapS1;
        }

        const std::vector<std::string>& getPath0S0() const {
            return Path0S0;
        }

        const std::vector<std::string>& getPath1S0() const {
            return Path1S0;
        }

        const std::vector<std::string>& getPath2S0() const {
            return Path2S0;
        }

        const std::vector<std::string>& getPath3S0() const {
            return Path3S0;
        }

        const std::vector<std::string>& getPath4S0() const {
            return Path4S0;
        }

        const std::vector<std::string>& getPath5S0() const {
            return Path5S0;
        }

        const std::vector<std::string>& getPath6S0() const {
            return Path6S0;
        }

        const std::vector<std::string>& getPath7S0() const {
            return Path7S0;
        }

        const std::vector<std::string>& getPath8S0() const {
            return Path8S0;
        }

        const std::vector<std::string>& getPath9S0() const {
            return Path9S0;
        }

        const std::vector<std::string>& getPath10S0() const {
            return Path10S0;
        }

        const std::vector<std::string>& getPath11S0() const {
            return Path11S0;
        }


        const std::vector<std::string>& getPath0S1() const {
            return Path0S1;
        }

        const std::vector<std::string>& getPath1S1() const {
            return Path1S1;
        }

        const std::vector<std::string>& getPath2S1() const {
            return Path2S1;
        }

        const std::vector<std::string>& getPath3S1() const {
            return Path3S1;
        }

        const std::vector<std::string>& getPath4S1() const {
            return Path4S1;
        }

        const std::vector<std::string>& getPath5S1() const {
            return Path5S1;
        }

        const std::vector<std::vector<int>>& getNodeGroups() const {
            return nodeGroups;
        }

        bool getOptimal()
        {
            return optimal;
        }


    private:
        
        std::string name;
        
        std::vector<std::string> nodeListS0;
        std::vector<MazeEdge> edgeListS0;
        std::unordered_map<std::string, std::vector<std::string>> turnNodeMapS0;
        
        std::vector<std::string> nodeListS1;
        std::vector<MazeEdge> edgeListS1;
        std::unordered_map<std::string, std::vector<std::string>> turnNodeMapS1;

        std::vector<std::string> Path0S0;
        std::vector<std::string> Path1S0;
        std::vector<std::string> Path2S0;
        std::vector<std::string> Path3S0;
        std::vector<std::string> Path4S0;
        std::vector<std::string> Path5S0;
        std::vector<std::string> Path6S0;
        std::vector<std::string> Path7S0;
        std::vector<std::string> Path8S0;
        std::vector<std::string> Path9S0;
        std::vector<std::string> Path10S0;
        std::vector<std::string> Path11S0;

        std::vector<std::string> Path0S1;
        std::vector<std::string> Path1S1;
        std::vector<std::string> Path2S1;
        std::vector<std::string> Path3S1;
        std::vector<std::string> Path4S1;
        std::vector<std::string> Path5S1;


        std::vector<std::vector<int>> nodeGroups;
        bool optimal;



        

};

#endif