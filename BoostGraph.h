#ifndef BOOSTGRAPH_H
#define BOOSTGRAPH_H

#include "MazeGraph.h"
#include <iostream>
#include <vector>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>
#include <cstdlib>
#include <stdexcept>
//#include <mutex>



struct VertexProperties
{
    std::string node;
    double credit;
    int node_id;
    double eligibilityTrace;

};

struct EdgeProperties
{
    double probability; //log probability
};

class BoostGraph
{
public:
    typedef boost::adjacency_list<
        boost::vecS,      // OutEdgeList type
        boost::vecS,      // VertexList type
        boost::directedS, // Directed/Undirected graph
        VertexProperties, // Vertex properties
        EdgeProperties    // Edge properties
        >
        Graph;

    typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    typedef boost::graph_traits<Graph>::edge_descriptor Edge;
    typedef boost::graph_traits<Graph>::edge_iterator edge_iterator;
    typedef boost::graph_traits<Graph>::vertex_iterator vertex_iterator;

    BoostGraph() {
        //std::lock_guard<std::mutex> lck (mutex_);

    }

    BoostGraph(const MazeGraph& turnModel, int state, bool optimal)
    {

        std::vector<std::string> rcppNodeList;
        std::vector<MazeEdge> rcppEdgeList;
        std::unordered_map<std::string, std::vector<std::string>> turnNodes;

        if (state == 0)
        {
            rcppNodeList = turnModel.getNodeListS0();
            rcppEdgeList = turnModel.getEdgeListS0();
            turnNodes = turnModel.getTurnNodeMapS0();
            Path0 = turnModel.getPath0S0();
            Path1 = turnModel.getPath1S0();
            Path2 = turnModel.getPath2S0();
            Path3 = turnModel.getPath3S0();
            Path4 = turnModel.getPath4S0();
            Path5 = turnModel.getPath5S0();
            if(!optimal)
            {
                Path0S1 = turnModel.getPath6S0();
                Path1S1 = turnModel.getPath7S0();
                Path2S1 = turnModel.getPath8S0();
                Path3S1 = turnModel.getPath9S0();
                Path4S1 = turnModel.getPath10S0();
                Path5S1 = turnModel.getPath11S0();
            }

            
        }
        else
        {
            if (optimal) {
                //For Suboptimal_Hybrid3 (and other suboptimal models)
                rcppNodeList = turnModel.getNodeListS1();
                rcppEdgeList = turnModel.getEdgeListS1();
                turnNodes = turnModel.getTurnNodeMapS1();
                Path0 = turnModel.getPath0S1();
                Path1 = turnModel.getPath1S1();
                Path2 = turnModel.getPath2S1();
                Path3 = turnModel.getPath3S1();
                Path4 = turnModel.getPath4S1();
                Path5 = turnModel.getPath5S1();
            }else{

                return;
            }
            

        }


        for (int i = 0; i < rcppNodeList.size(); i++)
        {
            addNode(rcppNodeList[i], i);
        }
        
        for (const MazeEdge& edge : rcppEdgeList) 
        {
         
          addEdge(edge.src, edge.dest, std::log(edge.prob));
          // Access other members as needed
        }


        // std::cout << "BoostGraph:" << " state=" << state << ", optimal=" << optimal << std::endl;
        // std::cout << "Edges: ";
        // boost::print_edges(graph, get(boost::vertex_index, graph));
        // std::cout << std::endl;

        // // Print vertices
        // std::cout << "Vertices: ";
        // boost::print_vertices(graph, get(&VertexProperties::node, graph));
        // std::cout << std::endl;

        // // Print the entire graph
        // std::cout << "Graph: ";
        // boost::print_graph(graph, get(&VertexProperties::node, graph));
        // std::cout << std::endl;

    }

//     ~BoostGraph() {
//     // Destroy the mutex
//     mutex_.unlock();
//   }


    void addNode(const std::string &nodeName, int node_id, double initialCredit = 0.0)
    {
        Vertex v = boost::add_vertex(graph);
        graph[v].node = nodeName;
        graph[v].credit = initialCredit;
        graph[v].node_id = node_id;
        graph[v].eligibilityTrace = 0;
    }

    void addEdge(const std::string &srcNodeName, const std::string &destNodeName, double probability)
    {
        //std::cout << "Adding edge between src:" <<srcNodeName << " and dest:" << destNodeName <<std::endl; 
        Vertex src = findNode(srcNodeName);
        Vertex dest = findNode(destNodeName);
        if (src != boost::graph_traits<Graph>::null_vertex() &&
            dest != boost::graph_traits<Graph>::null_vertex())
        {
            boost::add_edge(src, dest, EdgeProperties{probability}, graph);
        }
    }

    Vertex findNode(const std::string &nodeName)
    {
        vertex_iterator v, vend;
        for (boost::tie(v, vend) = boost::vertices(graph); v != vend; ++v)
        {
            if (graph[*v].node == nodeName)
            {
                return *v;
            }
        }
        return boost::graph_traits<Graph>::null_vertex();
    }

    Vertex findNodeById(const int& id)
    {
        vertex_iterator v, vend;
        for (boost::tie(v, vend) = boost::vertices(graph); v != vend; ++v)
        {
            if (graph[*v].node_id == id)
            {
                return *v;
            }
        }
        return boost::graph_traits<Graph>::null_vertex();
    }

    Edge findEdge(const Vertex &src , const Vertex &dest)
    {
        Edge edge;  
        if (src != boost::graph_traits<Graph>::null_vertex() && dest != boost::graph_traits<Graph>::null_vertex())
        {
            edge_iterator e, eend;
            for (boost::tie(e, eend) = boost::edges(graph); e != eend; ++e)
            {
                if (boost::source(*e, graph) == src && boost::target(*e, graph) == dest)
                {
                    edge = *e;
                    break;
                }
            }
        }
        return edge;
    }

    double getEdgeProbability(const Edge &edge)
    {
        return graph[edge].probability;
    }

    double getNodeCredits(const Vertex &node)
    {
        return graph[node].credit;
    }

    int getNodeId(const Vertex &node)
    {
        return graph[node].node_id;
    }

    std::string getNodeName(const Vertex &node)
    {
        return graph[node].node;
    }


    void setNodeCredits(const Vertex &node, double credits)
    {
        graph[node].credit = credits;
    }

    
    double getEligibilityTrace(const Vertex &node)
    {
        return graph[node].eligibilityTrace;
    }

    
    void setEligibilityTrace(const Vertex &node, double trace)
    {
        //std::cout << "Setting eligibility trace of " << graph[node].node << " to " << trace << "\n";
        graph[node].eligibilityTrace = trace;
    }

    void updateAllEligibilityTraces(double factor)
    {
        //std::cout << "Updating all eligibility trace by factor = " << factor << "\n";
        for (auto vi = boost::vertices(graph); vi.first != vi.second; ++vi.first) {
            graph[*vi.first].eligibilityTrace *= factor ;
        }
    }

    void tdUpdateAllVertexCredits(double alpha, double td_err)
    {
        for (auto vi = boost::vertices(graph); vi.first != vi.second; ++vi.first) {
            graph[*vi.first].credit += (alpha*td_err*graph[*vi.first].eligibilityTrace) ;
            
            if (std::isinf(graph[*vi.first].credit)) {
                // std::cout << "Node: " << graph[*vi.first].node << " credit is infinity. Check" << std::endl;
                throw std::runtime_error("Error infinite credit val");
            }

        }
    }


    void setEdgeProbability(const Edge &edge, double probability)
    {
        graph[edge].probability = probability;
    }

    void setVertexCredits(const std::vector<double> &creditVector) {
        // Iterate through the vertices
        for (auto vd = boost::vertices(graph).first; vd != boost::vertices(graph).second; ++vd) {
            // Access the vertex descriptor and properties
            Vertex vertex = *vd;
            graph[vertex].credit = creditVector[graph[vertex].node_id];         
        }
    }

    std::vector<double> getVertexCredits()
    {
        std::vector<double> vertexCredits(boost::num_vertices(graph), 0.0);
        for (auto vd = boost::vertices(graph).first; vd != boost::vertices(graph).second; ++vd) {
            // Access the vertex descriptor and properties
            Vertex vertex = *vd;
            int nodeId = graph[vertex].node_id;
            double nodeCredits = graph[vertex].credit;
            vertexCredits[nodeId] = nodeCredits;
        
        }

        return vertexCredits;
    }

    


    Vertex findParent(const Vertex &node)
    {
        // Iterate through the vertices to find the parent
        Vertex parent = boost::graph_traits<Graph>::null_vertex();
        Vertex target = boost::graph_traits<Graph>::null_vertex();
        Graph::out_edge_iterator outIt, outEnd;
        vertex_iterator v, vend;
        for (boost::tie(v, vend) = boost::vertices(graph); v != vend; ++v)
        {
          for (boost::tie(outIt, outEnd) = boost::out_edges(*v, graph); outIt != outEnd; ++outIt)
            {
                target = boost::target(*outIt, graph);
                if(target == node)
                {
                    parent = boost::source(*outIt, graph);
                    break; // Assuming each node has only one parent
                }
                
            }
        }

        
        return parent;
    }

    Vertex findMaxCreditSibling(const Vertex &node)
    {
        double maxCredits = -std::numeric_limits<double>::infinity();
        Vertex maxCreditSibling = boost::graph_traits<Graph>::null_vertex();

        // Get the parent node of the given node
        Vertex parent = findParent(node);

        if (parent != boost::graph_traits<Graph>::null_vertex())
        {
            // Iterate through the adjacent nodes (siblings)
            boost::graph_traits<Graph>::adjacency_iterator adjIt, adjEnd;
            for (boost::tie(adjIt, adjEnd) = boost::adjacent_vertices(parent, graph); adjIt != adjEnd; ++adjIt)
            {
                double credits = getNodeCredits(*adjIt);
                if (credits > maxCredits)
                {
                    maxCredits = credits;
                    maxCreditSibling = *adjIt;
                }
            }
        }

        return maxCreditSibling;
    }

    bool isTerminalVertex(const std::string& vertexName) {
        Vertex v = findNode(vertexName); // Assuming getNode returns the vertex descriptor for the given vertexName

        // Get the out-degree of the vertex
        int outDegree = out_degree(v, graph);

        // If the out-degree is 0, the vertex is a terminal vertex
        return outDegree == 0;
    }

    std::vector<Vertex> getAllVertices() const {
        std::vector<Vertex> vertices;
        for (auto vi = boost::vertices(graph); vi.first != vi.second; ++vi.first) {
            vertices.push_back(*vi.first);
        }
        return vertices;
    }


    Vertex getChildWithMaxCredit(const std::string& parentVertexName) {
        Vertex parentVertex = findNode(parentVertexName); // Get the vertex descriptor of the parent node

        // Initialize variables to keep track of the child with max credit
        Vertex maxCreditChild = boost::graph_traits<Graph>::null_vertex();
        double maxCredit = -std::numeric_limits<double>::infinity(); // Initialize with negative infinity

        // Iterate through the outgoing edges of the parent vertex
        Graph::out_edge_iterator ei, ei_end;
        for (tie(ei, ei_end) = out_edges(parentVertex, graph); ei != ei_end; ++ei) {
            Vertex childVertex = target(*ei, graph); // Get the target vertex of the outgoing edge

            // Get the credit of the child node
            double childCredit = getNodeCredits(childVertex); // Implement this function to get the credit of a node

            // If the child has higher credit, update the max credit and maxCreditChild
            if (childCredit > maxCredit) {
                maxCredit = childCredit;
                maxCreditChild = childVertex;
            }
        }

        return maxCreditChild;
    }

    void printNodeCredits() const {
        //std::cout << "Node Credits:\n";
        vertex_iterator vi, vi_end;
        for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
            Vertex v = *vi;
            std::cout << graph[v].node << ": " << graph[v].credit << ",";
        }

        std::cout << std::endl;
    }

    void printNodeProbabilities() {
        
        for (vertex_iterator it = vertices(graph).first; it != vertices(graph).second; ++it)
        {
            Vertex node = *it;

            std::vector<Vertex> children;
            Graph::out_edge_iterator ei, ei_end;
            //double sumExponentials = 0.0;
            std::vector<double> values;
            for (tie(ei, ei_end) = out_edges(node, graph); ei != ei_end; ++ei) {
                Vertex childVertex = target(*ei, graph); // Get the target vertex of the outgoing                
                children.push_back(childVertex);
                //sumExponentials += std::exp(graph[childVertex].credit);
                values.push_back(graph[childVertex].credit);
            }

            
            for (Vertex child : children)
            {
                // Calculate the softmax probability
                //double softmaxProbability = std::exp(graph[child].credit) / sumExponentials;
                double logsumexp_value = logsumexp(values);


                // Update the probability of the edge
                double prob = exp(graph[child].credit-logsumexp_value);
                std::cout << graph[node].node << "-->"  << graph[child].node << ", prob=" << prob << "; ";

            }

        }
        std::cout << std::endl;
    }

    void printNodeEligibilityTraces() const {
        std::cout << "Node eligibility trace:\n";
        vertex_iterator vi, vi_end;
        for (boost::tie(vi, vi_end) = boost::vertices(graph); vi != vi_end; ++vi) {
            Vertex v = *vi;
            std::cout << "Vertex " << graph[v].node << ": " << graph[v].eligibilityTrace << ",";
        }

        std::cout << std::endl;
    }


    void decayCredits(double gamma)
    {
        for (vertex_iterator it = vertices(graph).first; it != vertices(graph).second; ++it)
        {
            Vertex node = *it;
            graph[node].credit *= gamma;
        }
    }

    std::vector<Vertex> findSiblings(Vertex node)
    {
        std::vector<Vertex> siblings;

        // Get the parent node of the given node
        Vertex parent = findParent(node);
        // Get the outgoing edges from the current node
        Graph::out_edge_iterator it, end;
        for (boost::tie(it, end) = out_edges(parent, graph); it != end; ++it)
        {
            Vertex sibling = boost::target(*it, graph);
            siblings.push_back(sibling);
        }

        return siblings;
    }

    std::vector<Edge> getOutGoingEdges(Vertex node)
    {
        std::vector<Edge> outgoingEdges;
        Graph::out_edge_iterator edgeIt, edgeEnd;
        for (tie(edgeIt, edgeEnd) = out_edges(node, graph); edgeIt != edgeEnd; ++edgeIt) {
            outgoingEdges.push_back(*edgeIt);
        }

        return outgoingEdges;
    }


    // Function to compute softmax probabilities for a vector
    std::vector<double> softmax(const std::vector<double>& values) {
        std::vector<double> probabilities;
        double sumExp = 0.0;

        for (double value : values) {
            sumExp += std::exp(value);
        }

        for (double value : values) {
            probabilities.push_back(std::exp(value) / sumExp);
        }

        return probabilities;
    }


    // Function to sample a vertex based on softmax probabilities
    Vertex sampleChild(Vertex parent) {

        // Get the out-edges of the vertex
        Graph::out_edge_iterator edgeIt, edgeEnd;
        std::vector<double> edgeCredits;
        std::vector<Vertex> children;

        for (tie(edgeIt, edgeEnd) = out_edges(parent, graph); edgeIt != edgeEnd; ++edgeIt) {
            Vertex child = target(*edgeIt, graph);
            children.push_back(child);
            edgeCredits.push_back(graph[child].credit);
        }


        // Compute the softmax probabilities of the credits
        std::vector<double> probs = softmax(edgeCredits);

    //    std::string parentName = getNodeName(parent);
    //    std::cout << "edge.src="<<parentName;
    //     for (int i=0; i < probs.size(); i++)
    //     {
    //         std::string dest  = getNodeName(children[i]);
    //         std::cout << ", edge.dest=" << dest <<  ", prob=" << probs[i] << "; " ;
    //     }
    //     std::cout << "\n" ;


        // Sample one target vertex based on the softmax probabilities
        std::random_device rd;
        std::mt19937 gen(rd());
        std::discrete_distribution<> dist(probs.begin(), probs.end());
        Vertex sampled = children[dist(gen)];
        return(sampled);
    }



    void updateEdgeProbabilitiesSoftmax()
    {
        // Iterate through each node in the graph
        for (vertex_iterator it = vertices(graph).first; it != vertices(graph).second; ++it)
        {
            Vertex node = *it;

            std::vector<Vertex> children;
            Graph::out_edge_iterator ei, ei_end;
            //double sumExponentials = 0.0;
            std::vector<double> values;
            for (tie(ei, ei_end) = out_edges(node, graph); ei != ei_end; ++ei) {
                Vertex childVertex = target(*ei, graph); // Get the target vertex of the outgoing                
                children.push_back(childVertex);
                //sumExponentials += std::exp(graph[childVertex].credit);
                values.push_back(graph[childVertex].credit);
            }

            
            for (Vertex child : children)
            {
                Edge edge = findEdge(node, child);
                // Calculate the softmax probability
                //double softmaxProbability = std::exp(graph[child].credit) / sumExponentials;
                double logsumexp_value = logsumexp(values);


                // Update the probability of the edge
                graph[edge].probability = graph[child].credit-logsumexp_value;

                if (std::isnan(graph[edge].probability)) {
                    
                    // std::cout << "Node credits: " ;
                    // for (const double& p : values) {
                    //     std::cout << p << " ";
                    // }
                    // std::cout << std::endl;

                    // std::cout << "Edge src: " << graph[node].node << " dest: "  << graph[child].node << " logprob is nan. Check" << std::endl;
                    throw std::runtime_error("Error nan probability");
                }

                if (std::isinf(graph[edge].probability)) {
                    // std::cout << "Edge src: " << graph[node].node << " dest: "  << graph[child].node << " logprob is infinity. Check" << std::endl;
                    throw std::runtime_error("Error infinite probability");
                }

            }

        }
    }

    double logsumexp(const std::vector<double>& values) {
        if (values.empty()) {
            return -std::numeric_limits<double>::infinity();
        }

        double max_val = values[0];
        for (double val : values) {
            if (val > max_val) {
                max_val = val;
            }
        }

        double sum_exp = 0.0;
        for (double val : values) {
            sum_exp += std::exp(val - max_val);
        }

        return max_val + std::log(sum_exp);
    }


    std::vector<std::string> getTurnsFromPaths(int path, int state, bool optimal)
    {
        //std::lock_guard<std::mutex> lck (mutex_);
        std::vector<std::string> turns;
        // Rcpp::Rcout << "path=" << path <<std::endl;
        
        // If not optimal and curr_state == 1
        if(!optimal && state ==1)
        {
            if (path == 0)
            {
                turns = Path0S1;
            }
            else if (path == 1)
            {
                turns = Path1S1;
            }
            else if (path == 2)
            {
                turns = Path2S1;
            }
            else if (path == 3)
            {
                turns = Path3S1;
            }
            else if (path == 4)
            {
                turns = Path4S1;
            }
            else if (path == 5)
            {
                turns = Path5S1;
            }

        }else{
            if (path == 0)
            {
                turns = Path0;
            }
            else if (path == 1)
            {
                turns = Path1;
            }
            else if (path == 2)
            {
                turns = Path2;
            }
            else if (path == 3)
            {
                turns = Path3;
            }
            else if (path == 4)
            {
                turns = Path4;
            }
            else if (path == 5)
            {
                turns = Path5;
            }

        }

        return (turns);
    }

    int getPathFromTurns(std::vector<std::string> turns, Vertex rootNode, bool optimal)
    {
        //std::lock_guard<std::mutex> lck (mutex_);
        int path = -1;
        if(!optimal && getNodeName(rootNode) == "I")
        {
            if (turns == Path0S1)
            {
                path = 0;
            }
            else if (turns == Path1S1)
            {
                path = 1;
            }
            else if (turns == Path2S1)
            {
                path = 2;
            }
            else if (turns == Path3S1)
            {
                path = 3;
            }
            else if (turns == Path4S1)
            {
                path = 4;
            }
            else if (turns == Path5S1)
            {
                path = 5;
            }

        }else{
            if (turns == Path0)
            {
                path = 0;
            }
            else if (turns == Path1)
            {
                path = 1;
            }
            else if (turns == Path2)
            {
                path = 2;
            }
            else if (turns == Path3)
            {
                path = 3;
            }
            else if (turns == Path4)
            {
                path = 4;
            }
            else if (turns == Path5)
            {
                path = 5;
            }

        }
        
        return (path);
    }

    std::vector<std::string> getTurnNodes(std::string nodeName)
    {
        //std::lock_guard<std::mutex> lck (mutex_);
        
        std::vector<std::string> turnNodesVec;
        auto it = turnNodes.find(nodeName);
    
        if (it != turnNodes.end() && !it->second.empty()) {
            // Return the first element of the vector (or modify as needed)
            turnNodesVec =  it->second;
        } 

        return turnNodesVec;

    }

    // void printGraph()
    // {
    //     vertex_iterator v, vend;
    //     edge_iterator e, eend;
    //     for (boost::tie(v, vend) = boost::vertices(graph); v != vend; ++v)
    //     {
    //         std::cout << "Node: " << graph[*v].node << " (Credit: " << graph[*v].credit << ")\n";
    //         for (boost::tie(e, eend) = boost::out_edges(*v, graph); e != eend; ++e)
    //         {
    //             Vertex src = boost::source(*e, graph);
    //             Vertex dest = boost::target(*e, graph);
    //             std::cout << "  Edge to: " << graph[dest].node << " (Probability: " << graph[*e].probability << ")\n";
    //         }
    //     }
    // }

    void printGraph()
    {
        boost::dynamic_properties dp;
        dp.property("node_id", boost::get(&VertexProperties::node_id, graph));
        dp.property("label", boost::get(&VertexProperties::node, graph));
        dp.property("weight", boost::get(&EdgeProperties::probability, graph));
        std::ofstream graph_file_out("./out.gv");
        boost::write_graphviz_dp(graph_file_out, graph, dp);


    }

    void resetNodeCredits() {

        // Iterate over all vertices and reset credits
        vertex_iterator v, vend;
        for (boost::tie(v, vend) = boost::vertices(graph); v != vend; ++v)
        {
            graph[*v].credit = 0.0; // Resetting the credits (modify based on your actual property)
        }
    }

    


private:
    Graph graph;
    std::vector<std::string> Path0;
    std::vector<std::string> Path1;
    std::vector<std::string> Path2;
    std::vector<std::string> Path3;
    std::vector<std::string> Path4;
    std::vector<std::string> Path5;
    std::vector<std::string> Path0S1;
    std::vector<std::string> Path1S1;
    std::vector<std::string> Path2S1;
    std::vector<std::string> Path3S1;
    std::vector<std::string> Path4S1;
    std::vector<std::string> Path5S1;
    std::unordered_map<std::string, std::vector<std::string>> turnNodes;
    //std::mutex mutex_;



};

// int main()
// {
//     RInside R;
//     std::string cmd = "load('/home/mattapattu/Projects/Rats-Credit/Sources/lib/TurnsNew/src/Hybrid3.Rdata')";
//     R.parseEvalQ(cmd);                  
//     Rcpp::S4 Hybrid3 = R.parseEval("get('Hybrid3')");
//     BoostGraph graph(Hybrid3, 1);
//     graph.printGraph();

//     return 0;
// }


#endif