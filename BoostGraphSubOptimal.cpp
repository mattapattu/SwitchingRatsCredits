#include <boost/graph/graph_traits.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/subgraph.hpp>
#include <boost/property_map/property_map.hpp>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>


//#include "BoostGraph.h"

struct VertexProperties
{
    std::string node;
    double credit;
    int node_id;

};

struct EdgeProperties
{
    double probability;
};





// class BoostGraphSuboptimal {
// public:
//     // BoostGraphSuboptimal() {
//     //     createMainGraph();
//     //     createSubgraphs();
//     // }

//     virtual ~BoostGraphSuboptimal() = default;

//     virtual void createSubgraphs() = 0;
//     virtual void createMainGraph() = 0; // Make createMainGraph virtual

// private:
//     Graph mainGraph;
// };


class SubOptimalHybrid3  {
public:

    typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::directedS, VertexProperties, boost::property<boost::edge_index_t, int, EdgeProperties>> Graph;
    typedef boost::subgraph<Graph> Subgraph;


    //typedef boost::graph_traits<Graph>::vertex_descriptor Vertex;
    //typedef boost::graph_traits<Graph>::edge_descriptor Edge;
    //typedef boost::subgraph<Graph> SubGraph;
    typedef boost::graph_traits<Subgraph>::vertex_descriptor VertexDescriptor;
    typedef boost::graph_traits<Subgraph>::edge_descriptor EdgeDescriptor;
    
    SubOptimalHybrid3()
    {
        subgraph1 = mainGraph.create_subgraph(), 
        subgraph2 = mainGraph.create_subgraph();
        // Implementation for creating subgraphs

        // Add nodes and edges to Subgraph 1
        // for (const auto& nodeName : {"E","dch", "dcba1", "fga1", "a2kj", "a2gf", "a2bch", "a2bcd"}) {
        //     auto v = boost::add_vertex(E,subgraph1);
        //     subgraph1[v].node = nodeNameToString(nodeName);
        //     subgraph1[v].credit = 0;
        //     subgraph1[v].node_id = 0;
        // }

        auto E = boost::add_vertex(mainGraph);
        mainGraph[E].node = "E";
        mainGraph[E].credit = 0;
        mainGraph[E].node_id = 0;
        auto dch = boost::add_vertex(mainGraph);
        mainGraph[dch].node = "dch";
        mainGraph[dch].credit = 0;
        mainGraph[dch].node_id = 0;
        auto dcba1 = boost::add_vertex(mainGraph);
        mainGraph[dcba1].node = "dcba1";
        mainGraph[dcba1].credit = 0;
        mainGraph[dcba1].node_id = 0;
        auto fga1 = boost::add_vertex(mainGraph);
        mainGraph[fga1].node = "fga1";
        mainGraph[fga1].credit = 0;
        mainGraph[fga1].node_id = 0;
        auto a2kj = boost::add_vertex(mainGraph);
        mainGraph[a2kj].node = "a2kj";
        mainGraph[a2kj].credit = 0;
        mainGraph[a2kj].node_id = 0;
        auto a2gf = boost::add_vertex(mainGraph);
        mainGraph[a2gf].node = "a2gf";
        mainGraph[a2gf].credit = 0;
        mainGraph[a2gf].node_id = 0;
        auto a2bch = boost::add_vertex(mainGraph);
        mainGraph[a2bch].node = "a2bch";
        mainGraph[a2bch].credit = 0;
        mainGraph[a2bch].node_id = 0;
        auto a2bcd = boost::add_vertex(mainGraph);
        mainGraph[a2bcd].node = "a2bcd";
        mainGraph[a2bcd].credit = 0;
        mainGraph[a2bcd].node_id = 0;

        auto I = boost::add_vertex(mainGraph);
        mainGraph[I].node = "I";
        mainGraph[I].credit = 0;
        mainGraph[I].node_id = 0;
        auto hcd = boost::add_vertex(mainGraph);
        mainGraph[hcd].node = "hcd";
        mainGraph[hcd].credit = 0;
        mainGraph[hcd].node_id = 0;
        auto hcba1 = boost::add_vertex(mainGraph);
        mainGraph[hcba1].node = "hcba1";
        mainGraph[hcba1].credit = 0;
        mainGraph[hcba1].node_id = 0;
        auto jka1 = boost::add_vertex(mainGraph);
        mainGraph[jka1].node = "jka1";
        mainGraph[jka1].credit = 0;
        mainGraph[jka1].node_id = 0;

        // boost::add_edge(E, dch,Subgraph::edge_property_type{0, EdgeProperties{0.333}} , mainGraph);
        // boost::add_edge(E, dcba1,Subgraph::edge_property_type{1, EdgeProperties{0.333}} , mainGraph);
        // boost::add_edge(E, fga1, Subgraph::edge_property_type{2, EdgeProperties{0.333}} , mainGraph);
        // boost::add_edge(dcba1, a2kj, Subgraph::edge_property_type{3, EdgeProperties{0.5}} , mainGraph);
        // boost::add_edge(dcba1, a2gf, Subgraph::edge_property_type{4, EdgeProperties{0.5}} , mainGraph);
        // boost::add_edge(fga1, a2bch, Subgraph::edge_property_type{5, EdgeProperties{0.333}} , mainGraph);
        // boost::add_edge(fga1, a2kj, Subgraph::edge_property_type{6, EdgeProperties{0.333}} , mainGraph);
        // boost::add_edge(fga1, a2bcd, Subgraph::edge_property_type{7, EdgeProperties{0.333}} , mainGraph);

        boost::add_edge(E, dch,Subgraph::edge_property_type{0, EdgeProperties{0.333}} , subgraph1);
        boost::add_edge(E, dcba1,Subgraph::edge_property_type{1, EdgeProperties{0.333}} , subgraph1);
        boost::add_edge(E, fga1, Subgraph::edge_property_type{2, EdgeProperties{0.333}} , subgraph1);
        boost::add_edge(dcba1, a2kj, Subgraph::edge_property_type{3, EdgeProperties{0.5}} , subgraph1);
        boost::add_edge(dcba1, a2gf, Subgraph::edge_property_type{4, EdgeProperties{0.5}} , subgraph1);
        boost::add_edge(fga1, a2bch, Subgraph::edge_property_type{5, EdgeProperties{0.333}} , subgraph1);
        boost::add_edge(fga1, a2kj, Subgraph::edge_property_type{6, EdgeProperties{0.333}} , subgraph1);
        boost::add_edge(fga1, a2bcd, Subgraph::edge_property_type{7, EdgeProperties{0.333}} , subgraph1);


        boost::add_edge(I, hcd,Subgraph::edge_property_type{0, EdgeProperties{0.333}} , subgraph2);
        boost::add_edge(I, hcba1,Subgraph::edge_property_type{1, EdgeProperties{0.333}} , subgraph2);
        boost::add_edge(I, jka1, Subgraph::edge_property_type{2, EdgeProperties{0.333}} , subgraph2);
        boost::add_edge(hcba1, a2gf, Subgraph::edge_property_type{3, EdgeProperties{0.5}} , subgraph2);
        boost::add_edge(hcba1, a2kj, Subgraph::edge_property_type{4, EdgeProperties{0.5}} , subgraph2);
        boost::add_edge(jka1, a2bcd, Subgraph::edge_property_type{5, EdgeProperties{0.333}} , subgraph2);
        boost::add_edge(jka1, a2gf, Subgraph::edge_property_type{6, EdgeProperties{0.333}} , subgraph2);
        boost::add_edge(jka1, a2bcd, Subgraph::edge_property_type{7, EdgeProperties{0.333}} , subgraph2);

        

        
        // auto a2kj = boost::add_vertex(subgraph1);
        // subgraph1[a2kj].node = "a2kj";
        // subgraph1[a2kj].credit = 0;
        // subgraph1[a2kj].node_id = 0;
        // auto a2gf = boost::add_vertex(subgraph1);
        // subgraph1[a2gf].node = "a2gf";
        // subgraph1[a2gf].credit = 0;
        // subgraph1[a2gf].node_id = 0;
        // auto a2bch = boost::add_vertex(subgraph1);
        // subgraph1[a2bch].node = "a2bch";
        // subgraph1[a2bch].credit = 0;
        // subgraph1[a2bch].node_id = 0;
        // auto a2bcd = boost::add_vertex(subgraph1);
        // subgraph1[a2bcd].node = "a2bcd";
        // subgraph1[a2bcd].credit = 0;
        // subgraph1[a2bcd].node_id = 0;
    }

    // void printGraph()
    // {
    //     boost::dynamic_properties dp;
    //     dp.property("node_id", boost::get(&VertexProperties::node_id, subgraph1));
    //     dp.property("label", boost::get(&VertexProperties::node, subgraph1));
    //     dp.property("weight", boost::get(&EdgeProperties::probability, subgraph1));
    //     std::ofstream graph_file_out("./out.gv");
    //     boost::write_graphviz_dp(graph_file_out, subgraph1, dp);


    // }

    void printGraph()
    {
        std::cout << "mainGraph:" << std::endl;
        boost::print_graph(mainGraph, get(boost::vertex_index, mainGraph));
        boost::print_edges2(mainGraph, get(boost::vertex_index, mainGraph), get(boost::edge_index, mainGraph));
        std::cout << std::endl;

        Subgraph::children_iterator ci, ci_end;
        int num = 1;
        for (boost::tie(ci, ci_end) = mainGraph.children(); ci != ci_end; ++ci) {
            std::cout << "G" << num++ << ":" << std::endl;
            boost::print_graph(*ci, get(boost::vertex_index, *ci));
            boost::print_edges2(*ci, get(boost::vertex_index, *ci), get(boost::edge_index, *ci));
            std::cout << std::endl;

        }
    }

private:
    Subgraph mainGraph;
    Subgraph subgraph1;
    Subgraph subgraph2;
};

int main() {
    // Example usage
    SubOptimalHybrid3 hybrid3;
    hybrid3.printGraph();

    return 0;
}
