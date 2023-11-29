
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/subgraph.hpp>


class SubOptimalHybrid3:BoostGraphSuboptimal {
public:
    BoostGraphSuboptimal() {
        createMainGraph(){
            VertexDescriptor vE = boost::add_vertex({"E"}, mainGraph);
            VertexDescriptor vdcba1 = boost::add_vertex({"dcba1"}, mainGraph);
            VertexDescriptor vfga1 = boost::add_vertex({"fga1"}, mainGraph);
            VertexDescriptor va2kj = boost::add_vertex({"a2kj"}, mainGraph);
            VertexDescriptor va2gf = boost::add_vertex({"a2gf"}, mainGraph);
            VertexDescriptor va2bch = boost::add_vertex({"a2bch"}, mainGraph);
            VertexDescriptor va2bcd = boost::add_vertex({"a2bcd"}, mainGraph);
            VertexDescriptor vI = boost::add_vertex({"I"}, mainGraph);
            VertexDescriptor vhcd = boost::add_vertex({"hcd"}, mainGraph);
            VertexDescriptor vhcba1 = boost::add_vertex({"hcba1"}, mainGraph);
            VertexDescriptor vjka1 = boost::add_vertex({"jka1"}, mainGraph);

            // Define edges
            boost::add_edge(vE, vdcba1, {0.333}, mainGraph);
            boost::add_edge(vE, vfga1, {0.333}, mainGraph);
            boost::add_edge(vdcba1, va2kj, {0.5}, mainGraph);
            boost::add_edge(vdcba1, va2gf, {0.5}, mainGraph);
            boost::add_edge(vfga1, va2bch, {0.333}, mainGraph);
            boost::add_edge(vfga1, va2kj, {0.333}, mainGraph);
            boost::add_edge(vfga1, va2bcd, {0.333}, mainGraph);

            boost::add_edge(vI, vhcd, {0.333}, mainGraph);
            boost::add_edge(vI, vhcba1, {0.333}, mainGraph);
            boost::add_edge(vI, vjka1, {0.333}, mainGraph);
            boost::add_edge(vhcba1, va2gf, {0.5}, mainGraph);
            boost::add_edge(vhcba1, va2kj, {0.5}, mainGraph);
            boost::add_edge(vjka1, va2bch, {0.333}, mainGraph);
            boost::add_edge(vjka1, va2gf, {0.333}, mainGraph);
            boost::add_edge(vjka1, va2bcd, {0.333}, mainGraph);

        }
    }

    virtual ~BoostGraphSuboptimal() = default;

    // Make this function pure virtual
    void createSubgraphs(){
            subgraph1 = SubGraph(mainGraph, boost::vertex("E", mainGraph), boost::vertex("fga1", mainGraph));
            subgraph2 = SubGraph(mainGraph, boost::vertex("I", mainGraph), boost::vertex("jka1", mainGraph));

            // Add nodes and edges to Subgraph 1
            for (const auto& nodeName : {"E", "dch", "dcba1", "fga1", "a2kj", "a2gf", "a2bch", "a2bcd"}) {
                VertexDescriptor v = boost::add_vertex({nodeName}, subgraph1);
            }

            // Add nodes and edges to Subgraph 2
            for (const auto& nodeName : {"I", "hcd", "hcba1", "jka1", "a2gf", "a2kj", "a2bch", "a2bcd"}) {
                VertexDescriptor v = boost::add_vertex({nodeName}, subgraph2);
            }

    };

private:
    Graph mainGraph;
    SubGraph subgraph1;
    SubGraph subgraph2;

    void createMainGraph() {
        // Define vertices
    }
};


        boost::add_edge(E, DCH,Subgraph::edge_property_type{0, EdgeProperties{0.333}} , subgraph1);
        boost::add_edge(E, DCBA1,Subgraph::edge_property_type{1, EdgeProperties{0.333}} , subgraph1);
        boost::add_edge(E, FGA1, Subgraph::edge_property_type{2, EdgeProperties{0.333}} , subgraph1);
        boost::add_edge(DCBA1, A2KJ, Subgraph::edge_property_type{3, EdgeProperties{0.5}} , subgraph1);
        boost::add_edge(DCBA1, A2GF, Subgraph::edge_property_type{4, EdgeProperties{0.5}} , subgraph1);
        boost::add_edge(FGA1, A2BCH, Subgraph::edge_property_type{5, EdgeProperties{0.333}} , subgraph1);
        boost::add_edge(FGA1, A2KJ, Subgraph::edge_property_type{6, EdgeProperties{0.333}} , subgraph1);
        boost::add_edge(FGA1, A2BCD, Subgraph::edge_property_type{7, EdgeProperties{0.333}} , subgraph1);



        // Add nodes and edges to Subgraph 2
        for (const auto& nodeName : {I, HCD, HCBA1, JKA1, A2GF, A2KJ, A2BCH, A2BCD}) {
            auto v = boost::add_vertex(subgraph2);
            subgraph2[v].node = nodeName;
            subgraph2[v].credit = 0;
            subgraph2[v].node_id = 0;
        }

        boost::add_edge(I, HCD, Subgraph::edge_property_type{0, EdgeProperties{0.333}}, subgraph2);
        boost::add_edge(I, HCBA1, Subgraph::edge_property_type{1, EdgeProperties{0.333}}, subgraph2);
        boost::add_edge(I, JKA1, Subgraph::edge_property_type{2, EdgeProperties{0.333}}, subgraph2);
        boost::add_edge(HCBA1, A2GF, Subgraph::edge_property_type{3, EdgeProperties{0.5}}, subgraph2);
        boost::add_edge(HCBA1, A2KJ, Subgraph::edge_property_type{4, EdgeProperties{0.5}}, subgraph2);
        boost::add_edge(JKA1, A2BCH, Subgraph::edge_property_type{5, EdgeProperties{0.333}}, subgraph2);
        boost::add_edge(JKA1, A2GF, Subgraph::edge_property_type{6, EdgeProperties{0.333}}, subgraph2);
        boost::add_edge(JKA1, A2BCD, Subgraph::edge_property_type{7, EdgeProperties{0.333}}, subgraph2);

    


        //VertexDescriptor vE = boost::add_vertex(mainGraph);
        // VertexDescriptor vdcba1 = boost::add_vertex({"dcba1"}, mainGraph);
        // VertexDescriptor vfga1 = boost::add_vertex({"fga1"}, mainGraph);
        // VertexDescriptor va2kj = boost::add_vertex({"a2kj"}, mainGraph);
        // VertexDescriptor va2gf = boost::add_vertex({"a2gf"}, mainGraph);
        // VertexDescriptor va2bch = boost::add_vertex({"a2bch"}, mainGraph);
        // VertexDescriptor va2bcd = boost::add_vertex({"a2bcd"}, mainGraph);
        // VertexDescriptor vI = boost::add_vertex({"I"}, mainGraph);
        // VertexDescriptor vhcd = boost::add_vertex({"hcd"}, mainGraph);
        // VertexDescriptor vhcba1 = boost::add_vertex({"hcba1"}, mainGraph);
        // VertexDescriptor vjka1 = boost::add_vertex({"jka1"}, mainGraph);

        // // Define edges
        //auto src = findNode(srcNodeName);
        //auto dest = findNode(destNodeName);

        // boost::add_edge(vE, vdcba1, {0.333}, mainGraph);
        // boost::add_edge(vE, vfga1, {0.333}, mainGraph);
        // boost::add_edge(vdcba1, va2kj, {0.5}, mainGraph);
        // boost::add_edge(vdcba1, va2gf, {0.5}, mainGraph);
        // boost::add_edge(vfga1, va2bch, {0.333}, mainGraph);
        // boost::add_edge(vfga1, va2kj, {0.333}, mainGraph);
        // boost::add_edge(vfga1, va2bcd, {0.333}, mainGraph);

        // boost::add_edge(vI, vhcd, {0.333}, mainGraph);
        // boost::add_edge(vI, vhcba1, {0.333}, mainGraph);
        // boost::add_edge(vI, vjka1, {0.333}, mainGraph);
        // boost::add_edge(vhcba1, va2gf, {0.5}, mainGraph);
        // boost::add_edge(vhcba1, va2kj, {0.5}, mainGraph);
        // boost::add_edge(vjka1, va2bch, {0.333}, mainGraph);
        // boost::add_edge(vjka1, va2gf, {0.333}, mainGraph);
        // boost::add_edge(vjka1, va2bcd, {0.333}, mainGraph);

