/*
Copyright 2017 Eric Schuyler Fried, Nicolas Per Dane Sawaya

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
        limitations under the License.

*/


/*

Line graph tools. Used primarily for determining contraction orderings.

KEEP IN MIND THAT IF MULTIPLE WIRES CONNECT A NODE PAIR, THEY ARE 
ALL CONTRACTED. HENCE (A) MAKE SURE THEY'RE SWITCHED TO NULL PTRS,
AND (B) TAKE THAT INTO ACCOUNT WITH SUBSEQUENT WIRES IN LIST.

*/

#pragma once

#include <sstream>
#include <fstream>
#include <iostream>
#include "Timer.h"
#include <vector>
#include "Network.h"
#include "Exceptions.h"
#include <array>
#include <sys/stat.h>


namespace qtorch {

    class LineGraph {

    public:
        // void LineGraph();
        LineGraph(std::shared_ptr<Network> origGraph);

        // Run QuickBB to get ordering
        bool runQuickBB(int MaxTimeInSec, Timer *tim = NULL, bool sixtyFourBit = true);

        void Reset(std::shared_ptr<Network> inpNetwork = nullptr);

        // Contracts the network based on linegraph
        bool LGContract();
		
		void SetQBBOutDirectory(std::string& pathToDirectory)
        {
            cnfName = pathToDirectory + std::string("lg.cnf");
            qbbOutName = pathToDirectory + std::string("qbb.out");
            qbbStatsName = pathToDirectory + std::string("qbb-stats.out");
        }
        void SetQBBOutFiles(const std::string& cnfNew, const std::string& qbbOutNew, const std::string& qbbStatsNew)
        {
        	cnfName = cnfNew;
            qbbOutName = qbbOutNew;
            qbbStatsName = qbbStatsNew;
        } 

    private:
        // Resulting line graph is also stored in Network object
        // std::shared_ptr<Network> locLineGraph;
        std::shared_ptr<Network> origNetwork;

        // This function outputs LG and calls QuickBB from command line.
        void GetLGOrdering(int TimeSec);

        // Collect the wires
        std::vector<std::shared_ptr<Wire> > GraphWires;

        // Create actual linegraph (vector of pairs of wires)
        std::vector<std::array<std::shared_ptr<Wire>, 2>> LGEdges;

        // Filenames
        std::string cnfName = "output/lg.cnf";
        std::string qbbOutName = "output/qbb.out";
        std::string qbbStatsName = "output/qbb-stats.out";
    };


    LineGraph::
    LineGraph(std::shared_ptr<Network> inpNetwork) { // added ampersand

        this->origNetwork = inpNetwork;

        /*
        This function creates a linegraph of the inputted graph (i.e. the
        inputted network).

        Once the ordering is complete, we can just contract the wires
        without worrying about what's going on in the background.
        */


        // Loop over nodes
        std::vector<std::shared_ptr<Node>> CurrNodes = inpNetwork->GetUncontractedNodes();
        int numNodes = CurrNodes.size();

        for (int nid = 0; nid < numNodes; nid++) {

            // Loop over wires in this node
            std::vector<std::shared_ptr<Wire>> WiresThisNode = CurrNodes[nid]->GetWires();

            //std::cout << "WiresThisNode.size(): " << WiresThisNode.size() << std::endl;

            for (int wid = 0; wid < WiresThisNode.size(); wid++) {
                std::shared_ptr<Wire> thisWire = WiresThisNode[wid];

                //std::out <<


                // Check if wire is already in the overall wire vector
                if (std::find(GraphWires.begin(), GraphWires.end(), thisWire) == GraphWires.end()) {
                    // Wire not already present

                    // Add mWireID to the wire, which will be useful later.
                    //std::cout << "GraphWires.size(): " << GraphWires.size() << std::endl;
                    thisWire->SetWireID(GraphWires.size());

                    // The vector does not contain the wire. So, insert it.
                    GraphWires.push_back(thisWire);

                    //std::cout << "wireID: " << thisWire->GetWireID() << std::endl;

                }

                //std::cout << (thisWire==GraphWires[0]) << std::endl;
                //std::cout << "GraphWires.size(): " << GraphWires.size() << std::endl;

                // Another inner loop. For a given node, every pair of
                // wires is a newWire* (e.g. a wire on the L(G) ).
                // Have (wid2<wid) to avoid double-counting.
                for (int wid2 = 0; wid2 < wid; wid2++) {

                    //LGEdges.push_back( {thisWire,WiresThisNode[wid2]} );
                    LGEdges.push_back({WiresThisNode[wid2], thisWire});

                }

            }

        }


        std::cout << "GraphWires.size(): " << GraphWires.size() << std::endl;

        /*
        // Print out linegraph
        std::cout << "Line graph:" << std::endl;
        for (int id=0;id<LGEdges.size();id++) {

            std::cout << LGEdges[id][0]->GetWireID() << "  " << LGEdges[id][1]->GetWireID() << std::endl;

        }
        std::cout << std::endl << std::endl;
        */



    }

    void LineGraph::
    Reset(std::shared_ptr<Network> inpNetwork) {
        if (inpNetwork == nullptr) {
            origNetwork->Reset();
            return;
        }
        this->origNetwork = inpNetwork;
        GraphWires.clear();
        LGEdges.clear();

        // Loop over nodes
        std::vector<std::shared_ptr<Node>> CurrNodes = inpNetwork->GetUncontractedNodes();
        int numNodes = CurrNodes.size();

        for (int nid = 0; nid < numNodes; nid++) {

            // Loop over wires in this node
            std::vector<std::shared_ptr<Wire>> WiresThisNode = CurrNodes[nid]->GetWires();

            //std::cout << "WiresThisNode.size(): " << WiresThisNode.size() << std::endl;

            for (int wid = 0; wid < WiresThisNode.size(); wid++) {
                std::shared_ptr<Wire> thisWire = WiresThisNode[wid];

                //std::out <<


                // Check if wire is already in the overall wire vector
                if (std::find(GraphWires.begin(), GraphWires.end(), thisWire) == GraphWires.end()) {
                    // Wire not already present

                    // Add mWireID to the wire, which will be useful later.
                    //std::cout << "GraphWires.size(): " << GraphWires.size() << std::endl;
                    thisWire->SetWireID(GraphWires.size());

                    // The vector does not contain the wire. So, insert it.
                    GraphWires.push_back(thisWire);

                    //std::cout << "wireID: " << thisWire->GetWireID() << std::endl;

                }

                //std::cout << (thisWire==GraphWires[0]) << std::endl;
                //std::cout << "GraphWires.size(): " << GraphWires.size() << std::endl;

                // Another inner loop. For a given node, every pair of
                // wires is a newWire* (e.g. a wire on the L(G) ).
                // Have (wid2<wid) to avoid double-counting.
                for (int wid2 = 0; wid2 < wid; wid2++) {

                    //LGEdges.push_back( {thisWire,WiresThisNode[wid2]} );
                    LGEdges.push_back({WiresThisNode[wid2], thisWire});

                }

            }

        }


        std::cout << "GraphWires.size(): " << GraphWires.size() << std::endl;
    }


    bool LineGraph::
    runQuickBB(int MaxTimeInSec, Timer *tim, bool sixtyFourBit) {  // Default for tim is null

        // Nodes and edges of the LineGraph, NOT of the orig graph
        int nLGNodes = this->GraphWires.size();
        int nLGEdges = this->LGEdges.size();

        // Creates 'output' dir if doesn't already exist
        mkdir("output", 0755);

        // Remove qbbOut
        remove(qbbOutName.c_str());

        // Output to CNF file
        std::ofstream cnfFile(cnfName);
        cnfFile << "p cnf " << nLGNodes << " " << nLGEdges << std::endl;
        for (int id = 0; id < LGEdges.size(); id++) {
            // PLUS 1 IS IMPORTANT, since quickbb is base-one indexing.
            cnfFile << LGEdges[id][0]->GetWireID() + 1 << " ";
            cnfFile << LGEdges[id][1]->GetWireID() + 1;

            // '0' just means end of edge, in cnf format
            cnfFile << " " << 0 << std::endl;
        }
        cnfFile.close();


        // Run quickbb, with specified time
        std::cout << "===== Output from QuickBB =====" << std::endl;
        std::ostringstream cmdss;
        cmdss << "quickbb_";
        if (sixtyFourBit) {
            cmdss << "64";
        } else {
            cmdss << "32";
        }
        cmdss << " --min-fill-ordering --lb --time ";
        cmdss << MaxTimeInSec;
        cmdss << " --outfile ";
        cmdss << qbbOutName;
        cmdss << " --statfile ";
        cmdss << qbbStatsName;
        cmdss << " --cnffile ";
        cmdss << cnfName;
        std::cout << "Executing:   " << cmdss.str() << std::endl;
        system(cmdss.str().c_str());
        std::cout << "===== End of QuickBB output =====" << std::endl << std::endl;

        // Output time, if object was given
        if (tim) {
            std::cout << "Time elapsed after outputting line graph and "
                      << "running QuickBB: { " << tim->getElapsed() << " }\n";
        }

        return true;

    }


    bool LineGraph::
    LGContract() {

        // Nodes and edges of the LineGraph, NOT of the orig graph
        int nLGNodes = this->GraphWires.size();
        int nLGEdges = this->LGEdges.size();


        // Parse qbb file to get ordering
        // this->parseQbbFile(qbbOutName);
        std::ifstream fQbb(qbbOutName);
        if (!fQbb) {
            std::cout << "Unable to open qbb file: "
                      << qbbOutName << std::endl;
            throw QbbFailure();
        }

        std::vector<int> qbbOrder;

        // Parse file
        std::string line;
        bool foundflag = false;
        while (fQbb) {

            std::getline(fQbb, line);

            if (line == " The optimal ordering is ") {

                // Read next line to get ordering
                std::getline(fQbb, line);
                std::stringstream ss(line);

                for (int i = 0; i < this->GraphWires.size(); i++) {
                    int wirenum;
                    ss >> wirenum;
                    qbbOrder.push_back(wirenum);
                }


                // Set flag
                foundflag = true;

                break;
            }

        }

        if (!foundflag) {
            std::cout << "ERROR reading quickbb contr ordering.\n";
            return false;
        }

        //
        std::cout << "The contraction ordering read from qbb (should match above output): \n";
        for (int i = 0; i < qbbOrder.size(); i++) {
            std::cout << qbbOrder[i] << " ";
        }
        std::cout << "\n\n";


        // Contract in specified order
        for (int i = 0; i < qbbOrder.size(); i++) {

            // Note change to BASE-ZERO
            std::shared_ptr<Wire> w = GraphWires[qbbOrder[i] - 1];

            //std::cout << i << " ";

            /* Only try to contract if wire is not already contracted.
               If >1 wire connects nodes A and B, then even wires that
               haven't yet been looped through are contracted. */
            if (!w->IsContracted()) {
                // Threshold deliberately set very large. qbb's ordering
                // is what this function does, regardless of crashing.
                int thresh = 100;
                origNetwork->ContractNodes(w->GetNodeA().lock(), w->GetNodeB().lock(), thresh);

                //std::cout << "Contracting\n";
            }

        }


        // Result of contraction
        std::vector<std::shared_ptr<Node>> remNodes = origNetwork->GetUncontractedNodes();
        //std::cout << " *** " << remNodes.size() << "\n";

        if (remNodes.size() != 1) {
            std::cout << "ERROR. After contraction, there is more than one "
                      << "remaining node.\n";
            throw ContractionFailure();
        }
        std::vector<std::complex<double>> finTensVals = remNodes[0]->GetTensorVals();
        if (finTensVals.size() != 1) {
            std::cout << "ERROR. Final node has more than one value.\n";
            throw ContractionFailure();
        }

        //std::cout << "Number of floating point ops in full contraction: "
        //<< origNetwork->getNumFloatOps() << "\n";

        //std::vector<std::shared_ptr<Node>> & GetUncontractedNodes()
        std::cout << "Result of contraction:\n"
                  << finTensVals[0] << "\n";

        // Note that we're NOT using "GetFinalValue" since it appears
        // to be used only when contraction is done within Network.h




        // Success
        return true;

    }
}












