/*
Copyright 2017 Eric Schuyler Fried, Nicolas Per Dane Sawaya, Alán Aspuru-Guzik

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


#pragma once





/*
 * This Network class implements a framework for holding an entire tensor network. The class offers functions to
 * contract two tensors, contract the network in the order of the network creation,
 * reduce the circuit to non-repeating 2 qubit gates, localize gates in the circuit by adding
 * swap gates, output the network to a visual graph, output the network to a graph that can be analyzed for treewidth
 * and reset the network. The rest of the functions are minor - please see implementations below
 *
 * The data contained in the network class is also explained below. Please use the parallelizer wrapper class
 */

#include <algorithm>  //npds
#include "Node.h"
#include "Timer.h"
#include <regex>
#include <fstream>
#include <random>
#include <mutex>
#include <iostream>
#include <csignal>
#include "Exceptions.h"


namespace qtorch {

#define THRESH_RANK_THREAD 8  // If rank of resulting threshold is >= this, it will use pthread.
    Timer totTimer;
    double maxTime(60.0);


    class Network {
    public:
        Network();  // Empty constructor, npds 2feb2017
        Network(const std::string &inputFile, const std::string &measureFile);

        std::shared_ptr<Node> ContractNodes(std::shared_ptr<Node> nodeA, std::shared_ptr<Node> nodeB, int threshold);

        const std::complex<double> &GetFinalValue() const noexcept { return mFinalVal; };

        void ContractNetworkLinearly();

        const std::vector<std::shared_ptr<Node>> &GetAllNodes() const noexcept { return mAllNodes; };

        void SetNumThreads(
                const int numThreads)noexcept { mNumberOfThreads = numThreads; }; //change the number of threads for large tensor contraction to a number other then the default (8)
        const int GetNumQubits() const noexcept { return mNumberOfQubits; };

        const std::vector<std::shared_ptr<Node>> &GetUncontractedNodes() const noexcept { return mUncontractedNodes; };

        const bool IsDone()noexcept { return mDone; };

        void MoveInitialStatesToBack();

        void ReduceCircuit();

        void LocalizeInteractions(const std::string &logFile);

        void OutputCircuitToVisualGraph(const std::string &toOutputTo) const;

        void OutputCircuitToTreewidthGraph(const std::string &toOutputTo) const;

        const bool HasFailed() const noexcept { return mFailure; };

        void Reset();

        const std::string &GetInputQasm() const noexcept { return mInputFile; };

        void resetFloatCounter() noexcept { mNumFloatOps = 0; };

        long long getNumFloatOps() noexcept { return mNumFloatOps; };
    private:
        std::vector<std::shared_ptr<Node>> mNetworkParsingNodes; //keeps a vector of all of the nodes you're currently working on when contracting
        std::vector<std::shared_ptr<Wire>> mNetworkParsingWires; //keeps track of the furthest wire on each line in the circuit when building (**** becomes useless after building ****)
        std::string mInputFile; //the path to the input qasm file
        std::string mMeasureFile; //the path to the input measure file
        std::complex<double> mFinalVal; //the final expectation value of the network
        std::mutex mLocker; //mutex to avoid errors when parallelizing calls to contract nodes
        int mNumberOfQubits; //the number of qubits in the network
        int mDepth; //the depth of the circuit - only used when the localize interactions function is called
        bool mDone{false}; //to determine whether the network is fully contracted or not
        bool mFailure{false}; //if the network fails to contract for some reason
        std::vector<std::shared_ptr<Node>> mAllNodes; //a vector with all the nodes in the circuit, including ones that have already been contracted.
        std::vector<std::vector<std::shared_ptr<Node>>> mNodesByWire; //a matrix that contains the nodes in the circuit in their respective places - a way to realize the 2d circuit
        std::vector<std::shared_ptr<Node>> mUncontractedNodes; //a vector with just the nodes that haven't been contracted yet
        std::unordered_map<std::string, std::string> mArbitraryOneQubitGates; //a vector with arbitrary one qubit gates that have been defined in the qasm file - see the node class for more info on this
        std::unordered_map<std::string, std::string> mArbitraryTwoQubitGates; //a vector with arbitrary two qubit gates that have been defined in the qasm file - see the node class for more info on this
        long long mNumFloatOps{
                0}; // Counting floating ops. Should probably be reset after the simple "network reduction" routine.
        int mNumberOfThreads{8};
    protected:
        inline void ContractIndices(const std::vector<std::pair<bool, int>> &toNotSumOn,
                                    const std::vector<std::pair<int, int>> &toSumOn,
                                    std::vector<int> &vectorIndexA,
                                    std::vector<int> &vectorIndexB,
                                    std::shared_ptr<Node> nodeA,
                                    std::shared_ptr<Node> nodeB,
                                    std::shared_ptr<Node> nodeC);

        void ParseTokens(std::string &input, std::vector<std::string> &output);

        void ParseNetwork(const std::string &inputFile);

        void ParseNode(std::string &inputLine);

        void CreateInitialStates();

        void AddMeasurementsOrTrace(std::vector<char> &measurements);

        void OutputCircuit(const std::vector<std::shared_ptr<Node>> &toOutput, const std::string &logFile) const;

        void FindAndReplace(std::vector<std::vector<std::shared_ptr<Node>>> &toSearch,
                            std::shared_ptr<Node> toFind, std::shared_ptr<Node> toReplaceWith) const;

        void FindAndReplace(std::vector<std::shared_ptr<Node>> &toSearch, std::shared_ptr<Node> toFind,
                            std::shared_ptr<Node> toReplaceWith) const;

        void FindAndRemove(std::vector<std::shared_ptr<Node>> &vect, std::shared_ptr<Node> toRemove) const;

    };

//empty constructor (npds 2feb2017)
    Network::Network() {

    }

//constructor - takes in the path to the input qasm file and parses it, generating the fully connected tensor network
    Network::Network(const std::string &inputFile, const std::string &measureFile) : mInputFile(inputFile),
                                                                                     mMeasureFile(measureFile) {
        ParseNetwork(inputFile);
    }

//this function resets all the data in the class and calls the ParseNetwork function again to regenerate the network
    void Network::Reset() {
        mNetworkParsingNodes.clear();
        mNetworkParsingWires.clear();
        mFinalVal = std::complex<double>(0.0);
        mNumberOfQubits = 0;
        mDepth = 0;
        mDone = false;
        mFailure = false;
        mAllNodes.clear();
        mNodesByWire.clear();
        mUncontractedNodes.clear();
        mArbitraryOneQubitGates.clear();
        mArbitraryTwoQubitGates.clear();


        ParseNetwork(mInputFile);
        //clear everything except for mInputFile and call ParseNetwork.
    }


//this function creates initial state nodes and adds them to the network
//by default, the starting state is |0><0|tensored n times, but you can start the network in a random, non normalized state by modifying
//the function - not recommended
    void Network::CreateInitialStates() {

        //Create initial states for every qubit
        for (int i = 0; i < mNumberOfQubits; i++) {
            std::shared_ptr<Node> temNode = std::make_shared<ZeroStateNode>();
            std::cout << "Creating qubit " << i << " in the initial state: |0><0|" << std::endl;
            //create a and attach a wire to each initial state
            std::shared_ptr<Wire> temWire = std::make_shared<Wire>(temNode, nullptr, i);
            temNode->GetWires().push_back(temWire);
            mNetworkParsingWires.push_back(temWire);
            mNetworkParsingNodes.push_back(temNode);
            mAllNodes.push_back(temNode);
            temNode->AddWireNumber(i);
            mNodesByWire[i].push_back(temNode);
            temNode->mID = mAllNodes.size() - 1;
        }

    }


//This function takes in a vector of measurements on each qubit to perform each index in the vector corresponds to each qubit
//to trace out a qubit, the T character is sent. - recognized measurements are X,Y,Z,0,1,T where 0 and 1 are projection ops
    void Network::AddMeasurementsOrTrace(std::vector<char> &measurements) {
        //Create and add either trace out or measurement operators
        for (int i = 0; i < mNumberOfQubits; i++) {
            std::shared_ptr<Node> measureNodeTemp;
            if (measurements.size() <= i) {
                std::cout << "Tracing out qubit: " << i << std::endl;
                measureNodeTemp = std::make_shared<TraceNode>();
            } else if (measurements[i] == 'X') //X measurement
            {
                std::cout << "Creating X measurement on qubit: " << i << std::endl;
                measureNodeTemp = std::make_shared<XMeasure>();
            } else if (measurements[i] == 'Y')//Y measurement
            {
                std::cout << "Creating Y measurement on qubit: " << i << std::endl;
                measureNodeTemp = std::make_shared<YMeasure>();
            } else if (measurements[i] == 'Z') //Z Measurement
            {
                std::cout << "Creating Z measurement on qubit: " << i << std::endl;
                measureNodeTemp = std::make_shared<ZMeasure>();
            } else if (measurements[i] == '0') {
                std::cout << "Creating Projection |0><0| measurement on qubit: " << i << std::endl;
                measureNodeTemp = std::make_shared<ProjectZero>();
            } else if (measurements[i] == '1') {
                std::cout << "Creating Projection |1><1| measurement on qubit: " << i << std::endl;
                measureNodeTemp = std::make_shared<ProjectOne>();
            } else //Trace out/I measurement
            {
                std::cout << "Tracing out qubit: " << i << std::endl;
                measureNodeTemp = std::make_shared<TraceNode>();
            }

            //add the measurement and connect it
            mNetworkParsingWires[i]->SetNodeB(measureNodeTemp);
            measureNodeTemp->GetWires().push_back(mNetworkParsingWires[i]);
            mAllNodes.push_back(measureNodeTemp);
            measureNodeTemp->AddWireNumber(i);
            mNodesByWire[i].push_back(measureNodeTemp);
            measureNodeTemp->mID = mAllNodes.size() - 1;
        }


    }


//This function takes in the qasm input file and parses it, adding nodes to the circuit for the initial states, each gate
//, and then finally measurements - modify the test case file measureTest.txt to modify the measurements or modify the function below, so
//the user has to specify the measurement file when creating the network
    void Network::ParseNetwork(const std::string &inputFile) {
        mAllNodes.reserve(1000000);
        std::ifstream input(inputFile);
        if (!input.is_open()) {
            std::cout << "Failed to open QASM file!" << std::endl;
            mFailure = true;
            throw InvalidFile();
        }

        //get the first line from the qasm file - should be number of qubits and parse it
        std::string numQubString, temp;
        std::getline(input, numQubString);
        mNumberOfQubits = std::stoi(numQubString);

        //reserve space
        mNodesByWire.resize(mNumberOfQubits);
        std::for_each(mNodesByWire.begin(), mNodesByWire.end(), [](std::vector<std::shared_ptr<Node>> temp) {
            temp.reserve(1000);
        });

        //Create and add initial states
        CreateInitialStates();


        std::cout << "Parsing nodes from file...." << std::endl;
        //Parse Gates from file
        while (!input.eof()) {
            std::getline(input, temp);
            ParseNode(temp);
        }
        input.close();


        //Add measurements

        //testing case::
        std::vector<char> measurements(mNumberOfQubits);

        //change this input file path to change where the user specifies the measurements
        std::ifstream measureStream(mMeasureFile);
        if (!measureStream.is_open()) {
            std::cout << "Measurement file failed to open - all qubits will be traced out" << std::endl;
        }

        //parse the measurement file
        for (int i = 0; i < mNumberOfQubits; i++) {
            char t;
            if (measureStream >> t)
                measurements[i] = t;
            else {
                measurements[i] = 'T';
            }
        }
        AddMeasurementsOrTrace(measurements);

        mNetworkParsingNodes.clear();
        mNetworkParsingWires.clear();
        mUncontractedNodes = mAllNodes;
    }


//this function contracts the tensor network in the order of creation of nodes - use this only to test. Otherwise, use the
//functions in the parallelizer wrapper class
    void Network::ContractNetworkLinearly() {
        std::shared_ptr<Node> temp;
        while (!mDone) {
            if(!mUncontractedNodes[0]->GetWires()[0]->GetNodeA().expired() && !mUncontractedNodes[0]->GetWires()[0]->GetNodeB().expired())
            {
                std::shared_ptr<Node> tempA(mUncontractedNodes[0]->GetWires()[0]->GetNodeA().lock());
                std::shared_ptr<Node> tempB(mUncontractedNodes[0]->GetWires()[0]->GetNodeB().lock());
                ContractNodes(tempA, tempB, 1000);
            } else
            {
                throw InvalidContractionMethod();
            }
        }
    }


//this function takes in a line from the qasm file and parses it, creating the appropriate node or arbitrary gate definition
    void Network::ParseNode(std::string &inputLine) {
        // std::cout<<"Attempting To Parse A Node"<<std::endl;
        std::vector<std::string> parsedLine;
        //parse the line into tokens (with a space as the delimiter
        ParseTokens(inputLine, parsedLine);

        if (parsedLine.size() == 0) {
            return;
        }
        std::shared_ptr<Node> newNode;

        if (parsedLine[0] == "Rx") //if the line is an Rx gate
        {
            //Rx 3.1415 0 = Rx(pi) on qubit 0

            //convert the phase to a float
            float tempPhaseVal{std::stof(parsedLine[1])};

            //to debug
            //std::cout<<"Created Rx Node..."<<std::endl;

            //create the Rx Node
            newNode = std::make_shared<RxNode>(tempPhaseVal);

            //convert the qubit index to an int
            int tempQubitValOne{std::stoi(parsedLine[2])};
            if (tempQubitValOne > mNumberOfQubits - 1) {
                throw InvalidFileFormat();
            }

            //add the current hanging wire to the node
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);

            //add the node to the current hanging wire
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);

            //create a new hanging wire
            std::shared_ptr<Wire> newWire = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);

            //set the current hanging wire to the new wire
            mNetworkParsingWires[tempQubitValOne] = newWire;

            //add the hanging wire to the new node
            newNode->GetWires().push_back(newWire);

            //add the wire number to the new node
            newNode->AddWireNumber(tempQubitValOne);

            //set the index of the previous node
            newNode->mIndexOfPreviousNode = mNodesByWire[tempQubitValOne].size() - 1;

            //add the node to mNodes by wire
            mNodesByWire[tempQubitValOne].push_back(newNode);
        } else if (parsedLine[0] == "Ry") //if the line is an Ry gate
        {

            //see Rx gate
            float tempPhaseVal{std::stof(parsedLine[1])};
            //std::cout<<"Created Ry Node..."<<std::endl;
            newNode = std::make_shared<RyNode>(tempPhaseVal);
            int tempQubitValOne{std::stoi(parsedLine[2])};
            if (tempQubitValOne > mNumberOfQubits - 1) {
                throw InvalidFileFormat();
            }
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWire = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWire;
            newNode->GetWires().push_back(newWire);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->mIndexOfPreviousNode = mNodesByWire[tempQubitValOne].size() - 1;
            mNodesByWire[tempQubitValOne].push_back(newNode);
        } else if (parsedLine[0] == "Rz") //if the line is an Rz gate
        {
            //see Rx gate
            float tempPhaseVal{std::stof(parsedLine[1])};
            //std::cout<<"Created Rz Node..."<<std::endl;
            newNode = std::make_shared<RzNode>(tempPhaseVal);
            int tempQubitValOne{std::stoi(parsedLine[2])};
            if (tempQubitValOne > mNumberOfQubits - 1) {
                throw InvalidFileFormat();
            }
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWire = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWire;
            newNode->GetWires().push_back(newWire);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->mIndexOfPreviousNode = mNodesByWire[tempQubitValOne].size() - 1;
            mNodesByWire[tempQubitValOne].push_back(newNode);
        } else if (parsedLine[0] == "H") //if the line is an H gate
        {
            //see Rx gate
            //std::cout<<"Created H Node..."<<std::endl;
            newNode = std::make_shared<HNode>();
            int tempQubitValOne{std::stoi(parsedLine[1])};
            if (tempQubitValOne > mNumberOfQubits - 1) {
                throw InvalidFileFormat();
            }
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWire = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWire;
            newNode->GetWires().push_back(newWire);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->mIndexOfPreviousNode = mNodesByWire[tempQubitValOne].size() - 1;
            mNodesByWire[tempQubitValOne].push_back(newNode);
        } else if (parsedLine[0] == "X") //if the line is an X gate
        {
            //see Rx gate
            //std::cout<<"Created X Node..."<<std::endl;
            newNode = std::make_shared<XNode>();
            int tempQubitValOne{std::stoi(parsedLine[1])};
            if (tempQubitValOne > mNumberOfQubits - 1) {
                throw InvalidFileFormat();
            }
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWire = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWire;
            newNode->GetWires().push_back(newWire);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->mIndexOfPreviousNode = mNodesByWire[tempQubitValOne].size() - 1;
            mNodesByWire[tempQubitValOne].push_back(newNode);
        } else if (parsedLine[0] == "Y") //if the line is a Y gate
        {
            //see Rx gate
            //std::cout<<"Created Y Node..."<<std::endl;
            newNode = std::make_shared<YNode>();
            int tempQubitValOne{std::stoi(parsedLine[1])};
            if (tempQubitValOne > mNumberOfQubits - 1) {
                throw InvalidFileFormat();
            }
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWire = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWire;
            newNode->GetWires().push_back(newWire);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->mIndexOfPreviousNode = mNodesByWire[tempQubitValOne].size() - 1;
            mNodesByWire[tempQubitValOne].push_back(newNode);
        } else if (parsedLine[0] == "Z") //if the line is a Z gate
        {
            //see Rx gate
            //std::cout<<"Created Z Node..."<<std::endl;
            newNode = std::make_shared<ZNode>();
            int tempQubitValOne{std::stoi(parsedLine[1])};
            if (tempQubitValOne > mNumberOfQubits - 1) {
                throw InvalidFileFormat();
            }
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWire = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWire;
            newNode->GetWires().push_back(newWire);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->mIndexOfPreviousNode = mNodesByWire[tempQubitValOne].size() - 1;
            mNodesByWire[tempQubitValOne].push_back(newNode);
        } else if (parsedLine[0] == "CNOT") //if the line is a CNOT gate
        {

            //std::cout<<"Created CNOT Node..."<<std::endl;
            newNode = std::make_shared<CNOTNode>();
            int tempQubitValOne{std::stoi(parsedLine[1])}; //qubit val one is control, and qubit val 2 is target
            int tempQubitValTwo{std::stoi(parsedLine[2])};
            if (tempQubitValOne > mNumberOfQubits - 1 || tempQubitValTwo > mNumberOfQubits - 1 ||
                tempQubitValOne == tempQubitValTwo) {
                throw InvalidFileFormat();
            }
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValTwo]);
            mNetworkParsingWires[tempQubitValTwo]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWireOne = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWireOne;
            newNode->GetWires().push_back(newWireOne);
            std::shared_ptr<Wire> newWireTwo = std::make_shared<Wire>(newNode, nullptr, tempQubitValTwo);
            mNetworkParsingWires[tempQubitValTwo] = newWireTwo;
            newNode->GetWires().push_back(newWireTwo);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->AddWireNumber(tempQubitValTwo);
            mNodesByWire[tempQubitValOne].push_back(newNode);
            mNodesByWire[tempQubitValTwo].push_back(newNode);
        } else if (parsedLine[0] == "SWAP") //if the line is a SWAP gate
        {
            //std::cout<<"Created SWAP Node..."<<std::endl;
            newNode = std::make_shared<SwapNode>();
            int tempQubitValOne{std::stoi(parsedLine[1])};
            int tempQubitValTwo{std::stoi(parsedLine[2])};
            if (tempQubitValOne > mNumberOfQubits - 1 || tempQubitValTwo > mNumberOfQubits - 1 ||
                tempQubitValOne == tempQubitValTwo) {
                throw InvalidFileFormat();
            }
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValTwo]);
            mNetworkParsingWires[tempQubitValTwo]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWireOne = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWireOne;
            newNode->GetWires().push_back(newWireOne);
            std::shared_ptr<Wire> newWireTwo = std::make_shared<Wire>(newNode, nullptr, tempQubitValTwo);
            mNetworkParsingWires[tempQubitValTwo] = newWireTwo;
            newNode->GetWires().push_back(newWireTwo);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->AddWireNumber(tempQubitValTwo);
            mNodesByWire[tempQubitValOne].push_back(newNode);
            mNodesByWire[tempQubitValTwo].push_back(newNode);
        } else if (parsedLine[0] == "CRk") //if the line is a controlled Rk gate
        {
            //std::cout<<"Created CRk Node..."<<std::endl;
            int tempQubitValOne{std::stoi(parsedLine[1])}; //qubit val one is control, and qubit val 2 is target
            int tempQubitValTwo{std::stoi(parsedLine[2])};
            if (tempQubitValOne > mNumberOfQubits - 1 || tempQubitValTwo > mNumberOfQubits - 1 ||
                tempQubitValOne == tempQubitValTwo) {
                throw InvalidFileFormat();
            }
            newNode = std::make_shared<CRkNode>(tempQubitValOne);
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValTwo]);
            mNetworkParsingWires[tempQubitValTwo]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWireOne = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWireOne;
            newNode->GetWires().push_back(newWireOne);
            std::shared_ptr<Wire> newWireTwo = std::make_shared<Wire>(newNode, nullptr, tempQubitValTwo);
            mNetworkParsingWires[tempQubitValTwo] = newWireTwo;
            newNode->GetWires().push_back(newWireTwo);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->AddWireNumber(tempQubitValTwo);
            mNodesByWire[tempQubitValOne].push_back(newNode);
            mNodesByWire[tempQubitValTwo].push_back(newNode);
        } else if (parsedLine[0] == "CZ") //if the line is a controlled Z gate
        {
            //std::cout<<"Created CZ Node..."<<std::endl;
            int tempQubitValOne{std::stoi(parsedLine[1])}; //qubit val one is control, and qubit val 2 is target
            int tempQubitValTwo{std::stoi(parsedLine[2])};
            if (tempQubitValOne > mNumberOfQubits - 1 || tempQubitValTwo > mNumberOfQubits - 1 ||
                tempQubitValOne == tempQubitValTwo) {
                throw InvalidFileFormat();
            }
            newNode = std::make_shared<CZNode>();
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValTwo]);
            mNetworkParsingWires[tempQubitValTwo]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWireOne = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWireOne;
            newNode->GetWires().push_back(newWireOne);
            std::shared_ptr<Wire> newWireTwo = std::make_shared<Wire>(newNode, nullptr, tempQubitValTwo);
            mNetworkParsingWires[tempQubitValTwo] = newWireTwo;
            newNode->GetWires().push_back(newWireTwo);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->AddWireNumber(tempQubitValTwo);
            mNodesByWire[tempQubitValOne].push_back(newNode);
            mNodesByWire[tempQubitValTwo].push_back(newNode);
        } else if (parsedLine[0] == "CPHASE") //if the line is a controlled phase gate
        {
            //std::cout<<"Created CPHASE Node..."<<std::endl;
            double tempPhaseVal{std::stod(parsedLine[1])};
            int tempQubitValOne{std::stoi(parsedLine[2])}; //qubit val one is control, and qubit val 2 is target
            int tempQubitValTwo{std::stoi(parsedLine[3])};
            if (tempQubitValOne > mNumberOfQubits - 1 || tempQubitValTwo > mNumberOfQubits - 1 ||
                tempQubitValOne == tempQubitValTwo) {
                throw InvalidFileFormat();
            }
            newNode = std::make_shared<CPhaseNode>(tempPhaseVal);
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValTwo]);
            mNetworkParsingWires[tempQubitValTwo]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWireOne = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWireOne;
            newNode->GetWires().push_back(newWireOne);
            std::shared_ptr<Wire> newWireTwo = std::make_shared<Wire>(newNode, nullptr, tempQubitValTwo);
            mNetworkParsingWires[tempQubitValTwo] = newWireTwo;
            newNode->GetWires().push_back(newWireTwo);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->AddWireNumber(tempQubitValTwo);
            mNodesByWire[tempQubitValOne].push_back(newNode);
            mNodesByWire[tempQubitValTwo].push_back(newNode);
        } else if (parsedLine[0] == "def1") //if the line defines an arbitrary one qubit gate
        {
            //add the name of the gate and the file path to the matrix values to a map
            mArbitraryOneQubitGates.insert({parsedLine[1], parsedLine[2]});
            return;
        } else if (parsedLine[0] == "def2") //if the line defines an arbitrary two qubit gate
        {
            //add the name of the gate and the file path to the matrix values to a map
            mArbitraryTwoQubitGates.insert({parsedLine[1], parsedLine[2]});
            return;
        } else if (mArbitraryOneQubitGates.find(parsedLine[0]) !=
                   mArbitraryOneQubitGates.end()) //if the line is an arbitrary one qubit gate
        {
            newNode = std::make_shared<ArbitraryOneQubitNode>(mArbitraryOneQubitGates[parsedLine[0]], parsedLine[0]);
            //std::cout<<"Created "<<parsedLine[0]<<" Node..."<<std::endl;
            int tempQubitValOne{std::stoi(parsedLine[1])};
            if (tempQubitValOne > mNumberOfQubits - 1) {
                throw InvalidFileFormat();
            }
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWire = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWire;
            newNode->GetWires().push_back(newWire);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->mIndexOfPreviousNode = mNodesByWire[tempQubitValOne].size() - 1;
            mNodesByWire[tempQubitValOne].push_back(newNode);
        } else if (mArbitraryTwoQubitGates.find(parsedLine[0]) !=
                   mArbitraryTwoQubitGates.end()) //if the line is an arbitrary two qubit gate
        {
            // std::cout<<"Created "<<parsedLine[0]<<" Node..."<<std::endl;
            newNode = std::make_shared<ArbitraryTwoQubitNode>(mArbitraryTwoQubitGates[parsedLine[0]], parsedLine[0]);
            int tempQubitValOne{std::stoi(parsedLine[1])};
            int tempQubitValTwo{std::stoi(parsedLine[2])};
            if (tempQubitValOne > mNumberOfQubits - 1 || tempQubitValTwo > mNumberOfQubits - 1 ||
                tempQubitValOne == tempQubitValTwo) {
                throw InvalidFileFormat();
            }
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValOne]);
            mNetworkParsingWires[tempQubitValOne]->SetNodeB(newNode);
            newNode->GetWires().push_back(mNetworkParsingWires[tempQubitValTwo]);
            mNetworkParsingWires[tempQubitValTwo]->SetNodeB(newNode);
            std::shared_ptr<Wire> newWireOne = std::make_shared<Wire>(newNode, nullptr, tempQubitValOne);
            mNetworkParsingWires[tempQubitValOne] = newWireOne;
            newNode->GetWires().push_back(newWireOne);
            std::shared_ptr<Wire> newWireTwo = std::make_shared<Wire>(newNode, nullptr, tempQubitValTwo);
            mNetworkParsingWires[tempQubitValTwo] = newWireTwo;
            newNode->GetWires().push_back(newWireTwo);

            newNode->AddWireNumber(tempQubitValOne);
            newNode->AddWireNumber(tempQubitValTwo);
            mNodesByWire[tempQubitValOne].push_back(newNode);
            mNodesByWire[tempQubitValTwo].push_back(newNode);
        }
            // add more node functionality here --> kraus op gate,
        else //if the does not define any recognized command
        {
            std::cout << "Failed to compile line: " << std::endl;
            for (const auto &t: parsedLine) {
                std::cout << t << " ";
            }
            std::cout << std::endl;
            throw InvalidFileFormat();
        }

        //add the node to mAllNodes and give it an ID number
        mAllNodes.push_back(newNode);
        newNode->mID = mAllNodes.size() - 1;
    }


//this function takes in pointers to two nodes and a threshold value and contracts them under certain conditions:
//1. the two nodes are connected
//2. the resulting contracted node has a rank greater than the max rank of the two nodes plus the threshold value

//the function returns a pointer to the resulting contracted node -> nullptr if contraction fails or the conditions were not met
    std::shared_ptr<Node>
    Network::ContractNodes(std::shared_ptr<Node> nodeA, std::shared_ptr<Node> nodeB, int threshold) {

        mLocker.lock();
        if(nodeA->mContracted || nodeB->mContracted)
        {
            mLocker.unlock();
            return nullptr;
        }
        std::vector<int> indicesA;//vector to store the indices in node A on which to contract
        // indicesA.reserve(nodeA->mRank);
        std::vector<int> indicesB;//vector to store the indices in node B on which to contract
        // indicesB.reserve(nodeB->mRank);
        std::vector<std::pair<bool, int>> indicesC; //vector to store the indices from nodes A and B which will not be contracted -
        // a true value in the pair means the index comes from Node A, and a false value means from node B
        indicesC.reserve(nodeA->mRank + nodeB->mRank);
        //vectors to store the wires which will be contracted on, and the wires which remain uncontracted (from both nodes)
        std::vector<std::shared_ptr<Wire>> connectedWires;
        connectedWires.reserve(nodeA->mRank + nodeB->mRank);
        std::vector<std::shared_ptr<Wire>> remainingWires;
        remainingWires.reserve(nodeA->mRank + nodeB->mRank);

        //this loop iterates through the wires attached to node A and determines if they are also attached to node B
        int i(0);
        for (auto &temp: nodeA->GetWires()) {
            if (temp->GetNodeB().lock() == nodeB || temp->GetNodeA().lock() == nodeB) {
                indicesA.push_back(i);
                connectedWires.push_back(temp);
            } else {
                int t(i);
                indicesC.push_back(std::make_pair<bool, int>(true, std::move(t)));
                remainingWires.push_back(temp);
            }
            ++i;
        }

        //this loop determines which index in B the connected wire is attached to
        for (int j = 0; j < connectedWires.size(); j++) {
            for (int k = 0; k < nodeB->GetWires().size(); k++) {
                if (connectedWires[j] == nodeB->GetWires()[k]) {
                    indicesB.push_back(k);
                }
            }
        }

        //this loop determines the unconnected wires in Node B
        int j = 0;
        for (auto &temp: nodeB->GetWires()) {
            if (!(temp->GetNodeB().lock() == nodeA || temp->GetNodeA().lock() == nodeA)) {
                int t(j);
                indicesC.push_back(std::make_pair<bool, int>(false, std::move(t)));
                remainingWires.push_back(temp);
            }
            ++j;
        }

        //if there are no connected wires or if the rank of resulting node > max(rankA, rankB) + threshold, return nullptr
        if (indicesA.size() == 0 || remainingWires.size() > std::max(nodeA->mRank, nodeB->mRank) + threshold ||
            nodeA->mContracted || nodeB->mContracted) {
            //if no nodes are shared
            mLocker.unlock();
            return nullptr;
        }


        //uncomment to debug
/*
    std::cout<<"FOR DEBUGGING PURPOSES: "<<std::endl;
    std::cout<<"Indices in Node A that matched:"<<std::endl;
    for(auto& t: indicesA)
        std::cout<<t<<", ";
    std::cout<<std::endl<<"Indices in Node B that matched:"<<std::endl;
    for(auto& t: indicesB)
        std::cout<<t<<", ";
    std::cout<<std::endl;
*/

        //create a vector that corresponds to the pairs of connected indices
        std::vector<std::pair<int, int>> indexPairs(indicesA.size());
        int k = 0;
        //populate it
        std::generate(indexPairs.begin(), indexPairs.end(), [&indicesA, &indicesB, &k]() {
            std::pair<int, int> temp(std::move(indicesA[k]), std::move(indicesB[k]));
            ++k;
            return temp;
        });

        //update the wires to point to nodeC instead of A and B
        //Uncomment to debug
        //std::cout<<"rank of A: "<<nodeA->mRank<<std::endl;
        //std::cout<<"rank of B: "<<nodeB->mRank<<std::endl;
        //std::cout<<"New rank of C: "<<remainingWires.size()<<std::endl;

        //create node C, and update the remaining wires to point to nodeC instead of A and B

        std::shared_ptr<Node> nodeC = std::make_shared<Node>(indicesC.size());
        std::for_each(remainingWires.begin(), remainingWires.end(),
                      [nodeC, nodeA, nodeB](std::shared_ptr<Wire> tempWire) {
                          nodeC->GetWires().push_back(tempWire);
                          if (tempWire->GetNodeA().lock() == nodeA || tempWire->GetNodeA().lock() == nodeB) {
                              tempWire->SetNodeA(nodeC);
                          } else if (tempWire->GetNodeB().lock() == nodeA || tempWire->GetNodeB().lock() == nodeB) {
                              tempWire->SetNodeB(nodeC);
                          }
                      });

        //create vectors to hold the current index in nodes A, B, and C
        std::vector<int> aIndexHolder(nodeA->mRank);
        std::vector<int> bIndexHolder(nodeB->mRank);
        // std::vector<int> cIndexHolder(nodeC->mRank);

        //create a complex to hold the value for each summation
        //  std::complex<double> startingSum(0.0);

        //warn the user if contracting a large tensor
        if (nodeC->mRank >= THRESH_RANK_THREAD) {
            std::cout << "Contracting Nodes of Rank " << nodeA->mRank << " and " << nodeB->mRank
                      << " to get a Node of Rank: " << nodeC->mRank << " Hold On....." << std::endl;
        }
        //
        nodeB->mContracted = true;
        nodeA->mContracted = true;
        for (int i = 0; i < connectedWires.size(); i++) {
            connectedWires[i]->SetIsContracted(true);
        }
        mLocker.unlock();
        ContractIndices(indicesC, indexPairs, aIndexHolder, bIndexHolder, nodeA, nodeB, nodeC);

        // Label the wires as having been contracted


        // std::cout<<nodeA->mRank<<", "<<nodeB->mRank<<", "<<nodeC->mRank<<", "<<t1.getElapsed()<<std::endl;
        //set the ID, created from, and remove the node from uncontracted nodes
        if (mDone) {
            std::shared_ptr<Node> t = std::make_shared<Node>(0);
            t->mID = mAllNodes.size();
            t->mCreatedFrom.first = nodeA->mID;
            t->mCreatedFrom.second = nodeB->mID;
            mAllNodes.push_back(t);
            FindAndRemove(mUncontractedNodes, nodeA);
            FindAndReplace(mUncontractedNodes, nodeB, nodeC);
            nodeA->ClearNodeData();
            nodeB->ClearNodeData();
            return nullptr;
        }
        mLocker.lock();
        nodeC->mID = mAllNodes.size();
        nodeC->mCreatedFrom.first = nodeA->mID;
        nodeC->mCreatedFrom.second = nodeB->mID;
        mAllNodes.push_back(nodeC);
        nodeA->ClearNodeData();
        nodeB->ClearNodeData();
        FindAndRemove(mUncontractedNodes, nodeA);
        FindAndReplace(mUncontractedNodes, nodeB, nodeC);
        mLocker.unlock();
        return nodeC;
    }




//This function takes in a list of indices that will be the indices of the resultant node (toNotSumOn) - false value means it came from
/*Node A, and a true value means it came from node B. The function also takes in a vector of pairs, which corresponds to the indices in A and B
 * On which the contraction should be performed. Finally, the function takes in vectors to hold the current index of contraction
 * for nodes A and B, and pointers to all three nodes
 * The function updates the values in nodeC according to the standard way of contracting tensors A and B
 */

    inline void Network::ContractIndices(const std::vector<std::pair<bool, int>> &toNotSumOn,
                                         const std::vector<std::pair<int, int>> &toSumOn,
                                         std::vector<int> &vectorIndexA,
                                         std::vector<int> &vectorIndexB,
                                         std::shared_ptr<Node> nodeA, std::shared_ptr<Node> nodeB,
                                         std::shared_ptr<Node> nodeC) {

        // Update number of floating point ops
        int numIndepInd = toNotSumOn.size() + toSumOn.size();
        this->mNumFloatOps += pow(4, numIndepInd);


        //if rank of C is > 9, parallelize outer loop
        //if rank of A or B is > 9, parallelize inner loop
        int toContractOn((vectorIndexA.size() + vectorIndexB.size() - toNotSumOn.size()) / 2);

        auto f1 = [&toNotSumOn, &toSumOn, &toContractOn](
                int indexASize, int indexBSize,
                std::shared_ptr<Node> nodeA, std::shared_ptr<Node> nodeB, std::shared_ptr<Node> nodeC,
                unsigned long long maxCount,
                unsigned long long minCount) {
            std::vector<int> vectorIndexA(indexASize);
            std::vector<int> vectorIndexB(indexBSize);
            for (unsigned long long Ccounter(minCount); Ccounter < maxCount && totTimer.getElapsed() <
                                                                               maxTime; Ccounter++) //outer loop loops through all indices in C
            {
                for (int i = 0; i < toNotSumOn.size(); i++) {
                    if (toNotSumOn[i].first) {
                        vectorIndexA[toNotSumOn[i].second] = (Ccounter >> (2 * i)) % 4; //update the index in A
                    } else {
                        vectorIndexB[toNotSumOn[i].second] = (Ccounter >> (2 * i)) % 4; //update the index in B
                    }
                }
                for (unsigned long long ABcounter(0); ABcounter < pow(4, toSumOn.size()) && totTimer.getElapsed() <
                                                                                            maxTime; ABcounter++)//inner loop loops through all indices being contracted on
                {
                    for (int i = 0; i < toContractOn; i++) {
                        //update indices in A and B
                        vectorIndexA[toSumOn[toContractOn - 1 - i].first] = (ABcounter >> (2 * i)) % 4;
                        vectorIndexB[toSumOn[toContractOn - 1 - i].second] = (ABcounter >> (2 * i)) % 4;
                    }

                    //convert the vectors of indices in A and B to a single number
                    unsigned long long sum(0);
                    unsigned long long multiplier(0);
                    std::for_each(vectorIndexA.begin(), vectorIndexA.end(), [&sum, &multiplier](int t) {
                        sum += t << (2 * multiplier);
                        ++multiplier;
                    });
                    unsigned long long sum2(0);
                    unsigned long long multiplier2(0);
                    std::for_each(vectorIndexB.begin(), vectorIndexB.end(), [&sum2, &multiplier2](int t) {
                        sum2 += t << (2 * multiplier2);
                        ++multiplier2;
                    });
                    nodeC->Index(Ccounter) += nodeA->Access(sum) * nodeB->Access(sum2);
                    //perform the math
                }
            }
        };

//actually do the threading: using the lambda above
        if (nodeA->GetTensorVals().size() == 0 || nodeB->GetTensorVals().size() == 0) {
            throw InvalidFunctionInput();
        }
        if (nodeC->mRank >= THRESH_RANK_THREAD) {
            std::vector<std::thread> threads(mNumberOfThreads - 1);
            for (int i = 0; i < mNumberOfThreads; i++) {
                if (i != (mNumberOfThreads - 1)) {
                    threads[i] = std::thread(f1, vectorIndexA.size(), vectorIndexB.size(), nodeA, nodeB, nodeC,
                                             (i + 1) *
                                             static_cast<long long>(pow(4, toNotSumOn.size()) / mNumberOfThreads),
                                             i * static_cast<long long>(pow(4, toNotSumOn.size()) / mNumberOfThreads));
                } else {
                    f1(vectorIndexA.size(), vectorIndexB.size(), nodeA, nodeB, nodeC,
                       static_cast<long long>(pow(4, toNotSumOn.size())),
                       i * static_cast<long long>(pow(4, toNotSumOn.size()) / mNumberOfThreads));
                }
            }
            for (auto &tem: threads) {
                tem.join();
            }
        } else {
            f1(vectorIndexA.size(), vectorIndexB.size(), nodeA, nodeB, nodeC, pow(4, toNotSumOn.size()), 0);
        }
        if (toNotSumOn.size() == 0) {
            //if you're contracting two nodes to get a rank 0 tensor
            mDone = true;
            mFinalVal = nodeC->Access({0});
        }

    }


//this function takes in a string input line and populates an output vector with all of the tokens in the line
//the function avoids all lines that begin with '#' and uses a space as its delimiter
    void Network::ParseTokens(std::string &input, std::vector<std::string> &output) {
// passing -1 as the submatch index parameter performs splitting
        if (input.length() == 0) {
            return;
        }
        int tempIndex = static_cast<int>(input.find("#"));
        if (tempIndex != std::string::npos) {
            input.substr(0, (input.find("#")));
        }
        if (tempIndex == 0) {
            return;
        }
        std::string regex{" "};
        std::regex re{regex};
        std::sregex_token_iterator first{input.begin(), input.end(), re, -1}, last;
        std::copy(first, last, std::back_inserter(output));
    }

//This function takes the vectors of AllNodes and UncontractedNodes and moves the rank 1 initial state nodes to the back of
//the vectors
    void Network::MoveInitialStatesToBack() {
        int count = 0;
        for (int i = mAllNodes.size() - 1 - mNumberOfQubits; i >= mAllNodes.size() - 1 - (2 * mNumberOfQubits); i--) {
            std::swap(mAllNodes[count], mAllNodes[i]);
            count++;
        }

        count = 0;
        for (int i = mUncontractedNodes.size() - 1 - mNumberOfQubits;
             i >= mUncontractedNodes.size() - 1 - (2 * mNumberOfQubits); i--) {
            std::swap(mUncontractedNodes[count], mUncontractedNodes[i]);
            count++;
        }
    }


/*This function takes the tensor network and reduces it, contracting any one qubit gates or two qubit gates in succession
 * it updates mNodesByWire with the resulting circuit. Note that mAllNodes and mUncontractedNodes are also updated
 */
    void Network::ReduceCircuit() {
        std::vector<std::vector<std::shared_ptr<Node>>> placeHolder(mNumberOfQubits);
        //remember to remove the nodes you contract from mAllNodes
        //first step is contract all of the rank 2 tensors... -> delete one contracted (nullptr) - replace the other
        int i = 0;
        for (auto &tempWireVect: mNodesByWire) {
            for (auto &tempRankTwoNode: tempWireVect) {
                if (tempRankTwoNode->mRank == 2) {
                    std::shared_ptr<Node> temp = ContractNodes(placeHolder[i].back(), tempRankTwoNode, 0);
                    temp->AddWireNumber(tempRankTwoNode->GetWires()[0]->GetNodeA().lock()->GetWireNumber()[0]);
                    if (tempRankTwoNode->GetWires()[0]->GetNodeA().lock()->mRank > 2) {
                        temp->AddWireNumber(tempRankTwoNode->GetWires()[0]->GetNodeA().lock()->GetWireNumber()[1]);
                    }
                    //update all nodes that should be contracted -> replace the previous nodes
                    std::shared_ptr<Node> toFind = placeHolder[i].back();
                    if (toFind->GetTypeOfNode() == GateType::INITSTATE) {
                        temp->SetTypeOfNode(GateType::INITSTATE);
                        temp->SetTypeOfNodeString("INITSTATE(Manipulated)");
                    }
                    FindAndReplace(mNodesByWire, toFind, temp);
                    FindAndReplace(placeHolder, toFind, temp);
                } else {
                    placeHolder[i].push_back(tempRankTwoNode);
                }
            }
            i++;
        }
        mNodesByWire = std::move(placeHolder);




        //second step is to contract all the successive rank 4 tensors:
        std::vector<std::vector<std::shared_ptr<Node>>> temporaryNodesByWire(mNumberOfQubits);
        std::vector<std::shared_ptr<Node>> tempCurrentNode(mNumberOfQubits);
        std::vector<int> currentIndex(mNumberOfQubits);
        std::vector<bool> updated(mNumberOfQubits);
        std::fill(updated.begin(), updated.end(), false);
        std::fill(currentIndex.begin(), currentIndex.end(), 0);
        bool flag{false};
        bool found{false};
        while (!flag) {
            for (int j = 0; j < mNumberOfQubits; j++) {
                if (currentIndex[j] >= mNodesByWire[j].size() || updated[j]) {
                    continue;
                }

                std::shared_ptr<Node> temporary(mNodesByWire[j][currentIndex[j]]);
                if (temporary->mRank == 1) {
                    tempCurrentNode[j] = mNodesByWire[j][currentIndex[j]];
                    updated[j] = true;
                    currentIndex[j]++;
                    continue;
                } else if (!updated[temporary->GetWireNumber()[1]] && !updated[temporary->GetWireNumber()[0]]) {
                    if (mNodesByWire[temporary->GetWireNumber()[0]][currentIndex[temporary->GetWireNumber()[0]]] ==
                        mNodesByWire[temporary->GetWireNumber()[1]][currentIndex[temporary->GetWireNumber()[1]]]) {
                        if (tempCurrentNode[temporary->GetWireNumber()[0]] ==
                            tempCurrentNode[temporary->GetWireNumber()[1]]) {
                            tempCurrentNode[temporary->GetWireNumber()[0]] = ContractNodes(temporary,
                                                                                           tempCurrentNode[temporary->GetWireNumber()[0]],
                                                                                           0);
                            tempCurrentNode[temporary->GetWireNumber()[1]] = tempCurrentNode[temporary->GetWireNumber()[0]];
                            tempCurrentNode[temporary->GetWireNumber()[0]]->AddWireNumber(
                                    temporary->GetWireNumber()[0]);
                            tempCurrentNode[temporary->GetWireNumber()[0]]->AddWireNumber(
                                    temporary->GetWireNumber()[1]);
                            found = true;
                        } else {
                            tempCurrentNode[temporary->GetWireNumber()[0]] = temporary;
                            tempCurrentNode[temporary->GetWireNumber()[1]] = temporary;
                        }
                        currentIndex[temporary->GetWireNumber()[0]]++;
                        currentIndex[temporary->GetWireNumber()[1]]++;
                        updated[temporary->GetWireNumber()[0]] = true;
                        updated[temporary->GetWireNumber()[1]] = true;
                    }
                }


            }
            for (int j = 0; j < updated.size(); j++) {
                if (updated[j]) {
                    if (found) {
                        temporaryNodesByWire[j].back() = tempCurrentNode[j];
                    } else {
                        temporaryNodesByWire[j].push_back(tempCurrentNode[j]);
                    }
                } else if (currentIndex[j] < mNodesByWire[j].size() && !found) {
                    temporaryNodesByWire[j].push_back(nullptr);
                }
            }
            if (std::all_of(updated.begin(), updated.end(), [](bool i) { return !i; })) {
                flag = true;
            }

            std::fill(updated.begin(), updated.end(), false);
            found = false;
            //add current nodes to temporary wires by node
        }

        //reassign
        mNodesByWire = std::move(temporaryNodesByWire);

    }


/*This function is mainly unused, but it localizes the interactions between all qubits in the circuit using swap gates
 * It updates the depth of the circuit
 */
    void Network::LocalizeInteractions(const std::string &logFile) {
        std::vector<std::shared_ptr<Node>> temporaryListOfNodes;
        temporaryListOfNodes.reserve(mAllNodes.size() * mNumberOfQubits);
        std::vector<std::vector<std::shared_ptr<Node>>> tempNodesByWire(mNumberOfQubits);
        std::vector<std::vector<std::shared_ptr<Wire>>> tempWiresListing(mNumberOfQubits);
        std::vector<std::tuple<int, int, int, int>> twoQubGatesInSequence; //first int is the first wire number, 2nd is the 2nd wire #, last two are the two indices of the nodes on the first wire
        std::vector<int> gatesInSeqCount(mNumberOfQubits);
        std::fill(gatesInSeqCount.begin(), gatesInSeqCount.end(), 0);
        tempNodesByWire.resize(mNumberOfQubits);
        tempWiresListing.resize(mNumberOfQubits);
        std::for_each(tempNodesByWire.begin(), tempNodesByWire.end(), [](std::vector<std::shared_ptr<Node>> t) {
            t.reserve(500);
        });
        int i = 0;
        for (auto &node: mUncontractedNodes) {
            node->GetWires().clear();
            if (node->mRank == 4) {
                //HERE is where the magic happens
                int t = node->GetWireNumber()[0];
                while (abs(node->GetWireNumber()[1] - t) != 1) {
                    int u;
                    bool flag{false};
                    if (t > node->GetWireNumber()[1]) {
                        u = t - 1;
                    } else {
                        flag = true;
                        u = t + 1;
                    }
                    std::shared_ptr<Node> swapGate = std::make_shared<SwapNode>();
                    swapGate->AddWireNumber(t);
                    swapGate->AddWireNumber(u);
                    //add appropriate wire connections here..... otherwise, swapgate will be unconnected
                    //note that no new wires need to be created->only reassigned
                    std::shared_ptr<Wire> newWireOne = std::make_shared<Wire>(swapGate, nullptr, t);
                    std::shared_ptr<Wire> newWireTwo = std::make_shared<Wire>(swapGate, nullptr, u);
                    swapGate->GetWires().push_back(tempWiresListing[t].back());
                    swapGate->GetWires().push_back(tempWiresListing[u].back());
                    swapGate->GetWires().push_back(newWireOne);
                    swapGate->GetWires().push_back(newWireTwo);




                    //check to see if there is a previous swap gate before
                    if (!tempNodesByWire[t].empty() &&
                        !tempNodesByWire[u].empty()) //checks to see if the two gates act in sequence
                    {
                        if (tempNodesByWire[t].back() == tempNodesByWire[u].back() &&
                            tempNodesByWire[t].back()->GetTypeOfNode() == GateType::SWAP) {
                            tempNodesByWire[t].pop_back();
                            tempNodesByWire[u].pop_back();
                            tempWiresListing[t].pop_back();
                            tempWiresListing[u].pop_back();

                        } else if (tempNodesByWire[t].back() == tempNodesByWire[u].back()) {
                            twoQubGatesInSequence.push_back(
                                    std::make_tuple(t, u, tempNodesByWire[t].size() - 1, tempNodesByWire[t].size()));
                            gatesInSeqCount[u]++;
                            gatesInSeqCount[t]++;
                            tempWiresListing[u].back()->SetNodeB(swapGate);
                            tempWiresListing[t].back()->SetNodeB(swapGate);
                            tempWiresListing[u].push_back(newWireOne);
                            tempWiresListing[t].push_back(newWireTwo);
                            tempNodesByWire[t].push_back(swapGate);
                            tempNodesByWire[u].push_back(swapGate);
                        } else {
                            tempWiresListing[u].back()->SetNodeB(swapGate);
                            tempWiresListing[t].back()->SetNodeB(swapGate);
                            tempWiresListing[u].push_back(newWireOne);
                            tempWiresListing[t].push_back(newWireTwo);
                            tempNodesByWire[t].push_back(swapGate);
                            tempNodesByWire[u].push_back(swapGate);
                        }

                    }
                    temporaryListOfNodes.push_back(swapGate);
                    flag ? t++ : t--;
                }


                //add the node
                if (tempNodesByWire[node->GetWireNumber()[1]].back() == tempNodesByWire[t].back()) {
                    gatesInSeqCount[t]++;
                    gatesInSeqCount[node->GetWireNumber()[1]]++;
                    twoQubGatesInSequence.push_back(
                            std::make_tuple(t, node->GetWireNumber()[1], tempNodesByWire[t].size() - 1,
                                            tempNodesByWire.size()));
                }
                node->GetWires().clear();
                int newGateNum(t);
                std::shared_ptr<Wire> newWireOne = std::make_shared<Wire>(node, nullptr, node->GetWireNumber()[1]);
                std::shared_ptr<Wire> newWireTwo = std::make_shared<Wire>(node, nullptr, t);
                tempWiresListing[node->GetWireNumber()[1]].back()->SetNodeB(node);
                tempWiresListing[t].back()->SetNodeB(node);
                node->GetWires().push_back(tempWiresListing[t].back());
                node->GetWires().push_back(tempWiresListing[node->GetWireNumber()[1]].back());
                node->GetWires().push_back(newWireTwo);
                node->GetWires().push_back(newWireOne);
                tempWiresListing[t].push_back(newWireTwo);
                tempWiresListing[node->GetWireNumber()[1]].push_back(newWireOne);
                tempNodesByWire[t].push_back(node);
                tempNodesByWire[node->GetWireNumber()[1]].push_back(node);
                temporaryListOfNodes.push_back(node);
                //swap back

                while (t != node->GetWireNumber()[0]) {

                    int u;
                    bool flag{false};
                    if (t > node->GetWireNumber()[0]) {
                        u = t - 1;
                    } else {
                        flag = true;
                        u = t + 1;
                    }
                    std::shared_ptr<Node> swapGate = std::make_shared<SwapNode>();
                    swapGate->AddWireNumber(t);
                    swapGate->AddWireNumber(u);
                    //add appropriate wire connections here..... otherwise, swapgate will be unconnected
                    //note that no new wires need to be created->only reassigned
                    std::shared_ptr<Wire> newWireOne = std::make_shared<Wire>(swapGate, nullptr, t);
                    std::shared_ptr<Wire> newWireTwo = std::make_shared<Wire>(swapGate, nullptr, u);
                    tempWiresListing[u].back()->SetNodeB(swapGate);
                    tempWiresListing[t].back()->SetNodeB(swapGate);
                    swapGate->GetWires().push_back(tempWiresListing[t].back());
                    swapGate->GetWires().push_back(tempWiresListing[u].back());
                    swapGate->GetWires().push_back(newWireOne);
                    swapGate->GetWires().push_back(newWireTwo);




                    //check to see if there is a previous swap gate before
                    if (tempNodesByWire[t].back() == tempNodesByWire[u].back()) {
                        twoQubGatesInSequence.push_back(
                                std::make_tuple(t, u, tempNodesByWire[t].size() - 1, tempNodesByWire[t].size()));
                        gatesInSeqCount[u]++;
                        gatesInSeqCount[t]++;
                    }
                    tempWiresListing[u].push_back(newWireOne);
                    tempWiresListing[t].push_back(newWireTwo);
                    tempNodesByWire[t].push_back(swapGate);
                    tempNodesByWire[u].push_back(swapGate);
                    temporaryListOfNodes.push_back(swapGate);
                    flag ? t++ : t--;
                }
                node->SetWireNumber(0, newGateNum);

            } else {
                node->GetWires().clear();
                if (node->GetTypeOfNode() == GateType::INITSTATE) {
                    std::shared_ptr<Wire> newWire = std::make_shared<Wire>(node, nullptr, node->GetWireNumber()[0]);
                    tempWiresListing[node->GetWireNumber()[0]].push_back(newWire);
                    node->GetWires().push_back(newWire);

                } else if (node->GetTypeOfNode() == GateType::MEASURETRACE) {
                    node->GetWires().push_back(tempWiresListing[node->GetWireNumber()[0]].back());
                    tempWiresListing[node->GetWireNumber()[0]].back()->SetNodeB(node);
                }
                tempNodesByWire[node->GetWireNumber()[0]].push_back(node);
                temporaryListOfNodes.push_back(node);
            }
        }


        //subtract the number of nodes that were repeated from
        int maxD = 0;
        int k = 0;
        for (auto &temp: tempNodesByWire) {
            if (temp.size() - gatesInSeqCount[k] - 2 > maxD) {
                maxD = temp.size() - gatesInSeqCount[k] - 2;
            }
            k++;
        }

        mDepth = maxD;
        mUncontractedNodes = std::move(temporaryListOfNodes);
        mNodesByWire = std::move(tempNodesByWire);


        //for debugging
        OutputCircuit(mUncontractedNodes, logFile);
    }

//This outputs a vector of Nodes to a file, along with the depth of the circuit
//Also pretty much unused
    void Network::OutputCircuit(const std::vector<std::shared_ptr<Node>> &toOutput, const std::string &logFile) const {
        std::ofstream output(logFile);
        if (!output.is_open()) {
            std::cout << "Failure to Output Circuit to File" << std::endl;
            throw InvalidFile();
        }
        std::for_each(toOutput.begin(), toOutput.end(), [&output](std::shared_ptr<Node> out) {
            output << out->GetTypeOfNodeString() << " ";
            std::for_each(out->GetWireNumber().begin(), out->GetWireNumber().end(), [&output](int i) {
                output << i << " ";
            });
            output << std::endl;
        });
        if (mDepth != 0) {
            output << "Depth: " << mDepth << std::endl;
        } else {
            output << "Depth Has Not Been Calculated due to Non-Local Interactions" << std::endl;
        }
        output.close();

    }


/*This function takes in the string path of a file to output to, and outputs the circuit held in the network to a .dgf file
 * the file's contents can then be copied and pasted into webgraphviz or graphviz, so you can view what the circuit looks like
 */
    void Network::OutputCircuitToVisualGraph(const std::string &toOutputTo) const {
        std::unordered_map<std::shared_ptr<Node>, int> nodeNum(mUncontractedNodes.size());
        std::ofstream output(toOutputTo);
        if (!output.is_open()) {
            std::cout << "Failure to Output Circuit to Visual Graph" << std::endl;
            throw InvalidFile();
        }

        //header
        output << "graph " << mInputFile.substr(0, mInputFile.find('.')) << "{" << std::endl;
        output << "node [height=1, width=.1];\n"
                " rankdir=LR;" << std::endl;
        int count(0);

        //output nodes in order
        std::for_each(mUncontractedNodes.begin(), mUncontractedNodes.end(),
                      [&count, &output, &nodeNum](std::shared_ptr<Node> toAdd) {
                          output << "node" << count << " [" << "label=\"" << toAdd->GetTypeOfNodeString() << "\"";
                          if (toAdd->mRank == 1) {
                              output << ", height = .5";
                          }
                          output << "];" << std::endl;
                          nodeNum.insert({toAdd, count});
                          count++;
                      });

//output the connections between the nodes
        int k(0);
        for (auto &tempVect: mNodesByWire) {
            k = 0;
            for (int i = 0; i < tempVect.size(); i++) {
                if (i > 0 && tempVect[i] != nullptr) {
                    output << "node" << nodeNum[tempVect[k]] << " -- " << "node" << nodeNum[tempVect[i]] << std::endl;
                    k = i;
                }
            }
        }


        output << "}" << std::endl;
        output.close();
    }

/*This function takes in the string path of a file to output to, and outputs the circuit held in the network to a .dgf file
 * the file's contents can then be analyzed using treewidth software or used to as a graph in other ways
 */
    void Network::OutputCircuitToTreewidthGraph(const std::string &toOutputTo) const {
        int count(0);
        std::unordered_map<std::shared_ptr<Node>, int> nodeNum(mUncontractedNodes.size());
        std::ofstream output(toOutputTo);
        if (!output.is_open()) {
            std::cout << "Failure to Output Circuit to TW Graph" << std::endl;
            throw InvalidFile();
        }
        output << "c Created From File: " << mInputFile << std::endl;

        //give each node a vertex index
        std::for_each(mUncontractedNodes.begin(), mUncontractedNodes.end(),
                      [&count, &nodeNum](std::shared_ptr<Node> toAdd) {
                          nodeNum.insert({toAdd, count});
                          count++;
                      });

        //output the edges between the nodes
        int k(0);
        int counter(0);
        for (auto &tempVect: mNodesByWire) {
            k = 0;
            for (int i = 0; i < tempVect.size(); i++) {
                if (i > 0 && tempVect[i] != nullptr) {
                    if (!(i == tempVect.size() - 1 && counter == mNodesByWire.size() - 1)) {
                        output << "e " << nodeNum[tempVect[k]] << " " << nodeNum[tempVect[i]] << std::endl;
                    } else {
                        output << "e " << nodeNum[tempVect[k]] << " " << nodeNum[tempVect[i]];
                    }
                    k = i;
                }
            }
            counter++;
        }

    }

//this function takes in an array of node pointers, a node to find, and a node to replace it with
//the function then searches for the node and replaces it if found
    void
    Network::FindAndReplace(std::vector<std::vector<std::shared_ptr<Node>>> &toSearch, std::shared_ptr<Node> toFind,
                            std::shared_ptr<Node> toReplaceWith) const {
        int count = 0;
        for (auto &tempVect: toSearch) {
            for (auto &tempNode: tempVect) {
                if (tempNode == toFind) {
                    tempNode = toReplaceWith;
                    count++;
                }
                if (count >= 2) {
                    break;
                }
            }
        }

    }

//this function takes in a vector of node pointers, a node to find, and a node to replace it with
//the function then searches for the node and replaces it if found
    void Network::FindAndReplace(std::vector<std::shared_ptr<Node>> &toSearch, std::shared_ptr<Node> toFind,
                                 std::shared_ptr<Node> toReplaceWith) const {
        if (toSearch.empty()) {
            return;
        }
        for (auto &tempNode: toSearch) {
            if (tempNode == toFind) {
                tempNode = toReplaceWith;
                return;
            }
        }

    }

//this function takes in a vector of node pointers and a node to remove.. it then searches the vector for the node
//and if found, removes the node from the vector
    void Network::FindAndRemove(std::vector<std::shared_ptr<Node>> &vect, std::shared_ptr<Node> toRemove) const {
        if (vect.empty()) {
            return;
        }
        int i = 0;
        for (auto &v:vect) {
            if (v == toRemove) {
                vect.erase(vect.begin() + i);
                return;
            }
            i++;
        }

    }

}