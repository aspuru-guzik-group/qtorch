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



#pragma once

#include <algorithm> // npds 2016-12-12
#include "Network.h"
#include <iostream>
#include <memory>
#include <thread>
#include "zconf.h"
#include "Exceptions.h"
#include "LineGraph.h"



/*
The class ContractionTools is a wrapper class for a tensor network simulation of a quantum circuit.
 The constructor for the class is explicit, and you must provide the full path of the quantum
 circuit file you want to simulate on instantiation of a ContractionTools instance.

 Each instance of a ContractionTools class contains a mersenne twister random number generator, which
 is instantiated in the constructor by a random device. Additionally, the class contains mFinalVal,
 which is the final expected value from the tensor network contraction. The getter function is implemented
 inline. Additonally, the class contains a vector of vector pointers: mPartitionedNodes. This vector of vectors
 is used in the ParallelContract function, where the function contracts each different vector of nodes within
 the overlying vector in parallel.

 Additionally, the class contains functions for analyzing treewidth and visually representing the tensor network
 graph.

 See below for comments on individual functions
*/
namespace qtorch {

    enum ContractionType {
        Stochastic, FromEdges, CostContractSimple, CostContractBruteForce
    };

    class ContractionTools {
    public:
        explicit ContractionTools(const std::string &inputFile, const std::string &measureFile,
                                  const int numThreads = 8) : mString(inputFile), mMeasureFile(measureFile),
                                                              mCopyCreated(false), mNumThreadsInNetwork(
                        numThreads) { mRandGen = std::mt19937(mRandDevice()); };

        explicit ContractionTools(const std::shared_ptr<Network> network) : mCopyCreated(true) {
            mRandGen = std::mt19937(mRandDevice());
            mNetwork = network;
        };

        std::complex<double> GetFinalVal() noexcept { return mFinalVal; };

        std::shared_ptr<Network> Contract(ContractionType type, int pValue = 1, int numSamples = 1);

        std::shared_ptr<Network> ReduceAndPrintCircuitToVisualGraph(const std::string &toPrintTo) const;

        std::shared_ptr<Network> ReduceAndPrintCircuitToTWGraph(const std::string &toPrintTo) const;

        std::shared_ptr<Network> ContractUserDefinedSequenceOfWires(const std::string &userInputFilePath);

        const int CalculateTreewidth(const int qbbseconds, const bool sixtyFourBitOpSystem = true) const;

        std::shared_ptr<Network> ContractGivenSequence(const std::vector<std::pair<int, int>> &sequence);

        void Reset(const std::string &inputFile, const std::string &measureFile, const int numThreads = 8);

        void Reset();

        void Reset(std::shared_ptr<Network> network) {
            mNetwork = network;
            mCopyCreated = true;
        };
    private:
        std::string mString;
        std::string mMeasureFile;
        std::complex<double> mFinalVal;
        std::vector<std::shared_ptr<std::vector<std::shared_ptr<Node>>>> mPartitionedNodes;
        std::random_device mRandDevice;
        std::shared_ptr<Network> mNetwork;
        std::mt19937 mRandGen;
        bool mCopyCreated;
        int mNumThreadsInNetwork;
    protected:
        void CreateChunksOfNodes(std::shared_ptr<Network> &myNetwork);

        std::shared_ptr<Network> ParallelContract(std::mt19937 &randomGenerator);

        std::shared_ptr<Network> ContractFromEdges(std::mt19937 &randomGenerator);

        std::shared_ptr<Network> CostBasedContractionSimple(const int pValue);

        std::shared_ptr<Network> CostBasedContractionBruteForce(const int numSamples);

        int NumberOfConnectedWires(std::shared_ptr<Node> nodeA, std::shared_ptr<Node> nodeB);

        long long CalculateCost(const int pVal, const int indexA, const int indexB, const int thresholdFinalRank,
                                const int thresholdNumwires);

        const int NumberOfConnectedBetweenTwoSuperNodes(const std::vector<std::shared_ptr<Node>> &node1,
                                                        const std::vector<std::shared_ptr<Node>> &node2);

    };


    void ContractionTools::Reset() {
        mNetwork = nullptr;
        mCopyCreated = false;
    }

    void ContractionTools::Reset(const std::string &inputFile, const std::string &measureFile, const int numThreads) {
        mNetwork = nullptr;
        mCopyCreated = false;
        mString = inputFile;
        mMeasureFile = measureFile;
        mNumThreadsInNetwork = numThreads;
    }

//this function takes in an enum which is the contraction algorithm you want to run and runs that algorithm
//the function returns a pointer to the network which was contracted after the contraction is complete
    std::shared_ptr<Network> ContractionTools::Contract(ContractionType type, int pValue, int numSamples) {
        if (type == Stochastic) {
            return ParallelContract(mRandGen);
        } else if (type == FromEdges) {
            return ContractFromEdges(mRandGen);
        } else if (type == CostContractSimple) {
            return CostBasedContractionSimple(pValue);
        } else if (type == CostContractBruteForce) {
            return CostBasedContractionBruteForce(numSamples);
        }
        return nullptr;
    }

/*This method lets the user input a defined sequence of wires to contract. The wires are defined by a pair of the two corresponding node indices from the original graph.
 *the index of the node is defined by the order defined by the user in the qasm file. However, the first n indices are the initial states and the last n indices are the
 * projection measurements.
 */
    std::shared_ptr<Network>
    ContractionTools::ContractUserDefinedSequenceOfWires(const std::string &userInputFilePath) {
        std::shared_ptr<Network> myNetwork;
        if (!mCopyCreated) {
            myNetwork = std::make_shared<Network>(mString, mMeasureFile);
            myNetwork->SetNumThreads(mNumThreadsInNetwork);
        } else {
            myNetwork = mNetwork;
        }
        if (myNetwork->HasFailed()) //if you fail to open the network
        {
            return nullptr;
        }
        std::vector<std::pair<int, int>> wireOrdering;
        std::ifstream userInputFile(userInputFilePath);
        if (!userInputFile) {
            throw InvalidFile();
        }
        while (!userInputFile.eof()) {
            std::string temp;
            std::getline(userInputFile, temp);
            std::stringstream ss;
            ss << temp;
            int nodeOne;
            int nodeTwo;
            ss >> nodeOne >> nodeTwo;
            if (nodeOne < 0 || nodeOne > myNetwork->GetAllNodes().size() || nodeTwo < 0 ||
                nodeTwo > myNetwork->GetAllNodes().size()) {
                throw InvalidFileFormat();
            }
            wireOrdering.push_back({nodeOne, nodeTwo});
        }
        userInputFile.close();

        std::vector<int> map(myNetwork->GetAllNodes().size());
        int k(0);
        std::generate(map.begin(), map.end(), [&k]() {
            return k++;
        });

        for (int i = 0; i < wireOrdering.size(); i++) {
            //CONTRACT ORDERING and update map

            if (myNetwork->GetAllNodes()[map[wireOrdering[i].first]]->mContracted ||
                myNetwork->GetAllNodes()[map[wireOrdering[i].second]]->mContracted) {
                continue;
            }

            myNetwork->ContractNodes(myNetwork->GetAllNodes()[map[wireOrdering[i].first]],
                                     myNetwork->GetAllNodes()[map[wireOrdering[i].second]], 10000);
            map[wireOrdering[i].first] = myNetwork->GetAllNodes().size() - 1;
            map[wireOrdering[i].second] = myNetwork->GetAllNodes().size() - 1;
        }
        if (myNetwork->IsDone()) {
            mFinalVal = myNetwork->GetFinalValue();
        } else {
            std::cout << "Error - contraction sequence was incomplete." << std::endl;
            throw InvalidUserContractionSequence();
        }
        return myNetwork;
    }

//this contraction algorithm creates a network and runs an algorithm that divides the network into n partitions,
//contracting each partition separately before combining the result into one vector of the nodes that are left and
//contracting that final vector
    std::shared_ptr<Network> ContractionTools::ParallelContract(std::mt19937 &randomGenerator) {
        //create the initial network
        std::shared_ptr<Network> myNetwork;
        if (!mCopyCreated) {
            myNetwork = std::make_shared<Network>(mString, mMeasureFile);
            myNetwork->SetNumThreads(mNumThreadsInNetwork);
        } else {
            myNetwork = mNetwork;
        }
        if (myNetwork->HasFailed()) //if you fail to open the network
        {
            return nullptr;
        }
        std::vector<std::shared_ptr<std::vector<std::shared_ptr<Node>>>> tempVect; //create a temporary vector of partitioned nodes


        //this function takes in the network and divides the nodes into partitions, storing the partitions into
        //mPartitionedNodes
        CreateChunksOfNodes(myNetwork);


        //this lambda captures the network and random number generator, and takes in a vector of partitions and an integer k
        //which is the index in partition vector to contract. The lambda uses a random number generator to contract that
        //portion of the partition vector
        auto contractPieceFunction = [&myNetwork, &randomGenerator](
                std::vector<std::shared_ptr<std::vector<std::shared_ptr<Node>>>> temp, int k) {
            //threshold of accepted resulting rank value for contraction of two nodes
            // the function will not contract any two nodes unless the rank of the resulting node is
            //<= the max rank of the two input nodes + threshold
            int threshold = 1;

            //temp ptr to hold the result of contracting two nodes
            std::shared_ptr<Node> result;

            //to count the number of failed contractions (i.e. when the rank of the resulting node was above the threshold
            int failureCount = 0;

            //while there is more than 1 node left in the partition to contract and the failure count is less than
            //the number of nodes left in the partition squared
            while (temp[k]->size() != 1 && (failureCount < pow(temp[k]->size(), 2))) {
                //create a uniform int distribution over all nodes in the parition and choose two numbers
                std::uniform_int_distribution<> tempDist(0, temp[k]->size() - 1);
                int one = tempDist(randomGenerator);
                int two = tempDist(randomGenerator);
                if (one == two) //can't contract the same node with itself
                {
                    continue;
                }
                std::shared_ptr<Node> tempOne = (temp[k]->at(one));
                std::shared_ptr<Node> tempTwo = (temp[k]->at(two));

                //contract the nodes
                if (totTimer.getElapsed() > maxTime) {
                    return;
                }
                result = myNetwork->ContractNodes(tempOne, tempTwo, threshold);

                //if the contraction didn't fail
                if (result != nullptr) {

                    //remove the nodes from the partition
                    if (two == temp[k]->size() - 1) {
                        temp[k]->at(two) = temp[k]->back();
                        temp[k]->pop_back();
                        temp[k]->at(one) = temp[k]->back();
                        temp[k]->pop_back();
                    } else {
                        temp[k]->at(one) = temp[k]->back();
                        temp[k]->pop_back();
                        temp[k]->at(two) = temp[k]->back();
                        temp[k]->pop_back();
                    }

                    //reset the failure count and add the result from the contraction to the partition
                    failureCount = 0;
                    temp[k]->push_back(result);
                } else if (!myNetwork->IsDone()) //if the contraction failed
                {
                    failureCount++;
                } else //never accessed
                {
                    break;
                }
            }

        };


        //this next lambda takes in a vector pointer of nodes left and works similarly to the lambda above,
        //contracting the nodes in the vector until there is only a tensor of rank zero left
        auto contractNodesLeftFunction = [&myNetwork, &randomGenerator](std::vector<std::shared_ptr<Node>> *temp) {
            int threshold = -1;
            int fails = 0;
            std::shared_ptr<Node> result;
            while (!myNetwork->IsDone()) //while the network is fully contracted
            {
                if (fails > pow(temp->size(), 2)) //if the failure count grows too large, increase the threshold
                {
                    // std::cout<<"Too many failures"<<std::endl;
                    std::cout << "Nodes left: " << temp->size() << std::endl;
                    threshold++;
                    fails = 0;
                }
                //uncomment to debug
                //std::cout<<temp->size()<<std::endl;
                std::uniform_int_distribution<> tempDist(0, temp->size() - 1);
                int one = tempDist(randomGenerator);
                int two = tempDist(randomGenerator);
                if (one == two || temp->at(one)->mContracted || temp->at(two)->mContracted) {
                    continue;
                }
                std::shared_ptr<Node> tempOne = (temp->at(one));
                std::shared_ptr<Node> tempTwo = (temp->at(two));
                if (totTimer.getElapsed() > maxTime) {
                    return;
                }
                result = myNetwork->ContractNodes(tempOne, tempTwo, threshold);
                if (result != nullptr) {
                    if (two == temp->size() - 1) {
                        temp->at(two) = temp->back();
                        temp->pop_back();
                        temp->at(one) = temp->back();
                        temp->pop_back();
                    } else {
                        temp->at(one) = temp->back();
                        temp->pop_back();
                        temp->at(two) = temp->back();
                        temp->pop_back();
                    }
                    temp->push_back(result);
                    fails = 0;
                    threshold = -1;
                } else {
                    fails++;
                }
            }

        };


        //create threads to parallelize the contraction of all the different partitions
        std::vector<std::thread> threads;
        for (int i = 0; i < mPartitionedNodes.size(); i++) {
            tempVect.push_back(mPartitionedNodes[i]);
            threads.push_back(std::thread(contractPieceFunction, tempVect, i));
        }
        //join the threads
        for (auto &t: threads) {
            t.join();
        }

        //copy over all the nodes that are left in the partitions into one vector
        std::vector<std::shared_ptr<Node>> nodesLeft;
        for (int i = 0; i < tempVect.size(); i++) {
            std::copy(tempVect[i]->begin(), tempVect[i]->end(), std::back_inserter(nodesLeft));
        }

        //contract that vector on a separate thread
        contractNodesLeftFunction(&nodesLeft);

        //if the network is done contracting
        if (myNetwork->IsDone()) {
            mFinalVal = myNetwork->GetFinalValue();
        } else //never reached
        {
            std::cout << "Error contracting network did not result in a final value..." << std::endl;
            throw ContractionFailure();
        }
        return myNetwork;
    }

//this contraction algorithm takes in a vector of pairs and contracts the nodes in the network according the the pair sequence
//defined
    std::shared_ptr<Network> ContractionTools::ContractGivenSequence(const std::vector<std::pair<int, int>> &sequence) {
        if (!mCopyCreated) {
            mNetwork = std::make_shared<Network>(mString, mMeasureFile);
            mNetwork->SetNumThreads(mNumThreadsInNetwork);
        }
        if (mNetwork->HasFailed()) //if you fail to open the network
        {
            return nullptr;
        }

        for (const std::pair<int, int> &pair:sequence) {
            mNetwork->ContractNodes(mNetwork->GetAllNodes()[pair.first], mNetwork->GetAllNodes()[pair.second], 100);
        }
        if (mNetwork->IsDone()) {
            mFinalVal = mNetwork->GetFinalValue();
        } else {
            throw ContractionFailure();
        }
        return mNetwork;
    }

/*
 *
 * It would, based on the size on the graph, calculate the cost of n random connections.
 * The cost would be based on the size of the two connected tensors and the size of the resulting tensor.
 * However, the user can also provide a p value, which will determine how much of the tensor network is
 * seen by the cost calculating function. I.e. for p=1, the cost will be calculated from just the two
 * connected tensors and the resulting tensor. For p=2, the cost will be calculated from the two connected tensors,
 * the resulting tensor, and then that resulting tensors' cost of being contracted with its neighbors.
 * Therefore, we would be able to obtain the optimum contraction sequence as p goes to infinity.
 */
    struct retVal {
        int indexOne;
        int indexTwo;
        int maxRank;
        std::vector<int> rankCounter;
        bool fail;
    };

    std::shared_ptr<Network> ContractionTools::CostBasedContractionBruteForce(const int numSamples) {
        if (!mCopyCreated) {
            mNetwork = std::make_shared<Network>(mString, mMeasureFile);
            mNetwork->SetNumThreads(mNumThreadsInNetwork);
        }
        std::shared_ptr<Network> myNetwork = mNetwork;
        std::mt19937 tempRandGen = mRandGen;
        std::vector<std::vector<std::pair<int, int>>> ncrPairsUpTo30;
        int nCrRange(30);
        ncrPairsUpTo30.resize(nCrRange + 1);
        for (int i(2); i <= nCrRange; i++) {
            for (int j(0); j <= i - 1; j++) {
                for (int k(j); k <= i - 1; ++k) {
                    if (j == k) {
                        continue;
                    }
                    ncrPairsUpTo30[i].push_back(std::pair<int, int>(j, k));
                }
            }
        }

        std::mutex protector;
        auto samplerFunction = [&tempRandGen, &protector, &ncrPairsUpTo30, &myNetwork](int numberOfSamples, retVal *ret,
                                                                                       int threshold) {
            auto numberOfConnectedWiresLocal = [](std::shared_ptr<Node> nodeA, std::shared_ptr<Node> nodeB) {
                int count(0);
                for (const auto &wire: nodeA->GetWires()) {
                    if (wire->GetNodeA().lock() == nodeB || wire->GetNodeB().lock() == nodeB) {
                        count++;
                    }
                }
                return count;

            };
            if (myNetwork->GetUncontractedNodes().size() == 2) {
                ret->indexOne = 0;
                ret->indexTwo = 1;
                ret->maxRank =
                        myNetwork->GetUncontractedNodes()[0]->mRank + myNetwork->GetUncontractedNodes()[1]->mRank - 2 *
                                                                                                                    numberOfConnectedWiresLocal(
                                                                                                                            myNetwork->GetUncontractedNodes()[0],
                                                                                                                            myNetwork->GetUncontractedNodes()[1]);
                return ret;
            }
            auto numberOfConnectedWiresSuperNodesLocal = [&numberOfConnectedWiresLocal](
                    const std::vector<std::shared_ptr<Node>> &node1, const std::vector<std::shared_ptr<Node>> &node2) {
                int cost(0);
                for (const auto &nodeA: node1) {
                    for (const auto &nodeB: node2) {
                        cost += numberOfConnectedWiresLocal(nodeA, nodeB);
                    }
                }
                return cost;
            };
            auto incrementCounter = [](std::vector<int> &counter,
                                       std::vector<std::vector<std::pair<int, int>>> &ncrPairs) {
                if (counter.size() == 0) {
                    return false;
                }
                ++counter[0];
                for (int j(0); j < counter.size(); ++j) {
                    if (counter[j] > ncrPairs[j + 3].size() - 1) {
                        counter[j] = 0;
                        if (j != counter.size() - 1) {
                            ++counter[j + 1];
                        } else {
                            return false;
                        }
                    } else {
                        break;
                    }
                }
                return true;
            };
            /* if(ncrPairsUpTo30[myNetwork->GetUncontractedNodes().size()].size()<numberOfSamples)
             {
                 numberOfSamples = ncrPairsUpTo30[myNetwork->GetUncontractedNodes().size()].size();
             }*/
            ret->maxRank = 1;
            ret->indexOne = 0;
            ret->indexTwo = 0;
            std::uniform_int_distribution<> tempDist(0, static_cast<int>(myNetwork->GetUncontractedNodes().size()) - 1);
            int fails(0);
            int fails2(0);
            for (int i = 0; i < numberOfSamples && totTimer.getElapsed() < maxTime; i++) {
                int indexA = tempDist(tempRandGen);
                int indexB = tempDist(tempRandGen);
                if (indexA == indexB || numberOfConnectedWiresLocal(myNetwork->GetUncontractedNodes()[indexA],
                                                                    myNetwork->GetUncontractedNodes()[indexB]) == 0) {
                    --i;
                    continue;
                }
                if (fails > myNetwork->GetUncontractedNodes().size() * 10) {
                    ret->fail = true;
                    return ret;
                }
                if (myNetwork->GetUncontractedNodes()[indexA]->mRank +
                    myNetwork->GetUncontractedNodes()[indexB]->mRank - 2 * numberOfConnectedWiresLocal(
                        myNetwork->GetUncontractedNodes()[indexA], myNetwork->GetUncontractedNodes()[indexB]) >
                    threshold) {
                    --i;
                    fails++;
                    continue;
                }
                std::vector<int> costFunction(20, 0);
                int maxRank(0);
                //tempCost = static_cast<long long>(pow(4,myNetwork->GetUncontractedNodes()[indexA]->mRank + myNetwork->GetUncontractedNodes()[indexB]->mRank -  numberOfConnectedWiresLocal(myNetwork->GetUncontractedNodes()[indexA], myNetwork->GetUncontractedNodes()[indexB])));
                protector.lock();
                std::vector<std::shared_ptr<Node>> rangeOfAlgorithm;
                rangeOfAlgorithm.push_back(myNetwork->GetUncontractedNodes()[indexA]);
                rangeOfAlgorithm.push_back(myNetwork->GetUncontractedNodes()[indexB]);
                myNetwork->GetUncontractedNodes()[indexA]->mSelectedInCostContractionAlgorithm = true;
                myNetwork->GetUncontractedNodes()[indexB]->mSelectedInCostContractionAlgorithm = true;
                std::vector<std::shared_ptr<Node>> tempNodesList;
                int currentRank = 0;
                int currentIndex = 0;
                for (const auto &wire: rangeOfAlgorithm[0]->GetWires()) {
                    if (!wire->GetNodeA().lock()->mSelectedInCostContractionAlgorithm) {
                        //add to tempList and set to true;
                        tempNodesList.push_back(wire->GetNodeA().lock());
                        wire->GetNodeA().lock()->mSelectedInCostContractionAlgorithm = true;
                    } else if (!wire->GetNodeB().lock()->mSelectedInCostContractionAlgorithm) {
                        tempNodesList.push_back(wire->GetNodeB().lock());
                        wire->GetNodeB().lock()->mSelectedInCostContractionAlgorithm = true;
                    }
                }
                for (const auto &wire: rangeOfAlgorithm[1]->GetWires()) {
                    if (!wire->GetNodeA().lock()->mSelectedInCostContractionAlgorithm) {
                        //add to tempList and set to true;
                        tempNodesList.push_back(wire->GetNodeA().lock());
                        wire->GetNodeA().lock()->mSelectedInCostContractionAlgorithm = true;
                    } else if (!wire->GetNodeB().lock()->mSelectedInCostContractionAlgorithm) {
                        tempNodesList.push_back(wire->GetNodeB().lock());
                        wire->GetNodeB().lock()->mSelectedInCostContractionAlgorithm = true;
                    }
                }

                std::copy(tempNodesList.begin(), tempNodesList.end(), std::back_inserter(rangeOfAlgorithm));
                tempNodesList.clear();

                std::vector<std::vector<int>> adjacencyMatrixPermanent;
                std::vector<int> ranksPermanent(rangeOfAlgorithm.size() - 1);
                ranksPermanent[0] = rangeOfAlgorithm[0]->mRank + rangeOfAlgorithm[1]->mRank -
                                    2 * numberOfConnectedWiresLocal(rangeOfAlgorithm[0], rangeOfAlgorithm[1]);
                maxRank = ranksPermanent[0];
                int l(2);
                std::generate(ranksPermanent.begin() + 1, ranksPermanent.end(), [&rangeOfAlgorithm, &l]() {
                    return rangeOfAlgorithm[l++]->mRank;
                });
                for (int j(1); j < rangeOfAlgorithm.size(); j++) {
                    std::vector<int> tempVect(rangeOfAlgorithm.size() - 1);
                    if (j == 1) {
                        tempVect[0] = 0;
                        for (int k(2); k < rangeOfAlgorithm.size(); k++) {
                            tempVect[k - 1] = numberOfConnectedWiresSuperNodesLocal(
                                    {rangeOfAlgorithm[0], rangeOfAlgorithm[1]}, {rangeOfAlgorithm[k]});
                        }
                        adjacencyMatrixPermanent.push_back(std::move(tempVect));
                    } else {
                        for (int k(1); k < rangeOfAlgorithm.size(); k++) {
                            if (k < j) {
                                tempVect[k - 1] = adjacencyMatrixPermanent[k - 1][j - 1];
                            } else if (k == j) {
                                tempVect[k - 1] = 0;
                            } else {
                                tempVect[k - 1] = numberOfConnectedWiresLocal(rangeOfAlgorithm[j], rangeOfAlgorithm[k]);
                            }
                        }
                        adjacencyMatrixPermanent.push_back(tempVect);
                    }
                }
                for (auto &r:rangeOfAlgorithm) {
                    r->mSelectedInCostContractionAlgorithm = false;
                }
                protector.unlock();
                //maybe a threshold for the number of iterations in this loop!
                long long numIterationsInBruteForceLoop(1);
                for (int j(3); j <= rangeOfAlgorithm.size() - 1; j++) {
                    numIterationsInBruteForceLoop *= ncrPairsUpTo30[j].size();
                }
                if (fails2 > myNetwork->GetUncontractedNodes().size() * 10) {
                    ret->indexOne = indexA;
                    ret->indexTwo = indexB;
                    return ret;
                }
                if (numIterationsInBruteForceLoop > 1000000) {
                    --i;
                    fails2++;
                    continue;
                }
                std::vector<int> counter(rangeOfAlgorithm.size() - 3, 0);
                bool isFirst(false);
                do {
                    std::vector<std::vector<int>> adjacencyMatrix(adjacencyMatrixPermanent);
                    std::vector<int> ranks(ranksPermanent);
                    bool flag(false);
                    int maxRankTemp(0);
                    std::vector<int> rankCounter(20, 0);
                    std::vector<int> indicesLeft(rangeOfAlgorithm.size() - 1);
                    int l(-1);
                    std::generate(indicesLeft.begin(), indicesLeft.end(), [&l]() { return ++l; });
                    for (int c(counter.size() - 1); c >= 0; --c) {
                        if (adjacencyMatrix[indicesLeft[ncrPairsUpTo30[c +
                                                                       3][counter[c]].first]][indicesLeft[ncrPairsUpTo30[
                                c + 3][counter[c]].second]] == 0) {
                            flag = true;
                            break;
                        } else {
                            ranks[indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].first]] =
                                    ranks[indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].first]] +
                                    ranks[indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].second]] - 2 *
                                                                                                   adjacencyMatrix[indicesLeft[ncrPairsUpTo30[
                                                                                                           c +
                                                                                                           3][counter[c]].first]][indicesLeft[ncrPairsUpTo30[
                                                                                                           c +
                                                                                                           3][counter[c]].second]];
                            if (ranks[indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].first]] > 17) {
                                flag = true;
                                break;
                            }
                            if (ranks[indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].first]] > maxRankTemp) {
                                maxRankTemp = ranks[indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].first]];
                            }
                            rankCounter[ranks[indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].first]]]++;
                            ranks[indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].second]] = 0;
                            for (int d(0); d < adjacencyMatrix.size(); d++) {
                                if (d != indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].first]) {
                                    adjacencyMatrix[indicesLeft[ncrPairsUpTo30[c +
                                                                               3][counter[c]].first]][d] += adjacencyMatrix[indicesLeft[ncrPairsUpTo30[
                                            c + 3][counter[c]].second]][d];
                                    adjacencyMatrix[d][indicesLeft[ncrPairsUpTo30[c +
                                                                                  3][counter[c]].first]] = adjacencyMatrix[indicesLeft[ncrPairsUpTo30[
                                            c + 3][counter[c]].first]][d];
                                }
                                adjacencyMatrix[indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].second]][d] = 0;
                                adjacencyMatrix[d][indicesLeft[ncrPairsUpTo30[c + 3][counter[c]].second]] = 0;
                            }
                            indicesLeft.erase(indicesLeft.begin() + ncrPairsUpTo30[c + 3][counter[c]].second);
                        }
                    }
                    if (flag) {
                        continue;
                    }
                    if (isFirst) {
                        maxRank = maxRankTemp;
                        costFunction = rankCounter;
                    } else if (maxRank < maxRankTemp) {
                        maxRank = maxRankTemp;
                        costFunction = rankCounter;
                    } else if (maxRank == maxRankTemp) {
                        for (int q(maxRankTemp - 1); q > 1; --q) {
                            if (costFunction[q] > rankCounter[q]) {
                                maxRank = maxRankTemp;
                                costFunction = rankCounter;
                                break;
                            } else if (costFunction[q] < rankCounter[q]) {
                                break;
                            }
                        }
                    }
                    isFirst = true;
                } while (incrementCounter(counter, ncrPairsUpTo30) && totTimer.getElapsed() < maxTime);

                if (ret->maxRank == 1) {
                    ret->indexOne = indexA;
                    ret->indexTwo = indexB;
                    ret->maxRank = maxRank;
                    ret->rankCounter = costFunction;
                } else if (maxRank < ret->maxRank) {
                    ret->indexOne = indexA;
                    ret->indexTwo = indexB;
                    ret->maxRank = maxRank;
                    ret->rankCounter = costFunction;
                } else if (maxRank == ret->maxRank) {
                    for (int q(maxRank - 1); q > 1; --q) {
                        if (costFunction[q] > ret->rankCounter[q]) {
                            ret->indexOne = indexA;
                            ret->indexTwo = indexB;
                            ret->maxRank = maxRank;
                            ret->rankCounter = costFunction;
                            break;
                        } else if (costFunction[q] < ret->rankCounter[q]) {
                            break;
                        }
                    }
                }
            }

            return ret;
        };

        if (myNetwork->HasFailed()) //if you fail to open the network
        {
            return nullptr;
        }
        int threshold = 10;
        while (!myNetwork->IsDone() && (totTimer.getElapsed() < maxTime)) {
            retVal rFinal;
            retVal r1;
            r1.fail = false;
            std::thread t1(samplerFunction, 3, &r1, threshold);
            retVal r2;
            r2.fail = false;
            std::thread t2(samplerFunction, 3, &r2, threshold);
            /* retVal r3;
             r3.fail=false;
             std::thread t3(samplerFunction,1,&r3,threshold);
             retVal r4;
             r4.fail=false;
             std::thread t4(samplerFunction,1,&r4,threshold);
             retVal r5;
             r5.fail=false;
             std::thread t5(samplerFunction,1,&r5,threshold);
             t1.join();
             t2.join();
             t3.join();
             t4.join();
             t5.join();
             if(r1.fail&&r2.fail&&r3.fail&&r4.fail&&r5.fail)
             {
                 threshold++;
                 continue;
             }
             rFinal = std::min({r1,r2,r3,r4,r5},[](retVal rOne, retVal rTwo){
                 if(rOne.fail&&rTwo.fail)
                 {
                     return true;
                 }
                 else if(rTwo.fail)
                 {
                     return true;
                 }
                 else if(rOne.fail)
                 {
                     return false;
                 }
                 if(rOne.maxRank>rTwo.maxRank)
                 {
                     return false;
                 }
                 else if(rOne.maxRank==rTwo.maxRank)
                 {
                     for (int q(rOne.maxRank - 1); q > 1; --q) {
                         if (rOne.rankCounter[q] > rTwo.rankCounter[q]) {
                             return false;
                         } else if (rOne.rankCounter[q] < rTwo.rankCounter[q]) {
                             return true;
                         }
                     }
                     return true;
                 }
                 else
                 {
                     return true;
                 }
             });

             */




            t1.join();
            t2.join();

            if (r1.fail && r2.fail) {
                ++threshold;
                continue;
            }
            rFinal = std::min({r1, r2}, [](retVal rOne, retVal rTwo) {
                if (rOne.fail && rTwo.fail) {
                    return true;
                } else if (rTwo.fail) {
                    return true;
                } else if (rOne.fail) {
                    return false;
                }
                if (rOne.maxRank > rTwo.maxRank) {
                    return false;
                } else if (rOne.maxRank == rTwo.maxRank) {
                    for (int q(rOne.maxRank - 1); q > 1; --q) {
                        if (rOne.rankCounter[q] > rTwo.rankCounter[q]) {
                            return false;
                        } else if (rOne.rankCounter[q] < rTwo.rankCounter[q]) {
                            return true;
                        }
                    }
                    return true;
                } else {
                    return true;
                }
            });
            myNetwork->ContractNodes(myNetwork->GetUncontractedNodes()[rFinal.indexOne],
                                     myNetwork->GetUncontractedNodes()[rFinal.indexTwo], 15);
            std::cout << "Nodes Left: " << myNetwork->GetUncontractedNodes().size() << std::endl;
        }
        if (myNetwork->IsDone()) {
            mFinalVal = myNetwork->GetFinalValue();
        } else {
            throw ContractionFailure();
            return nullptr;
        }
        return myNetwork;

    }

    std::shared_ptr<Network> ContractionTools::CostBasedContractionSimple(const int pValue) {

        if (!mCopyCreated) {
            mNetwork = std::make_shared<Network>(mString, mMeasureFile);
            mNetwork->SetNumThreads(mNumThreadsInNetwork);
        }
        if (mNetwork->HasFailed()) //if you fail to open the network
        {
            return nullptr;
        }
        while (!mNetwork->IsDone() && totTimer.getElapsed() < maxTime) {
            long long minCost(-1);
            int indicesOfMin[2];
            indicesOfMin[0] = 0;
            indicesOfMin[1] = 1;
            std::uniform_int_distribution<> tempDist(0, mNetwork->GetUncontractedNodes().size() - 1);
            int failureCount(0);
            int finalRankThreshold(11);
            int connectedWiresThreshold(8);
            if (mNetwork->GetUncontractedNodes().size() != 2) {
                for (int i = 0;
                     i < log2(mNetwork->GetUncontractedNodes().size()) && totTimer.getElapsed() < maxTime; i++) {
                    int one(tempDist(mRandGen));
                    int two(tempDist(mRandGen));
                    if (NumberOfConnectedWires(mNetwork->GetUncontractedNodes()[one],
                                               mNetwork->GetUncontractedNodes()[two]) == 0 ||
                        (indicesOfMin[0] == one && indicesOfMin[1] == two) ||
                        (indicesOfMin[1] == one && indicesOfMin[0] == two) || one == two) {
                        i--;
                        continue;
                    }
                    long long costTemp(CalculateCost(pValue, one, two, finalRankThreshold, connectedWiresThreshold));
                    if (costTemp == -1) {
                        ++failureCount;
                        --i;
                        if (failureCount > mNetwork->GetUncontractedNodes().size()) {
                            finalRankThreshold++;
                            connectedWiresThreshold++;
                            failureCount = 0;
                        }
                        continue;
                    }
                    if (costTemp < minCost || i == 0) {
                        indicesOfMin[0] = one;
                        indicesOfMin[1] = two;
                        minCost = costTemp;
                    }

                }
            }
            mNetwork->ContractNodes(mNetwork->GetUncontractedNodes()[indicesOfMin[0]],
                                    mNetwork->GetUncontractedNodes()[indicesOfMin[1]], 1000000);
            std::cout << "Nodes Left: " << mNetwork->GetUncontractedNodes().size() << std::endl;
        }

        if (mNetwork->IsDone()) {
            mFinalVal = mNetwork->GetFinalValue();
        } else {
            throw ContractionFailure();
        }
        return mNetwork;

    }

    long long
    ContractionTools::CalculateCost(const int pVal, const int indexA, const int indexB, const int thresholdFinalRank,
                                    const int thresholdNumwires) {

        /*
        if(pVal == 0)
        {
            return mNetwork->GetUncontractedNodes()[indexA]->mRank + mNetwork->GetUncontractedNodes()[indexB]->mRank -  NumberOfConnectedWires(mNetwork->GetUncontractedNodes()[indexA], mNetwork->GetUncontractedNodes()[indexB]);
        }
        else
        {
            std::vector<std::shared_ptr<Node>> rangeOfAlgorithm;
            rangeOfAlgorithm.push_back(mNetwork->GetUncontractedNodes()[indexA]);
            rangeOfAlgorithm.push_back(mNetwork->GetUncontractedNodes()[indexB]);
            mNetwork->GetUncontractedNodes()[indexA]->mSelectedInCostContractionAlgorithm = true;
            mNetwork->GetUncontractedNodes()[indexB]->mSelectedInCostContractionAlgorithm = true;
            std::vector<std::shared_ptr<Node>> tempNodesList;
            int currentRank = 0;
            int currentIndex = 0;
            while(currentRank<pVal)
            {
                for(currentIndex = currentIndex; currentIndex<rangeOfAlgorithm.size();currentIndex++)
                {

                    for (const auto &wire: rangeOfAlgorithm[currentIndex]->GetWires()) {
                        if (wire->GetNodeA()->mSelectedInCostContractionAlgorithm == false)
                        {
                            //add to tempList and set to true;
                            tempNodesList.push_back(wire->GetNodeA());
                            wire->GetNodeA()->mSelectedInCostContractionAlgorithm = true;
                        }
                        else if(wire->GetNodeB()->mSelectedInCostContractionAlgorithm == false)
                        {
                            tempNodesList.push_back(wire->GetNodeB());
                            wire->GetNodeB()->mSelectedInCostContractionAlgorithm = true;
                        }
                    }
                }
                std::copy(tempNodesList.begin(),tempNodesList.end(),std::back_inserter(rangeOfAlgorithm));
                tempNodesList.clear();
                currentRank++;
            }
            for(auto& r:rangeOfAlgorithm)
            {
                r->mSelectedInCostContractionAlgorithm=false;
            }
            return CalcHelper(rangeOfAlgorithm);
        }
         */
        if (pVal == 0) {
            return mNetwork->GetUncontractedNodes()[indexA]->mRank + mNetwork->GetUncontractedNodes()[indexB]->mRank -
                   NumberOfConnectedWires(mNetwork->GetUncontractedNodes()[indexA],
                                          mNetwork->GetUncontractedNodes()[indexB]);
        } else if (mNetwork->GetUncontractedNodes()[indexA]->mRank + mNetwork->GetUncontractedNodes()[indexB]->mRank -
                   2 * NumberOfConnectedWires(mNetwork->GetUncontractedNodes()[indexA],
                                              mNetwork->GetUncontractedNodes()[indexB]) > thresholdFinalRank ||
                   NumberOfConnectedWires(mNetwork->GetUncontractedNodes()[indexA],
                                          mNetwork->GetUncontractedNodes()[indexB]) > thresholdNumwires) {
            return -1;
        } else {
            std::vector<std::shared_ptr<Node>> selectedNodes;
            std::vector<std::shared_ptr<Node>> neighbors;
            int tempNumberConnectedWires(NumberOfConnectedWires(mNetwork->GetUncontractedNodes()[indexA],
                                                                mNetwork->GetUncontractedNodes()[indexB]));
            long long cost(static_cast<long long>(pow(4, mNetwork->GetUncontractedNodes()[indexA]->mRank +
                                                         mNetwork->GetUncontractedNodes()[indexB]->mRank -
                                                         tempNumberConnectedWires)));
            int rankOfSelected(
                    mNetwork->GetUncontractedNodes()[indexA]->mRank + mNetwork->GetUncontractedNodes()[indexB]->mRank -
                    2 * tempNumberConnectedWires);
            selectedNodes.push_back(mNetwork->GetUncontractedNodes()[indexA]);
            mNetwork->GetUncontractedNodes()[indexA]->mSelectedInCostContractionAlgorithm = true;
            selectedNodes.push_back(mNetwork->GetUncontractedNodes()[indexB]);
            mNetwork->GetUncontractedNodes()[indexB]->mSelectedInCostContractionAlgorithm = true;
            for (const auto &node:selectedNodes) {
                for (const auto &wire:node->GetWires()) {
                    if (!(wire->GetNodeA().lock()->mSelectedInCostContractionAlgorithm &&
                          wire->GetNodeB().lock()->mSelectedInCostContractionAlgorithm)) {
                        if (wire->GetNodeA().lock()->mSelectedInCostContractionAlgorithm) {
                            wire->GetNodeB().lock()->mSelectedInCostContractionAlgorithm = true;
                            neighbors.push_back(wire->GetNodeB().lock());
                        } else {
                            wire->GetNodeA().lock()->mSelectedInCostContractionAlgorithm = true;
                            neighbors.push_back(wire->GetNodeA().lock());
                        }
                    }
                }
            }
            int failureCount(0);
            for (int i = 0; i < pVal; i++) {
                if (neighbors.size() == 0) {
                    return cost;
                }
                //so I have four options for this: 1. I could take the max cost of contracting the supernode with one of its neighbors, repeating until i = p
                //2. take the min cost of contracting with one of its neighbors, repeating...
                //3. calculate both of the above and take the average
                //4. randomly choose nodes
                //I went with randomly choosing

                std::uniform_int_distribution<> dist(0, neighbors.size() - 1);
                int randNum(dist(mRandGen));//randomly pick a node from neighbors
                while (neighbors[randNum] == nullptr) {
                    randNum = dist(mRandGen);//randomly pick a node from neighbors
                }
                tempNumberConnectedWires = NumberOfConnectedBetweenTwoSuperNodes(selectedNodes, {neighbors[randNum]});
                if (rankOfSelected + neighbors[randNum]->mRank - tempNumberConnectedWires >= 12 &&
                    failureCount < neighbors.size() * 2) {
                    failureCount++;
                    i--;
                    continue;
                } else if (failureCount > neighbors.size() * 2) {
                    return -1;
                } else {
                    failureCount = 0;
                }
                cost += pow(4, rankOfSelected) * pow(4, neighbors[randNum]->mRank) / pow(4, tempNumberConnectedWires);
                rankOfSelected = rankOfSelected + neighbors[randNum]->mRank - 2 * tempNumberConnectedWires;
                for (const auto &wire: neighbors[randNum]->GetWires()) {
                    if (!(wire->GetNodeA().lock()->mSelectedInCostContractionAlgorithm &&
                          wire->GetNodeB().lock()->mSelectedInCostContractionAlgorithm)) {
                        if (wire->GetNodeA().lock()->mSelectedInCostContractionAlgorithm) {
                            wire->GetNodeB().lock()->mSelectedInCostContractionAlgorithm = true;
                            neighbors.push_back(wire->GetNodeB().lock());
                        } else {
                            wire->GetNodeA().lock()->mSelectedInCostContractionAlgorithm = true;
                            neighbors.push_back(wire->GetNodeA().lock());
                        }
                    }
                }
                selectedNodes.push_back(std::move(neighbors[randNum]));

            }

            for (auto &node:selectedNodes) {
                node->mSelectedInCostContractionAlgorithm = false;
            }
            for (auto &node:neighbors) {
                if (node != nullptr) {
                    node->mSelectedInCostContractionAlgorithm = false;
                }
            }
            return cost;
        }


    }

    const int ContractionTools::NumberOfConnectedBetweenTwoSuperNodes(const std::vector<std::shared_ptr<Node>> &node1,
                                                                      const std::vector<std::shared_ptr<Node>> &node2) {
        int cost(0);
        for (const auto &nodeA: node1) {
            for (const auto &nodeB: node2) {
                cost += NumberOfConnectedWires(nodeA, nodeB);
            }
        }
        return cost;
    }


    int ContractionTools::NumberOfConnectedWires(std::shared_ptr<Node> nodeA, std::shared_ptr<Node> nodeB) {
        int count(0);
        for (const auto &wire: nodeA->GetWires()) {
            if (wire->GetNodeA().lock() == nodeB || wire->GetNodeB().lock() == nodeB) {
                count++;
            }
        }
        return count;
    }

//this function is used to contract the network from the edges of the network (measurements and initial states)
//this is not a parallel function and returns a pointer to the fully contracted network
    std::shared_ptr<Network> ContractionTools::ContractFromEdges(std::mt19937 &randomGenerator) {
        std::shared_ptr<Network> myNetwork;
        if (!mCopyCreated) {
            myNetwork = std::make_shared<Network>(mString, mMeasureFile);
            myNetwork->SetNumThreads(mNumThreadsInNetwork);
        } else {
            myNetwork = mNetwork;
        }
        if (myNetwork->HasFailed()) {
            return nullptr;
        }

        //this function takes the vectors of all nodes in myNetwork and moves the initial state nodes to the back of the vector
        //so they can be split off from the rest of the nodes
        myNetwork->MoveInitialStatesToBack();

        //this lambda takes in a vector pointer of nodes in temp, which are all the nodes in the network minus the initial states
        //and the measurements. The lambda also takes in the vector pointer of working nodes, which is a vector with all the initial states
        //and measurements

        //The lambda picks random nodes from the vectors, one from either the working nodes vector or the temp vector
         //and one from the working nodes vector and attempts to contract them. If the contraction succeeds, the result is added
         //to the working nodes vector. Else, the failure count increases
         // The contraction will not succeed if the resulting node's rank does not pass the threshold function

        auto contractNodesLeftFunction = [&myNetwork, &randomGenerator](std::vector<std::shared_ptr<Node>> *temp,
                                                                        std::vector<std::shared_ptr<Node>> *working) {
            //initialize temp variables including threshold
            int threshold = -1;
            int fails = 0;
            std::shared_ptr<Node> result;
            while (!myNetwork->IsDone()) {

                if (fails > 100000) //if too many failures, increase the threshold
                {
                    // std::cout<<"Too many failures"<<std::endl;
                    // std::cout<<"Nodes left: "<<temp->size()<<std::endl;
                    threshold++;
                    fails = 0;
                }

                //uncomment to debug
                // std::cout<<temp->size()<<std::endl;

                //pick two nodes, one from either working or temp and one from working
                std::uniform_int_distribution<> tempDist(0, temp->size() + working->size() - 1);
                std::uniform_int_distribution<> tempDistEnd(0, working->size() - 1);
                int one = tempDist(randomGenerator);
                int two = tempDistEnd(randomGenerator);
                std::shared_ptr<Node> tempOne;
                bool flag = false;

                //find out whether the first node picked is in working or temp
                if (one >= temp->size()) {
                    flag = true;
                    one -= temp->size();
                    tempOne = working->at(one);
                } else {
                    tempOne = temp->at(one);
                }
                if (flag && one == two) {
                    continue;
                }
                std::shared_ptr<Node> tempTwo = (working->at(two));

                //contract the two nodes
                result = myNetwork->ContractNodes(tempOne, tempTwo, threshold);
                if (result != nullptr) //if contraction doesn't fail, remove the nodes and add the result
                {

                    if (flag) {
                        if (two > one) {
                            working->at(two) = working->back();
                            working->pop_back();
                            working->at(one) = working->back();
                            working->pop_back();
                        } else {
                            working->at(one) = working->back();
                            working->pop_back();
                            working->at(two) = working->back();
                            working->pop_back();
                        }
                    } else {
                        working->at(two) = working->back();
                        working->pop_back();
                        temp->at(one) = temp->back();
                        temp->pop_back();
                    }
                    working->push_back(result);
                    fails = 0;
                    threshold = -1;
                } else {
                    fails++;
                }
            }

        };

        //copy over all of the working nodes (initial states and measurements)
        std::vector<std::shared_ptr<Node>> nodesLeft;
        std::vector<std::shared_ptr<Node>> workingNodes;
        nodesLeft.reserve(myNetwork->GetUncontractedNodes().size());
        auto iterator = myNetwork->GetUncontractedNodes().begin();
        std::copy_n(iterator, myNetwork->GetUncontractedNodes().size() - 2 * myNetwork->GetNumQubits(),
                    std::back_inserter(nodesLeft));
        std::advance(iterator, myNetwork->GetUncontractedNodes().size() - 2 * myNetwork->GetNumQubits());
        std::copy(iterator, myNetwork->GetUncontractedNodes().end(), std::back_inserter(workingNodes));

        //call the lambda in a separate thread
        std::thread finalThread(contractNodesLeftFunction, &nodesLeft, &workingNodes);
        finalThread.join();
        if (myNetwork->IsDone()) {
            mFinalVal = myNetwork->GetFinalValue();
        } else //never reached
        {
            std::cout << "Error contracting network did not result in a final value..." << std::endl;
        }
        return myNetwork;
    }


//This function takes in a file path to print to, creates a network, removes all the 1 qubit gates and any
//successive 2 qubit gates before printing the graph to a file that can be analyzed by the treewidth
//library. The function returns a network pointer
    std::shared_ptr<Network> ContractionTools::ReduceAndPrintCircuitToTWGraph(const std::string &toPrintTo) const {
        std::shared_ptr<Network> myNetwork;
        if (!mCopyCreated) {
            myNetwork = std::make_shared<Network>(mString, mMeasureFile);
            myNetwork->SetNumThreads(mNumThreadsInNetwork);
        } else {
            myNetwork = mNetwork;
        }
        myNetwork->ReduceCircuit();
        myNetwork->OutputCircuitToTreewidthGraph(toPrintTo);
        return myNetwork;
    }



//This function takes in a file path to print to, creates a network, removes all the 1 qubit gates and any
//successive 2 qubit gates before printing the graph to a file that can be viewed using online graphviz to view the circuit
//The function returns a network pointer
    std::shared_ptr<Network> ContractionTools::ReduceAndPrintCircuitToVisualGraph(const std::string &toPrintTo) const {
        std::shared_ptr<Network> myNetwork;
        if (!mCopyCreated) {
            myNetwork = std::make_shared<Network>(mString, mMeasureFile);
            myNetwork->SetNumThreads(mNumThreadsInNetwork);
        } else {
            myNetwork = mNetwork;
        }
        myNetwork->ReduceCircuit();
        myNetwork->OutputCircuitToVisualGraph(toPrintTo);
        return myNetwork;
    }

    //calls quickbb to calculate the treewidth of a circuit
    const int ContractionTools::CalculateTreewidth(const int qbbseconds, const bool sixtyFourBitOpSystem) const {
        std::shared_ptr<Network> myNetwork;
        if (!mCopyCreated) {
            myNetwork = std::make_shared<Network>(mString, mMeasureFile);
            myNetwork->SetNumThreads(mNumThreadsInNetwork);
        } else {
            myNetwork = mNetwork;
        }
        LineGraph lg(myNetwork);
        Timer t;
        t.start();
        lg.runQuickBB(qbbseconds, &t, sixtyFourBitOpSystem);
        std::cout << "Please check output/qbb.out for more treewidth and quickbb stats" << std::endl;
        std::ifstream input("output/qbb.out");
        std::string line;

        std::getline(input, line);

        if (line.find(" The treewidth of the graph in the file ") != std::string::npos) {
            line = line.substr(line.find(" The treewidth of the graph in the file ") + 40);
            std::stringstream ss(line);
            std::getline(ss, line, ' ');
            std::getline(ss, line, ' ');
            int retVal;
            ss >> retVal;
            return retVal;
        }
        return -1;
    }


//This function takes in a network pointer, takes all the uncontracted nodes in the network
//and divides them into numPartitions partitions, storing the partitions in mPartitionedNodes
    void ContractionTools::CreateChunksOfNodes(std::shared_ptr<Network> &myNetwork) {
        auto iterator = myNetwork->GetUncontractedNodes().begin();
        int numPartitions = 2; //change this number to change the number of partitions to divide the network into
        //after some analysis, I determined that 2 was the optimal number for parallel threading, but feel free to play around
        mPartitionedNodes.reserve(numPartitions + 1);
        int size = static_cast<int>(myNetwork->GetUncontractedNodes().size());
        int skip = size / numPartitions;
        int counter = 0;
        while (iterator != myNetwork->GetUncontractedNodes().end()) {
            std::vector<std::shared_ptr<Node>> tempPartition;
            std::copy_n(iterator, skip, std::back_inserter(tempPartition));
            std::advance(iterator, skip);
            mPartitionedNodes.push_back(std::make_shared<std::vector<std::shared_ptr<Node>>>(tempPartition));
            counter += skip;
            if ((size - counter) < skip) {
                skip = size - counter;
            }
        }
    }

}

