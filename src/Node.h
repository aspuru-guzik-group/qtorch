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

/* READ ME
 * Class: Node
 *
 * This is a node class that holds tensor values for each vertex in the tensor network
 * The class is a hierarchy where certain types of gates inherit from the general node class
 * The possible gates are listed in the gate type enum and are initialized within their constructors
 * All of the initial values for each gate have been precalculated and are the respective tensor superoperators
 *
 * The node class contains the following functions:
 * Access - this function allows the user to access a certain index in the tensor ijkl... and uses a simple algorithm
 * to map the index to a dense vector. Note that the Access function is READ-ONLY. You cannot modify the value in the tensor
 * using the access function
 *
 * Index - this function is the exact same as Access except it's not read only. The value in the tensor is returned
 * by reference and can be modified
 *
 * The other functions are getter and setter functions and are pretty self-explanatory
 * Additionally, the arbitrary one qubit and two qubit node classes have functions that parse matrix values from an input file
 * the input file should have the matrix values for the basic operator (not superoperator) separated by spaces
 *
 */

#define PI 3.14159265358979323846

#include <algorithm>
#include <vector>
#include <complex>
#include <map>
#include <numeric>
#include <random>

/* READ ME
 * Class: Node
 *
 * This is a node class that holds tensor values for each vertex in the tensor network
 * The class is a hierarchy where certain types of gates inherit from the general node class
 * The possible gates are listed in the gate type enum and are initialized within their constructors
 * All of the initial values for each gate have been precalculated and are the respective tensor superoperators
 *
 * The node class contains the following functions:
 * Access - this function allows the user to access a certain index in the tensor ijkl... and uses a simple algorithm
 * to map the index to a dense vector. Note that the Access function is READ-ONLY. You cannot modify the value in the tensor
 * using the access function
 *
 * Index - this function is the exact same as Access except it's not read only. The value in the tensor is returned
 * by reference and can be modified
 *
 * The other functions are getter and setter functions and are pretty self-explanatory
 * Additionally, the arbitrary one qubit and two qubit node classes have functions that parse matrix values from an input file
 * the input file should have the matrix values for the basic operator (not superoperator) separated by spaces
 *
 */
#include <unordered_map>
#include "Wire.h"
#include <iostream>
#include <thread>
#include <fstream>
#include "Exceptions.h"


namespace qtorch {

    enum class GateType {
        CNOT,
        SWAP,
        HADAMARD,
        RX,
        RY,
        RZ,
        X,
        Y,
        Z,
        PHASE,
        DEPOLARIZER,
        CRK,
        CZ,
        CPHASE,
        INITSTATE,
        MEASURETRACE,
        INTERMEDIATESTATE,
        ARBITRARYONEQUBITUNITARY,
        ARBITRARYTWOQUBITUNITARY
    };


    class Node {
    public:
        int mRank;

        explicit Node(int rank0) : mRank(rank0), mSelectedInCostContractionAlgorithm(false),
                                   mContracted(false) { mVals.resize(pow(4, rank0), 0.0); };

        inline const std::complex<double> &Access(const std::vector<int> &indexVect);

        inline const std::complex<double> &Access(const long long &index);

        inline std::complex<double> &Index(const std::vector<int> &indexVect);

        inline std::complex<double> &Index(const long long &index);

        inline const std::vector<int> &GetWireNumber() const { return mWireNumbers; };

        inline void AddWireNumber(const int toSet) { mWireNumbers.push_back(toSet); };

        inline void SetWireNumber(const int index, const int toSet) { mWireNumbers[index] = toSet; };

        inline const GateType GetTypeOfNode() const { return mType; };

        inline void SetTypeOfNode(GateType newType) { mType = newType; };

        inline void SetTypeOfNodeString(const std::string &newType) { mStringType = std::move(newType); };

        inline const std::string &GetTypeOfNodeString() const { return mStringType; };

        inline void ClearNodeData() { mVals = std::vector<std::complex<double>>(); };
        int mID;
        int mIndexOfPreviousNode;
        bool mContracted;
        std::pair<int, int> mCreatedFrom;
        bool mSelectedInCostContractionAlgorithm;

        std::vector<std::shared_ptr<Wire>> &GetWires() {
            if (mWires.size() > mRank) {
                std::cout << "Node rank is:" << mRank << " and num wires is:" << mWires.size() << std::endl;
            }
            return mWires;
        };

        std::vector<std::complex<double>> &GetTensorVals() { return mVals; };

        virtual ~Node() = default;
        Node(Node&&) = default;
        Node(const Node&) = default;
        Node& operator=(const Node&) = default;
        Node& operator=(Node&&) = default;

    private:
        std::vector<std::complex<double>> mVals;
        std::vector<std::shared_ptr<Wire>> mWires;
        std::vector<int> mWireNumbers;
    protected:
        GateType mType{GateType::INTERMEDIATESTATE};
        std::string mStringType{"INTERMEDIATESTATE"};
    };

    inline const std::complex<double> &Node::Access(const std::vector<int> &indexVect) {
        unsigned long long sum(0);
        unsigned long long multiplier(0);
        std::for_each(indexVect.begin(), indexVect.end(), [&sum, &multiplier](int t) {
            sum += t << (2 * multiplier);
            ++multiplier;
        });
        return mVals.at(sum);
    }

    inline std::complex<double> &Node::Index(const std::vector<int> &indexVect) {
        unsigned long long sum(0);
        unsigned long long multiplier(1);
        std::for_each(indexVect.begin(), indexVect.end(), [&sum, &multiplier](int t) {
            sum += static_cast<unsigned long>(t) * multiplier;
            multiplier *= 4;
        });
        return mVals[sum];
    }

    inline std::complex<double> &Node::Index(const long long &index) {
        return mVals[index];
    }

    inline const std::complex<double> &Node::Access(const long long &index) {
        return mVals.at(index);
    }


    class CNOTNode : public Node {
    public:
        CNOTNode() : Node(4) {
            Index({0, 0, 0, 0}) = 1;
            Index({0, 1, 0, 1}) = 1;
            Index({0, 2, 0, 2}) = 1;
            Index({0, 3, 0, 3}) = 1;
            Index({1, 0, 1, 1}) = 1;
            Index({1, 1, 1, 0}) = 1;
            Index({1, 2, 1, 3}) = 1;
            Index({1, 3, 1, 2}) = 1;
            Index({2, 0, 2, 2}) = 1;
            Index({2, 1, 2, 3}) = 1;
            Index({2, 2, 2, 0}) = 1;
            Index({2, 3, 2, 1}) = 1;
            Index({3, 0, 3, 3}) = 1;
            Index({3, 1, 3, 2}) = 1;
            Index({3, 2, 3, 1}) = 1;
            Index({3, 3, 3, 0}) = 1;
            mType = GateType::CNOT;
            mStringType = ("CNOT");
        };

    };

    class RxNode : public Node {
    public:
        RxNode(const double tempPhaseVal) : Node(2) {
            Index({0, 0}) = pow(cos(tempPhaseVal / 2.0), 2);
            Index({0, 1}) = std::complex<double>(0.0, (sin(tempPhaseVal) / 2.0));
            Index({0, 2}) = std::complex<double>(0.0, -1.0 * (sin(tempPhaseVal) / 2.0));
            Index({0, 3}) = pow(sin(tempPhaseVal / 2.0), 2);
            Index({1, 0}) = std::complex<double>(0.0, (sin(tempPhaseVal) / 2.0));
            Index({1, 1}) = pow(cos(tempPhaseVal / 2.0), 2);
            Index({1, 2}) = pow(sin(tempPhaseVal / 2.0), 2);
            Index({1, 3}) = std::complex<double>(0.0, -1.0 * (sin(tempPhaseVal) / 2.0));
            Index({2, 0}) = std::complex<double>(0.0, -1.0 * (sin(tempPhaseVal) / 2.0));
            Index({2, 1}) = pow(sin(tempPhaseVal / 2.0), 2);
            Index({2, 2}) = pow(cos(tempPhaseVal / 2.0), 2);
            Index({2, 3}) = std::complex<double>(0.0, (sin(tempPhaseVal) / 2.0));
            Index({3, 0}) = pow(sin(tempPhaseVal / 2.0), 2);
            Index({3, 1}) = std::complex<double>(0.0, -1.0 * (sin(tempPhaseVal) / 2.0));
            Index({3, 2}) = std::complex<double>(0.0, (sin(tempPhaseVal) / 2.0));
            Index({3, 3}) = pow(cos(tempPhaseVal / 2.0), 2);
            mType = GateType::RX;
            mStringType = ("Rx");
        };
    };

    class RyNode : public Node {
    public:
        RyNode(const double tempPhaseVal) : Node(2) {
            double debug = sin(tempPhaseVal) / 2.0;
            Index({0, 0}) = pow(cos(tempPhaseVal / 2.0), 2);
            Index({0, 1}) = sin(tempPhaseVal) / 2.0;
            Index({0, 2}) = sin(tempPhaseVal) / 2.0;
            Index({0, 3}) = pow(sin(tempPhaseVal / 2.0), 2);
            Index({1, 0}) = -1.0 * sin(tempPhaseVal) / 2.0;
            Index({1, 1}) = pow(cos(tempPhaseVal / 2.0), 2);
            Index({1, 2}) = -1.0 * pow(sin(tempPhaseVal / 2), 2);
            Index({1, 3}) = sin(tempPhaseVal) / 2.0;
            Index({2, 0}) = -1.0 * sin(tempPhaseVal) / 2.0;
            Index({2, 1}) = -1.0 * pow(sin(tempPhaseVal / 2.0), 2);
            Index({2, 2}) = pow(cos(tempPhaseVal / 2.0), 2);
            Index({2, 3}) = sin(tempPhaseVal) / 2.0;
            Index({3, 0}) = pow(sin(tempPhaseVal / 2.0), 2);
            Index({3, 1}) = -1.0 * sin(tempPhaseVal) / 2.0;
            Index({3, 2}) = -1.0 * sin(tempPhaseVal) / 2.0;
            Index({3, 3}) = pow(cos(tempPhaseVal / 2.0), 2);
            mType = GateType::RY;
            mStringType = ("Ry");
        };
    };

    class RzNode : public Node {
    public:
        RzNode(const double tempPhaseVal) : Node(2) {
            Index({0, 0}) = 1.0;
            Index({1, 1}) = std::complex<double>(cos(tempPhaseVal), -1.0 * sin(tempPhaseVal));
            Index({2, 2}) = std::complex<double>(cos(tempPhaseVal), sin(tempPhaseVal));
            Index({3, 3}) = 1.0;
            mType = GateType::RZ;
            mStringType = ("Rz");
        };
    };
    
    
    class PhaseNode : public Node {
    public:
        PhaseNode(const double tempPhaseVal) : Node(2) {
            Index({0, 0}) = 1.0;
            Index({1, 1}) = std::complex<double>(cos(tempPhaseVal), -1.0 * sin(tempPhaseVal));
            Index({2, 2}) = std::complex<double>(cos(tempPhaseVal), sin(tempPhaseVal));
            Index({3, 3}) = 1.0;
            mType = GateType::PHASE;
            mStringType = ("Phase");
        };
    };

    class HNode : public Node {
    public:
        HNode() : Node(2) {
            Index({0, 0}) = 1.0 / 2.0;
            Index({0, 1}) = 1.0 / 2.0;
            Index({0, 2}) = 1.0 / 2.0;
            Index({0, 3}) = 1.0 / 2.0;
            Index({1, 0}) = 1.0 / 2.0;
            Index({1, 1}) = -1.0 / 2.0;
            Index({1, 2}) = 1.0 / 2.0;
            Index({1, 3}) = -1.0 / 2.0;
            Index({2, 0}) = 1.0 / 2.0;
            Index({2, 1}) = 1.0 / 2.0;
            Index({2, 2}) = -1.0 / 2.0;
            Index({2, 3}) = -1.0 / 2.0;
            Index({3, 0}) = 1.0 / 2.0;
            Index({3, 1}) = -1.0 / 2.0;
            Index({3, 2}) = -1.0 / 2.0;
            Index({3, 3}) = 1.0 / 2.0;
            mType = GateType::HADAMARD;
            mStringType = ("H");
        };
    };

    class XNode : public Node {
    public:
        XNode() : Node(2) {
            Index({0, 3}) = 1;
            Index({1, 2}) = 1;
            Index({2, 1}) = 1;
            Index({3, 0}) = 1;
            mType = GateType::X;
            mStringType = ("X");
        };
    };

    class YNode : public Node {
    public:
        YNode() : Node(2) {
            Index({0, 3}) = 1;
            Index({1, 2}) = -1;
            Index({2, 1}) = -1;
            Index({3, 0}) = 1;
            mType = GateType::Y;
            mStringType = ("Y");
        };
    };

    class ZNode : public Node {
    public:
        ZNode() : Node(2) {
            Index({0, 0}) = 1.0;
            Index({1, 1}) = -1.0;
            Index({2, 2}) = -1.0;
            Index({3, 3}) = 1.0;
            mType = GateType::Z;
            mStringType = ("Z");
        };
    };

    class ZeroStateNode : public Node {
    public:
        ZeroStateNode() : Node(1) {
            Index({0}) = 1;
            mType = GateType::INITSTATE;
            mStringType = ("|0><0|");
        };
    };

    class TraceNode : public Node {
    public:
        TraceNode() : Node(1) {
            Index({0}) = 1.0;
            Index({3}) = 1.0;
            mType = GateType::MEASURETRACE;
            mStringType = ("Trace");
        };
    };

    class XMeasure : public Node {
    public:
        XMeasure() : Node(1) {
            Index({1}) = 1.0;
            Index({2}) = 1.0;
            mType = GateType::MEASURETRACE;
            mStringType = ("X measure");
        };
    };

    class YMeasure : public Node {
    public:
        YMeasure() : Node(1) {
            Index({1}) = std::complex<double>(0, 1.0);
            Index({2}) = std::complex<double>(0, -1.0);
            mType = GateType::MEASURETRACE;
            mStringType = ("Y measure");
        };
    };

    class ZMeasure : public Node {
    public:
        ZMeasure() : Node(1) {
            Index({0}) = 1.0;
            Index({3}) = -1.0;
            mType = GateType::MEASURETRACE;
            mStringType = ("Z measure");
        };
    };

    class ProjectOne : public Node {
    public:
        ProjectOne() : Node(1) {
            Index({3}) = 1;
            mType = GateType::MEASURETRACE;
            mStringType = ("|1><1| measure");
        };
    };

    class ProjectZero : public Node {
    public:
        ProjectZero() : Node(1) {
            Index({0}) = 1;
            mType = GateType::MEASURETRACE;
            mStringType = ("|0><0| measure");
        };
    };


    class CRkNode : public Node {
    public:
        CRkNode(int controlBit) : Node(4) {
            Index({0, 0, 0, 0}) = 1.0;
            Index({0, 1, 0, 1}) = 1.0;
            Index({0, 2, 0, 2}) = 1.0;
            Index({0, 3, 0, 3}) = 1.0;
            Index({1, 0, 1, 0}) = 1.0;
            Index({1, 1, 1, 1}) = exp(2.0 * PI * std::complex<double>(0, -1) / pow(2, controlBit + 1.0));
            Index({1, 2, 1, 2}) = 1.0;
            Index({1, 3, 1, 3}) = exp(2.0 * PI * std::complex<double>(0, -1) / pow(2, controlBit + 1.0));
            Index({2, 0, 2, 0}) = 1.0;
            Index({2, 1, 2, 1}) = 1.0;
            Index({2, 2, 2, 2}) = exp(2.0 * PI * std::complex<double>(0, 1) / pow(2, controlBit + 1.0));
            Index({2, 3, 2, 3}) = exp(2.0 * PI * std::complex<double>(0, 1) / pow(2, controlBit + 1.0));
            Index({3, 0, 3, 0}) = 1.0;
            Index({3, 1, 3, 1}) = exp(2.0 * PI * std::complex<double>(0, -1) / pow(2, controlBit + 1.0));
            Index({3, 2, 3, 2}) = exp(2.0 * PI * std::complex<double>(0, 1) / pow(2, controlBit + 1.0));
            Index({3, 3, 3, 3}) = 1.0;
            mType = GateType::CRK;
            mStringType = ("CRk");
        }

    };

    class CZNode : public Node {
    public:
        CZNode() : Node(4) {
            Index({0, 0, 0, 0}) = 1.0;
            Index({0, 1, 0, 1}) = 1.0;
            Index({0, 2, 0, 2}) = 1.0;
            Index({0, 3, 0, 3}) = 1.0;
            Index({1, 0, 1, 0}) = 1.0;
            Index({1, 1, 1, 1}) = -1.0;
            Index({1, 2, 1, 2}) = 1.0;
            Index({1, 3, 1, 3}) = -1.0;
            Index({2, 0, 2, 0}) = 1.0;
            Index({2, 1, 2, 1}) = 1.0;
            Index({2, 2, 2, 2}) = -1.0;
            Index({2, 3, 2, 3}) = -1.0;
            Index({3, 0, 3, 0}) = 1.0;
            Index({3, 1, 3, 1}) = -1.0;
            Index({3, 2, 3, 2}) = -1.0;
            Index({3, 3, 3, 3}) = 1.0;
            mType = GateType::CZ;
            mStringType = ("CZ");
        }


    };

    class CPhaseNode : public Node {
    public:
        CPhaseNode(double tempPhaseVal) : Node(4) {
            Index({0, 0, 0, 0}) = 1.0;
            Index({0, 1, 0, 1}) = 1.0;
            Index({0, 2, 0, 2}) = 1.0;
            Index({0, 3, 0, 3}) = 1.0;
            Index({1, 0, 1, 0}) = 1.0;
            Index({1, 1, 1, 1}) = std::complex<double>(cos(tempPhaseVal), -1.0 * sin(tempPhaseVal));
            Index({1, 2, 1, 2}) = 1.0;
            Index({1, 3, 1, 3}) = std::complex<double>(cos(tempPhaseVal), -1.0 * sin(tempPhaseVal));
            Index({2, 0, 2, 0}) = 1.0;
            Index({2, 1, 2, 1}) = 1.0;
            Index({2, 2, 2, 2}) = std::complex<double>(cos(tempPhaseVal), sin(tempPhaseVal));
            Index({2, 3, 2, 3}) = std::complex<double>(cos(tempPhaseVal), sin(tempPhaseVal));
            Index({3, 0, 3, 0}) = 1.0;
            Index({3, 1, 3, 1}) = std::complex<double>(cos(tempPhaseVal), -1.0 * sin(tempPhaseVal));
            Index({3, 2, 3, 2}) = std::complex<double>(cos(tempPhaseVal), sin(tempPhaseVal));
            Index({3, 3, 3, 3}) = 1.0;
            mStringType = "CPhase";
            mType = GateType::CPHASE;
        }

    };

    class DepolarizingChannelNode : public Node {
    public:
        DepolarizingChannelNode(std::mt19937 &gen, std::uniform_real_distribution<float> &randDist) : Node(2) {
            float probability = randDist(gen);
            Index({0, 0}) = 1.0 - (2.0 * probability / 3.0);
            Index({1, 1}) = 1.0 - (4.0 * probability / 3.0);
            Index({1, 2}) = (2.0 * probability / 3.0);
            Index({2, 1}) = (2.0 * probability / 3.0);
            Index({2, 2}) = 1.0 - (4.0 * probability / 3.0);
            Index({3, 3}) = 1.0 - (2.0 * probability / 3.0);
            mType = GateType::DEPOLARIZER;
            mStringType = ("Depolarizer");
        }
    };

    class SwapNode : public Node {
    public:
        SwapNode() : Node(4) {
            mType = GateType::SWAP;
            mStringType = ("SWAP");

            Index({0, 0, 0, 0}) = 1.0;
            Index({0, 1, 1, 0}) = 1.0;
            Index({1, 0, 0, 1}) = 1.0;
            Index({1, 1, 1, 1}) = 1.0;

            Index({0, 2, 2, 0}) = 1.0;
            Index({0, 3, 3, 0}) = 1.0;
            Index({1, 2, 2, 1}) = 1.0;
            Index({1, 3, 3, 1}) = 1.0;

            Index({2, 0, 0, 2}) = 1.0;
            Index({2, 1, 1, 2}) = 1.0;
            Index({3, 0, 0, 3}) = 1.0;
            Index({3, 1, 1, 3}) = 1.0;

            Index({2, 2, 2, 2}) = 1.0;
            Index({2, 3, 3, 2}) = 1.0;
            Index({3, 2, 2, 3}) = 1.0;
            Index({3, 3, 3, 3}) = 1.0;
        }

    };

    class ArbitraryOneQubitNode : public Node {
    public:
        ArbitraryOneQubitNode(const std::string &inputFile, const std::string &nodeName) : Node(2) {
            mType = GateType::ARBITRARYONEQUBITUNITARY;
            mStringType = (nodeName);
            ParseMatrixValues(inputFile);
        }

    private:
        void ParseMatrixValues(const std::string &filename);

    };

    void ArbitraryOneQubitNode::ParseMatrixValues(const std::string &filename) {
        std::ifstream input(filename);
        if (!input.is_open()) {
            std::cout << "Failed To Open Arbitrary Matrix File" << std::endl;
            throw InvalidFile();
        }
        std::vector<std::complex<double>> nums(4);
        for (int i(0); i < 4; ++i) {
            if (input.eof()) {
                throw InvalidFileFormat();
            }
            input >> nums[i];
        }
        Index({0, 0}) = nums[0] * std::conj(nums[0]);
        Index({0, 1}) = nums[0] * std::conj(nums[2]);
        Index({0, 2}) = nums[2] * std::conj(nums[0]);
        Index({0, 3}) = nums[2] * std::conj(nums[2]);
        Index({1, 0}) = nums[0] * std::conj(nums[1]);
        Index({1, 1}) = nums[0] * std::conj(nums[3]);
        Index({1, 2}) = nums[2] * std::conj(nums[1]);
        Index({1, 3}) = nums[2] * std::conj(nums[3]);
        Index({2, 0}) = nums[1] * std::conj(nums[0]);
        Index({2, 1}) = nums[1] * std::conj(nums[2]);
        Index({2, 2}) = nums[3] * std::conj(nums[0]);
        Index({2, 3}) = nums[3] * std::conj(nums[2]);
        Index({3, 0}) = nums[1] * std::conj(nums[1]);
        Index({3, 1}) = nums[1] * std::conj(nums[3]);
        Index({3, 2}) = nums[3] * std::conj(nums[1]);
        Index({3, 3}) = nums[3] * std::conj(nums[3]);
    }

    class ArbitraryTwoQubitNode : public Node {
    public:
        ArbitraryTwoQubitNode(const std::string &inputFile, const std::string &nodeName) : Node(4) {
            mType = GateType::ARBITRARYTWOQUBITUNITARY;
            mStringType = (nodeName);
            ParseMatrixValues(inputFile);
        }

    private:
        void ParseMatrixValues(const std::string &filename);

    };

    void ArbitraryTwoQubitNode::ParseMatrixValues(const std::string &filename) {
        std::ifstream input(filename);
        if (!input.is_open()) {
            std::cout << "Failed To Open Arbitrary Matrix File" << std::endl;
            throw InvalidFile();
        }
        std::vector<std::complex<double>> nums(16);
        for (int i(0); i < 16; ++i) {
            if (input.eof()) {
                throw InvalidFileFormat();
            }
            input >> nums[i];
        }
        //when reading in from file, complex numbers must be in the format: (a,b) //(where the number is a + bi)


        /*std::cout<<"\n\n\n";
        std::cout<<"=============NODE DEBUG============"<<std::endl;
        for(auto& num:nums)
        {
            std::cout<<num<<" "<<std::endl;
        }
        std::cout<<"\n\n\n";*/


        Index({0, 0, 0, 0}) = nums[0] * std::conj(nums[0]);
        Index({0, 0, 0, 1}) = nums[0] * std::conj(nums[4]);
        Index({0, 0, 0, 2}) = nums[4] * std::conj(nums[0]);
        Index({0, 0, 0, 3}) = nums[4] * std::conj(nums[4]);
        Index({0, 0, 1, 0}) = nums[0] * std::conj(nums[8]);
        Index({0, 0, 1, 1}) = nums[0] * std::conj(nums[12]);
        Index({0, 0, 1, 2}) = nums[4] * std::conj(nums[8]);
        Index({0, 0, 1, 3}) = nums[4] * std::conj(nums[12]);
        Index({0, 0, 2, 0}) = nums[8] * std::conj(nums[0]);
        Index({0, 0, 2, 1}) = nums[8] * std::conj(nums[4]);
        Index({0, 0, 2, 2}) = nums[12] * std::conj(nums[0]);
        Index({0, 0, 2, 3}) = nums[12] * std::conj(nums[4]);
        Index({0, 0, 3, 0}) = nums[8] * std::conj(nums[8]);
        Index({0, 0, 3, 1}) = nums[8] * std::conj(nums[12]);
        Index({0, 0, 3, 2}) = nums[12] * std::conj(nums[8]);
        Index({0, 0, 3, 3}) = nums[12] * std::conj(nums[12]);

        Index({0, 1, 0, 0}) = nums[0] * std::conj(nums[1]);
        Index({0, 1, 0, 1}) = nums[0] * std::conj(nums[5]);
        Index({0, 1, 0, 2}) = nums[4] * std::conj(nums[1]);
        Index({0, 1, 0, 3}) = nums[4] * std::conj(nums[5]);
        Index({0, 1, 1, 0}) = nums[0] * std::conj(nums[9]);
        Index({0, 1, 1, 1}) = nums[0] * std::conj(nums[13]);
        Index({0, 1, 1, 2}) = nums[4] * std::conj(nums[9]);
        Index({0, 1, 1, 3}) = nums[4] * std::conj(nums[13]);
        Index({0, 1, 2, 0}) = nums[8] * std::conj(nums[1]);
        Index({0, 1, 2, 1}) = nums[8] * std::conj(nums[5]);
        Index({0, 1, 2, 2}) = nums[12] * std::conj(nums[1]);
        Index({0, 1, 2, 3}) = nums[12] * std::conj(nums[5]);
        Index({0, 1, 3, 0}) = nums[8] * std::conj(nums[9]);
        Index({0, 1, 3, 1}) = nums[8] * std::conj(nums[13]);
        Index({0, 1, 3, 2}) = nums[12] * std::conj(nums[9]);
        Index({0, 1, 3, 3}) = nums[12] * std::conj(nums[13]);

        Index({1, 0, 0, 0}) = nums[0] * std::conj(nums[2]);
        Index({1, 0, 0, 1}) = nums[0] * std::conj(nums[6]);
        Index({1, 0, 0, 2}) = nums[4] * std::conj(nums[2]);
        Index({1, 0, 0, 3}) = nums[4] * std::conj(nums[6]);
        Index({1, 0, 1, 0}) = nums[0] * std::conj(nums[10]);
        Index({1, 0, 1, 1}) = nums[0] * std::conj(nums[14]);
        Index({1, 0, 1, 2}) = nums[4] * std::conj(nums[10]);
        Index({1, 0, 1, 3}) = nums[4] * std::conj(nums[14]);
        Index({1, 0, 2, 0}) = nums[8] * std::conj(nums[2]);
        Index({1, 0, 2, 1}) = nums[8] * std::conj(nums[6]);
        Index({1, 0, 2, 2}) = nums[12] * std::conj(nums[2]);
        Index({1, 0, 2, 3}) = nums[12] * std::conj(nums[6]);
        Index({1, 0, 3, 0}) = nums[8] * std::conj(nums[10]);
        Index({1, 0, 3, 1}) = nums[8] * std::conj(nums[14]);
        Index({1, 0, 3, 2}) = nums[12] * std::conj(nums[10]);
        Index({1, 0, 3, 3}) = nums[12] * std::conj(nums[14]);

        Index({1, 1, 0, 0}) = nums[0] * std::conj(nums[3]);
        Index({1, 1, 0, 1}) = nums[0] * std::conj(nums[7]);
        Index({1, 1, 0, 2}) = nums[4] * std::conj(nums[3]);
        Index({1, 1, 0, 3}) = nums[4] * std::conj(nums[7]);
        Index({1, 1, 1, 0}) = nums[0] * std::conj(nums[11]);
        Index({1, 1, 1, 1}) = nums[0] * std::conj(nums[15]);
        Index({1, 1, 1, 2}) = nums[4] * std::conj(nums[11]);
        Index({1, 1, 1, 3}) = nums[4] * std::conj(nums[15]);
        Index({1, 1, 2, 0}) = nums[8] * std::conj(nums[3]);
        Index({1, 1, 2, 1}) = nums[8] * std::conj(nums[7]);
        Index({1, 1, 2, 2}) = nums[12] * std::conj(nums[3]);
        Index({1, 1, 2, 3}) = nums[12] * std::conj(nums[7]);
        Index({1, 1, 3, 0}) = nums[8] * std::conj(nums[11]);
        Index({1, 1, 3, 1}) = nums[8] * std::conj(nums[15]);
        Index({1, 1, 3, 2}) = nums[12] * std::conj(nums[11]);
        Index({1, 1, 3, 3}) = nums[12] * std::conj(nums[15]);

        Index({0, 2, 0, 0}) = nums[1] * std::conj(nums[0]);
        Index({0, 2, 0, 1}) = nums[1] * std::conj(nums[4]);
        Index({0, 2, 0, 2}) = nums[5] * std::conj(nums[0]);
        Index({0, 2, 0, 3}) = nums[5] * std::conj(nums[4]);
        Index({0, 2, 1, 0}) = nums[1] * std::conj(nums[8]);
        Index({0, 2, 1, 1}) = nums[1] * std::conj(nums[12]);
        Index({0, 2, 1, 2}) = nums[5] * std::conj(nums[8]);
        Index({0, 2, 1, 3}) = nums[5] * std::conj(nums[12]);
        Index({0, 2, 2, 0}) = nums[9] * std::conj(nums[0]);
        Index({0, 2, 2, 1}) = nums[9] * std::conj(nums[4]);
        Index({0, 2, 2, 2}) = nums[13] * std::conj(nums[0]);
        Index({0, 2, 2, 3}) = nums[13] * std::conj(nums[4]);
        Index({0, 2, 3, 0}) = nums[9] * std::conj(nums[8]);
        Index({0, 2, 3, 1}) = nums[9] * std::conj(nums[12]);
        Index({0, 2, 3, 2}) = nums[13] * std::conj(nums[8]);
        Index({0, 2, 3, 3}) = nums[13] * std::conj(nums[12]);

        Index({0, 3, 0, 0}) = nums[1] * std::conj(nums[1]);
        Index({0, 3, 0, 1}) = nums[1] * std::conj(nums[5]);
        Index({0, 3, 0, 2}) = nums[5] * std::conj(nums[1]);
        Index({0, 3, 0, 3}) = nums[5] * std::conj(nums[5]);
        Index({0, 3, 1, 0}) = nums[1] * std::conj(nums[9]);
        Index({0, 3, 1, 1}) = nums[1] * std::conj(nums[13]);
        Index({0, 3, 1, 2}) = nums[5] * std::conj(nums[9]);
        Index({0, 3, 1, 3}) = nums[5] * std::conj(nums[13]);
        Index({0, 3, 2, 0}) = nums[9] * std::conj(nums[1]);
        Index({0, 3, 2, 1}) = nums[9] * std::conj(nums[5]);
        Index({0, 3, 2, 2}) = nums[13] * std::conj(nums[1]);
        Index({0, 3, 2, 3}) = nums[13] * std::conj(nums[5]);
        Index({0, 3, 3, 0}) = nums[9] * std::conj(nums[9]);
        Index({0, 3, 3, 1}) = nums[9] * std::conj(nums[13]);
        Index({0, 3, 3, 2}) = nums[13] * std::conj(nums[9]);
        Index({0, 3, 3, 3}) = nums[13] * std::conj(nums[13]);

        Index({1, 2, 0, 0}) = nums[1] * std::conj(nums[2]);
        Index({1, 2, 0, 1}) = nums[1] * std::conj(nums[6]);
        Index({1, 2, 0, 2}) = nums[5] * std::conj(nums[2]);
        Index({1, 2, 0, 3}) = nums[5] * std::conj(nums[6]);
        Index({1, 2, 1, 0}) = nums[1] * std::conj(nums[10]);
        Index({1, 2, 1, 1}) = nums[1] * std::conj(nums[14]);
        Index({1, 2, 1, 2}) = nums[5] * std::conj(nums[10]);
        Index({1, 2, 1, 3}) = nums[5] * std::conj(nums[14]);
        Index({1, 2, 2, 0}) = nums[9] * std::conj(nums[2]);
        Index({1, 2, 2, 1}) = nums[9] * std::conj(nums[6]);
        Index({1, 2, 2, 2}) = nums[13] * std::conj(nums[2]);
        Index({1, 2, 2, 3}) = nums[13] * std::conj(nums[6]);
        Index({1, 2, 3, 0}) = nums[9] * std::conj(nums[10]);
        Index({1, 2, 3, 1}) = nums[9] * std::conj(nums[14]);
        Index({1, 2, 3, 2}) = nums[13] * std::conj(nums[10]);
        Index({1, 2, 3, 3}) = nums[13] * std::conj(nums[14]);

        Index({1, 3, 0, 0}) = nums[1] * std::conj(nums[3]);
        Index({1, 3, 0, 1}) = nums[1] * std::conj(nums[7]);
        Index({1, 3, 0, 2}) = nums[5] * std::conj(nums[3]);
        Index({1, 3, 0, 3}) = nums[5] * std::conj(nums[7]);
        Index({1, 3, 1, 0}) = nums[1] * std::conj(nums[11]);
        Index({1, 3, 1, 1}) = nums[1] * std::conj(nums[15]);
        Index({1, 3, 1, 2}) = nums[5] * std::conj(nums[11]);
        Index({1, 3, 1, 3}) = nums[5] * std::conj(nums[15]);
        Index({1, 3, 2, 0}) = nums[9] * std::conj(nums[3]);
        Index({1, 3, 2, 1}) = nums[9] * std::conj(nums[7]);
        Index({1, 3, 2, 2}) = nums[13] * std::conj(nums[3]);
        Index({1, 3, 2, 3}) = nums[13] * std::conj(nums[7]);
        Index({1, 3, 3, 0}) = nums[9] * std::conj(nums[11]);
        Index({1, 3, 3, 1}) = nums[9] * std::conj(nums[15]);
        Index({1, 3, 3, 2}) = nums[13] * std::conj(nums[11]);
        Index({1, 3, 3, 3}) = nums[13] * std::conj(nums[15]);

        Index({2, 0, 0, 0}) = nums[2] * std::conj(nums[0]);
        Index({2, 0, 0, 1}) = nums[2] * std::conj(nums[4]);
        Index({2, 0, 0, 2}) = nums[6] * std::conj(nums[0]);
        Index({2, 0, 0, 3}) = nums[6] * std::conj(nums[4]);
        Index({2, 0, 1, 0}) = nums[2] * std::conj(nums[8]);
        Index({2, 0, 1, 1}) = nums[2] * std::conj(nums[12]);
        Index({2, 0, 1, 2}) = nums[6] * std::conj(nums[8]);
        Index({2, 0, 1, 3}) = nums[6] * std::conj(nums[12]);
        Index({2, 0, 2, 0}) = nums[10] * std::conj(nums[0]);
        Index({2, 0, 2, 1}) = nums[10] * std::conj(nums[4]);
        Index({2, 0, 2, 2}) = nums[14] * std::conj(nums[0]);
        Index({2, 0, 2, 3}) = nums[14] * std::conj(nums[4]);
        Index({2, 0, 3, 0}) = nums[10] * std::conj(nums[8]);
        Index({2, 0, 3, 1}) = nums[10] * std::conj(nums[12]);
        Index({2, 0, 3, 2}) = nums[14] * std::conj(nums[8]);
        Index({2, 0, 3, 3}) = nums[14] * std::conj(nums[12]);

        Index({2, 1, 0, 0}) = nums[2] * std::conj(nums[1]);
        Index({2, 1, 0, 1}) = nums[2] * std::conj(nums[5]);
        Index({2, 1, 0, 2}) = nums[6] * std::conj(nums[1]);
        Index({2, 1, 0, 3}) = nums[6] * std::conj(nums[5]);
        Index({2, 1, 1, 0}) = nums[2] * std::conj(nums[9]);
        Index({2, 1, 1, 1}) = nums[2] * std::conj(nums[13]);
        Index({2, 1, 1, 2}) = nums[6] * std::conj(nums[9]);
        Index({2, 1, 1, 3}) = nums[6] * std::conj(nums[13]);
        Index({2, 1, 2, 0}) = nums[10] * std::conj(nums[1]);
        Index({2, 1, 2, 1}) = nums[10] * std::conj(nums[5]);
        Index({2, 1, 2, 2}) = nums[14] * std::conj(nums[1]);
        Index({2, 1, 2, 3}) = nums[14] * std::conj(nums[5]);
        Index({2, 1, 3, 0}) = nums[10] * std::conj(nums[9]);
        Index({2, 1, 3, 1}) = nums[10] * std::conj(nums[13]);
        Index({2, 1, 3, 2}) = nums[14] * std::conj(nums[9]);
        Index({2, 1, 3, 3}) = nums[14] * std::conj(nums[13]);

        Index({3, 0, 0, 0}) = nums[2] * std::conj(nums[2]);
        Index({3, 0, 0, 1}) = nums[2] * std::conj(nums[6]);
        Index({3, 0, 0, 2}) = nums[6] * std::conj(nums[2]);
        Index({3, 0, 0, 3}) = nums[6] * std::conj(nums[6]);
        Index({3, 0, 1, 0}) = nums[2] * std::conj(nums[10]);
        Index({3, 0, 1, 1}) = nums[2] * std::conj(nums[14]);
        Index({3, 0, 1, 2}) = nums[6] * std::conj(nums[10]);
        Index({3, 0, 1, 3}) = nums[6] * std::conj(nums[14]);
        Index({3, 0, 2, 0}) = nums[10] * std::conj(nums[2]);
        Index({3, 0, 2, 1}) = nums[10] * std::conj(nums[6]);
        Index({3, 0, 2, 2}) = nums[14] * std::conj(nums[2]);
        Index({3, 0, 2, 3}) = nums[14] * std::conj(nums[6]);
        Index({3, 0, 3, 0}) = nums[10] * std::conj(nums[10]);
        Index({3, 0, 3, 1}) = nums[10] * std::conj(nums[14]);
        Index({3, 0, 3, 2}) = nums[14] * std::conj(nums[10]);
        Index({3, 0, 3, 3}) = nums[14] * std::conj(nums[14]);

        Index({3, 1, 0, 0}) = nums[2] * std::conj(nums[3]);
        Index({3, 1, 0, 1}) = nums[2] * std::conj(nums[7]);
        Index({3, 1, 0, 2}) = nums[6] * std::conj(nums[3]);
        Index({3, 1, 0, 3}) = nums[6] * std::conj(nums[7]);
        Index({3, 1, 1, 0}) = nums[2] * std::conj(nums[11]);
        Index({3, 1, 1, 1}) = nums[2] * std::conj(nums[15]);
        Index({3, 1, 1, 2}) = nums[6] * std::conj(nums[11]);
        Index({3, 1, 1, 3}) = nums[6] * std::conj(nums[15]);
        Index({3, 1, 2, 0}) = nums[10] * std::conj(nums[3]);
        Index({3, 1, 2, 1}) = nums[10] * std::conj(nums[7]);
        Index({3, 1, 2, 2}) = nums[14] * std::conj(nums[3]);
        Index({3, 1, 2, 3}) = nums[14] * std::conj(nums[7]);
        Index({3, 1, 3, 0}) = nums[10] * std::conj(nums[11]);
        Index({3, 1, 3, 1}) = nums[10] * std::conj(nums[15]);
        Index({3, 1, 3, 2}) = nums[14] * std::conj(nums[11]);
        Index({3, 1, 3, 3}) = nums[14] * std::conj(nums[15]);

        Index({2, 2, 0, 0}) = nums[3] * std::conj(nums[0]);
        Index({2, 2, 0, 1}) = nums[3] * std::conj(nums[4]);
        Index({2, 2, 0, 2}) = nums[7] * std::conj(nums[0]);
        Index({2, 2, 0, 3}) = nums[7] * std::conj(nums[4]);
        Index({2, 2, 1, 0}) = nums[3] * std::conj(nums[8]);
        Index({2, 2, 1, 1}) = nums[3] * std::conj(nums[12]);
        Index({2, 2, 1, 2}) = nums[7] * std::conj(nums[8]);
        Index({2, 2, 1, 3}) = nums[7] * std::conj(nums[12]);
        Index({2, 2, 2, 0}) = nums[11] * std::conj(nums[0]);
        Index({2, 2, 2, 1}) = nums[11] * std::conj(nums[4]);
        Index({2, 2, 2, 2}) = nums[15] * std::conj(nums[0]);
        Index({2, 2, 2, 3}) = nums[15] * std::conj(nums[4]);
        Index({2, 2, 3, 0}) = nums[11] * std::conj(nums[8]);
        Index({2, 2, 3, 1}) = nums[11] * std::conj(nums[12]);
        Index({2, 2, 3, 2}) = nums[15] * std::conj(nums[8]);
        Index({2, 2, 3, 3}) = nums[15] * std::conj(nums[12]);

        Index({2, 3, 0, 0}) = nums[3] * std::conj(nums[1]);
        Index({2, 3, 0, 1}) = nums[3] * std::conj(nums[5]);
        Index({2, 3, 0, 2}) = nums[7] * std::conj(nums[1]);
        Index({2, 3, 0, 3}) = nums[7] * std::conj(nums[5]);
        Index({2, 3, 1, 0}) = nums[3] * std::conj(nums[9]);
        Index({2, 3, 1, 1}) = nums[3] * std::conj(nums[13]);
        Index({2, 3, 1, 2}) = nums[7] * std::conj(nums[9]);
        Index({2, 3, 1, 3}) = nums[7] * std::conj(nums[13]);
        Index({2, 3, 2, 0}) = nums[11] * std::conj(nums[1]);
        Index({2, 3, 2, 1}) = nums[11] * std::conj(nums[5]);
        Index({2, 3, 2, 2}) = nums[15] * std::conj(nums[1]);
        Index({2, 3, 2, 3}) = nums[15] * std::conj(nums[5]);
        Index({2, 3, 3, 0}) = nums[11] * std::conj(nums[9]);
        Index({2, 3, 3, 1}) = nums[11] * std::conj(nums[13]);
        Index({2, 3, 3, 2}) = nums[15] * std::conj(nums[9]);
        Index({2, 3, 3, 3}) = nums[15] * std::conj(nums[13]);

        Index({3, 2, 0, 0}) = nums[3] * std::conj(nums[2]);
        Index({3, 2, 0, 1}) = nums[3] * std::conj(nums[6]);
        Index({3, 2, 0, 2}) = nums[7] * std::conj(nums[2]);
        Index({3, 2, 0, 3}) = nums[7] * std::conj(nums[6]);
        Index({3, 2, 1, 0}) = nums[3] * std::conj(nums[10]);
        Index({3, 2, 1, 1}) = nums[3] * std::conj(nums[14]);
        Index({3, 2, 1, 2}) = nums[7] * std::conj(nums[10]);
        Index({3, 2, 1, 3}) = nums[7] * std::conj(nums[14]);
        Index({3, 2, 2, 0}) = nums[11] * std::conj(nums[2]);
        Index({3, 2, 2, 1}) = nums[11] * std::conj(nums[6]);
        Index({3, 2, 2, 2}) = nums[15] * std::conj(nums[2]);
        Index({3, 2, 2, 3}) = nums[15] * std::conj(nums[6]);
        Index({3, 2, 3, 0}) = nums[11] * std::conj(nums[10]);
        Index({3, 2, 3, 1}) = nums[11] * std::conj(nums[14]);
        Index({3, 2, 3, 2}) = nums[15] * std::conj(nums[10]);
        Index({3, 2, 3, 3}) = nums[15] * std::conj(nums[14]);

        Index({3, 3, 0, 0}) = nums[3] * std::conj(nums[3]);
        Index({3, 3, 0, 1}) = nums[3] * std::conj(nums[7]);
        Index({3, 3, 0, 2}) = nums[7] * std::conj(nums[3]);
        Index({3, 3, 0, 3}) = nums[7] * std::conj(nums[7]);
        Index({3, 3, 1, 0}) = nums[3] * std::conj(nums[11]);
        Index({3, 3, 1, 1}) = nums[3] * std::conj(nums[15]);
        Index({3, 3, 1, 2}) = nums[7] * std::conj(nums[11]);
        Index({3, 3, 1, 3}) = nums[7] * std::conj(nums[15]);
        Index({3, 3, 2, 0}) = nums[11] * std::conj(nums[3]);
        Index({3, 3, 2, 1}) = nums[11] * std::conj(nums[7]);
        Index({3, 3, 2, 2}) = nums[15] * std::conj(nums[3]);
        Index({3, 3, 2, 3}) = nums[15] * std::conj(nums[7]);
        Index({3, 3, 3, 0}) = nums[11] * std::conj(nums[11]);
        Index({3, 3, 3, 1}) = nums[11] * std::conj(nums[15]);
        Index({3, 3, 3, 2}) = nums[15] * std::conj(nums[11]);
        Index({3, 3, 3, 3}) = nums[15] * std::conj(nums[15]);

    }


}