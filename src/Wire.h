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

#include <memory>
#include "Exceptions.h"

//wire class
namespace qtorch {
    class Wire {
    public:
        explicit Wire(std::shared_ptr<class Node> nodeA, std::shared_ptr<class Node> nodeB, int qubitNum) :
                mNodeA(std::weak_ptr<class Node> (nodeA)),
                mNodeB(std::weak_ptr<class Node> (nodeB)),
                mQubitNumber(qubitNum)
        {};

        void SetNodeA(std::shared_ptr<Node> newNode) { mNodeA = std::weak_ptr<Node>(newNode); };

        void SetNodeB(std::shared_ptr<Node> newNode) { mNodeB = std::weak_ptr<Node>(newNode); };

        int GetQubitNumber() { return mQubitNumber; };

        void SetQubitNumber(int newNum) { mQubitNumber = newNum; };

        std::weak_ptr<Node> GetNodeA() { return mNodeA; };

        std::weak_ptr<Node> GetNodeB() { return mNodeB; };

        void SetWireID(int wid) { mWireID = wid; };  // npds
        int GetWireID() { return mWireID; };  // npds

        void SetIsContracted(bool inp) { this->boolIsContracted = inp; };  // npds
        bool IsContracted() { return boolIsContracted; }; // npds

    private:
        std::weak_ptr<Node> mNodeA;
        std::weak_ptr<Node> mNodeB;

        int mWireID;  // npds 1feb2017 // Used only for some subroutines. Like linegraph.

        int mQubitNumber;
        int boolIsContracted = false; // npds
    };
}