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

#include <csignal>
#include "ContractionTools.h"

namespace qtorch {

//the function runs a given circuit until it finds a contraction sequence that runs in <= maxTime seconds
//it returns true if a sequence is found, else it returns false
    bool preProcess(const std::string &fileName, std::vector<std::pair<int, int>> &optimalContractionSequence,
                    const double timeThreshold) {
        maxTime = timeThreshold;
        for (int i(0); i < 100; i++) {
            double newTime;
            totTimer = Timer();
            totTimer.start();
            ContractionTools p(fileName, "measureTest.txt");
            std::shared_ptr<Network> temp = p.Contract(Stochastic);
            remove("measureTest.txt");
            newTime = totTimer.getElapsed();
            if (newTime <= timeThreshold) //if sequence is found
            {
                std::for_each(temp->GetAllNodes().begin(), temp->GetAllNodes().end(),
                              [&optimalContractionSequence](std::shared_ptr<Node> node) {
                                  if (!(node->mCreatedFrom.first == 0 && node->mCreatedFrom.second == 0)) {
                                      optimalContractionSequence.push_back(node->mCreatedFrom);
                                  }
                              });
                totTimer.reset();
                return true;
            }
        }
        return false;
    };
}