#
# Copyright 2017 Eric Schuyler Fried, Nicolas Per Dane Sawaya, Alán Aspuru-Guzik
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
#         limitations under the License.






#pragma once

#include <iostream>
#include <vector>


//this struct is a vertex in the graph that counts the number of edges coming out from it
//it also contains a vector of all the connected nodes, and a boolean that is set to true if the node is connected to a graph of more than 0 nodes
struct GraphNode
{
    int mEdgeCount{0};
    bool mConnected{false};
    std::vector<std::shared_ptr<GraphNode>> mConnections;
};
