/*
Copyright 2017 Eric Schuyler Fried, Nicolas Per Dane Sawaya, Alán Aspuru-Guzik

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions 
and limitations under the License.
*/






#include <iostream>
#include "GraphNode.h"
#include <vector>
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

//this function determines if two vertices are already connected
bool alreadyConnected(std::shared_ptr<GraphNode> a, std::shared_ptr<GraphNode> b)
{
bool retVal{false};
    std::for_each(a->mConnections.begin(), a->mConnections.end(),[b, &retVal](std::shared_ptr<GraphNode> temp){
       if(temp == b)
       {
           retVal = true;
       }
    });
    return retVal;
}


//this function randomly generates one x-regular (x is an input param) graph with numNodes vertices
//returns -1 if it fails
int run(const int numNodes, std::mt19937& gen,const std::string& output, const int x)
{
    std::vector<std::shared_ptr<GraphNode>> nodesVect(numNodes);
    std::generate(nodesVect.begin(), nodesVect.end(),[](){
        std::shared_ptr<GraphNode> temp = std::make_shared<GraphNode>();
        return temp;
    });

    std::uniform_int_distribution<> dist(0,numNodes-1);
    std::ofstream out(output);
    if(!out.is_open())
    {
        std::cout<<"Output file not open!"<<std::endl;
        return -1;
    }
    out<<"c Randomly generated "<<numNodes<<" vertex "<<x<<"regular graph"<<std::endl;

    int one;
    int two;
    for(int i = 0; i< x * numNodes/2; i++)
    {
        bool satisfied(false);
        one = dist(gen);
        two = dist(gen);
        if(one != two)
        {
            if(nodesVect[one]->mEdgeCount<x && nodesVect[two]->mEdgeCount<x  && !alreadyConnected(nodesVect[one],nodesVect[two]))
            {
                nodesVect[one]->mConnections.push_back(nodesVect[two]);
                nodesVect[two]->mConnections.push_back(nodesVect[one]);
                nodesVect[one]->mEdgeCount++;
                nodesVect[two]->mEdgeCount++;
                out<<"e "<<one<<" "<<two<<std::endl;
                satisfied=true;
                nodesVect[one]->mConnected=true;
                nodesVect[two]->mConnected=true;
            }
        }
       for(int k =0; k< 100; k++)
       {
            one = dist(gen);
            two = dist(gen);
            if(one != two)
            {
                if(nodesVect[one]->mEdgeCount<x && nodesVect[two]->mEdgeCount<x && !alreadyConnected(nodesVect[one],nodesVect[two]))
                {
                    nodesVect[one]->mConnections.push_back(nodesVect[two]);
                    nodesVect[two]->mConnections.push_back(nodesVect[one]);
                    nodesVect[one]->mEdgeCount++;
                    nodesVect[two]->mEdgeCount++;
                    out<<"e "<<one<<" "<<two<<std::endl;
                    satisfied=true;
                    nodesVect[one]->mConnected=true;
                    nodesVect[two]->mConnected=true;
                }
            }
           if(satisfied)
           {
               break;
           }
        }
    }
    int count(0);
    bool correct=true;
    for(auto& temp: nodesVect)
    {

        if(temp->mEdgeCount<x)
        {
            count++;
            correct=false;
        }
    }
    out.close();
    if(!correct || count>0)
    {
        return -1;
    }
    return 0;
}


int main(int argc, char * argv[]) {
    std::random_device randomDevice;
    std::mt19937 gen(randomDevice());
    if(argc<3)
    {
        std::cout<<"Please provide the regularity, the number of graphs to generate, and the number of vertices in the graph"<<std::endl;
        return -1;
    }


    int numNodes = atoi(argv[3]);
    int regularity = atoi(argv[1]);
    int numGraphs = atoi(argv[2]);
    mkdir("Output",0755);

    for(int i=0; i<numGraphs; i++)
    {
        std::stringstream ss;
        ss << "Output/" << regularity << "regRand" << numNodes << "Node" << i << ".dgf";
        int retVal = run(numNodes, gen, ss.str(), regularity);
        while (retVal != 0) {
            retVal = run(numNodes, gen, ss.str(), regularity);
        }
    }
    return 0;
}
