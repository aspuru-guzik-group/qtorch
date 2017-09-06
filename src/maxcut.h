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

#include <iostream>
#include <fstream>
#include "Timer.h"
#include "ContractionTools.h"
#include "preprocess.h"
#include <nlopt.hpp>
#include <sys/stat.h>



struct ExtraData
{
    ExtraData(const int p0, const char* filename0):fileName(filename0),p(p0){ReadInData();PopulateIterations();};
    ExtraData(){};
    std::string fileName;
    std::vector<std::pair<int, int>> pairs;
    std::vector<std::vector<int>> adjacencyLists;
    std::vector<int> qubitsNeeded;
    std::string outputFile;
    int numQubits;
    int p;
    std::vector<std::vector<std::pair<int,int>>> iterations; //first pair in the list is the measurement to perform
    std::vector<std::vector<std::pair<int,int>>> realIterations;
    void ReadInData();
    void PopulateIterations();
    void PopulateIterationsHelper (int counter,
                                              std::vector<int>& workingVerticesList,
                                              std::vector<bool>& newWorkingVertices,
                                              std::vector<bool>& hasBeenChecked,
                                              std::vector<bool>& isInWorkingNodes,
                                              int iterationIndex,
                                                std::vector<int>& mapToRealIt);

};

void ExtraData::ReadInData()
{
    std::string t;
    pairs.reserve(1000);
    std::ifstream input(fileName);
    if(!input.is_open())
    {
        std::cout<<"Could Not Open File"<<std::endl;
        throw "File Not Open";
    }
    int max =0;
    char temp;
    while(input>>temp)
    {	
	//std::cout << temp;
	    
        if(temp == 'c') {
            // 'c' designates a comment
            std::getline(input,t);
            continue;
        } else if(temp != 'e') {
            std::cout<<"Error parsing file"<<std::endl;
            std::getline(input,t);
            continue;
        }
        int one, two;
        input>>one>>two;
        if(one>max)
        {
            max = one;
        }
        if(two>max)
        {
            max = two;
        }
        pairs.push_back({one,two});
    }
    numQubits=max+1;
    input.close();

    adjacencyLists.resize(numQubits);

    for(const auto& tempPair:pairs)
    {
        adjacencyLists[tempPair.first].push_back(tempPair.second);
        adjacencyLists[tempPair.second].push_back(tempPair.first);
    }
};

void ExtraData::PopulateIterations()
{
    qubitsNeeded = std::move(std::vector<int>(pairs.size(),0));
    int placeInIterations(0);
    iterations.resize(pairs.size());
    realIterations.resize(pairs.size());
    std::vector<int> mapToRealIterations(numQubits);
    for(const auto& currentPair: pairs)
    {
        std::fill(mapToRealIterations.begin(),mapToRealIterations.end(),-1);
        std::vector<int> workingVertices;
        workingVertices.reserve(numQubits);
        workingVertices.push_back(currentPair.first);
        workingVertices.push_back(currentPair.second);
        mapToRealIterations[currentPair.first] = 0;
        qubitsNeeded[placeInIterations]++;
        std::vector<bool>isNewWorking(numQubits,false);
        std::vector<bool>wasChecked(numQubits, false);
        std::vector<bool>isAlreadyInWorkingNodes(numQubits,false);
        isAlreadyInWorkingNodes[currentPair.first] = true;
        isAlreadyInWorkingNodes[currentPair.second]=true;
        int count(0);
        PopulateIterationsHelper(0,workingVertices,isNewWorking,wasChecked,isAlreadyInWorkingNodes,placeInIterations,mapToRealIterations);
        for(const auto& toCopy: iterations[placeInIterations])
        {
            realIterations[placeInIterations].push_back({mapToRealIterations[toCopy.first],mapToRealIterations[toCopy.second]});
        }
       placeInIterations++;
    }
};

void ExtraData::PopulateIterationsHelper (int counter,
                                          std::vector<int>& workingVerticesList,
                                          std::vector<bool>& newWorkingVertices,
                                          std::vector<bool>& hasBeenChecked,
                                          std::vector<bool>& isInWorkingNodes,
                                          int iterationIndex,
                                          std::vector<int>& mapToRealIt){
    if(counter==p)
    {
        return;
    }

    for(const auto& vertex: workingVerticesList)
    {
        for(const auto& tempVertex: adjacencyLists[vertex])
        {
            if(!hasBeenChecked[tempVertex])
            {
                if(mapToRealIt[tempVertex]==-1)
                {
                    mapToRealIt[tempVertex] = qubitsNeeded[iterationIndex];
                    qubitsNeeded[iterationIndex]++;
                }
                iterations[iterationIndex].push_back({vertex,tempVertex});
                if(!isInWorkingNodes[tempVertex])
                {
                    newWorkingVertices[tempVertex] = true;
                }
                //can't be a new working node if it's already in working nodes
            }
        }
        hasBeenChecked[vertex]=true;
    }
    workingVerticesList.clear();
    std::fill(isInWorkingNodes.begin(),isInWorkingNodes.end(), false);
    int i(0);
    std::for_each(newWorkingVertices.begin(),newWorkingVertices.end(),[&workingVerticesList,&i, &isInWorkingNodes](bool isNew)
    {
        if(isNew)
        {
            workingVerticesList.push_back(i);
            isInWorkingNodes[i]=true;
        }
        i++;
    });

    std::fill(newWorkingVertices.begin(),newWorkingVertices.end(), false);

    PopulateIterationsHelper(++counter,workingVerticesList,newWorkingVertices,hasBeenChecked,isInWorkingNodes,iterationIndex, mapToRealIt);

};

//this function takes in a vector of pairs to fill, a filename to read in from, and a reference to the number of qubits
//the function reads in lines from the file and adds the pair to the pairs vector, taking the max value of any one of the nodes
//to be the number of qubits minus one
//note that pairs are not repeated --> i.e. if there is e 30 45, e 45 30 will not be present


void outputInitialPlusStateToFile(std::ofstream& qasmFile, const int numQubits)
{
    for(int i=0; i<numQubits; i++)
    {
        qasmFile<<"H "<<i<<std::endl;
    }

}



void applyU_CsThenU_Bs(const std::vector<std::pair<int,int>>& objectiveF, const int p, const std::vector<double>& betas_gammas, const int numQubits, std::ofstream& output)
{
    //function definitions
    auto applyU_C = [](const std::pair<int,int>& toApplyTo, std::ofstream& op, const double gamma)
    {
        op<<"CNOT "<<toApplyTo.first<<" "<<toApplyTo.second<<std::endl;
        op<<"Rz "<<-gamma<<" "<<toApplyTo.second<<std::endl;
        op<<"CNOT "<<toApplyTo.first<<" "<<toApplyTo.second<<std::endl;
    };

    auto applyU_B = [](const int toApplyTo, std::ofstream& op, const double beta)
    {
        op<<"Rx "<<beta*2.0<<" "<<toApplyTo<<std::endl;
    };


    for(int i =0; i<p; i++)
    {

        for(auto& tempPair: objectiveF)
        {
            applyU_C(tempPair,output,betas_gammas[i+p]);
        }
        for (int (j) = 0; (j) < numQubits; ++(j))
        {
            applyU_B(j,output,betas_gammas[i]);
        }
    }
}

