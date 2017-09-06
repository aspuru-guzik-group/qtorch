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



//this is a maxcut algorithm testing class for the tensor network code
#include "maxcut.h"
#include "nlopt.h"
#include "Exceptions.h"

using namespace qtorch;

//this function finds the string that supposedly solves the maxcut problem given as input a contraction sequence from the preprocessing routine
//and the optimized gammas and betas.. it prints the answer string and cut data to maxcutAnswerString.txt
void maxcutGetFinalString(std::string& graphFilePath, int p,std::vector<std::pair<int,int>>& contractionSequence, std::vector<double>& gAndB, const std::string& outfilePath){
    //the goal of this main class is to run the maxcut algorithm on a graph supplied in the command line arguments
    //the class will use qaoa to solve the maxcut problem, outputting a qasm circuit and simulating it for each iteration of of the maximization algorithm
    //the user will specify the "p" and the graph file in the command line, and this class will run a maxcut simulation based on the graph
    //a graph file will be in .dgf format -- an example can be found @qft8.dgf, but lines should only include e [node#1] [node#2]
    //part of the algorithm will be to generate a .qasm file for each iteration of the minimization (maximization) function
    //additionally, a simple optimization function will be needed
    Timer z;
    z.start();
    ExtraData e(p,graphFilePath.c_str());
    e.outputFile = outfilePath;


    auto calculateFinalString = [&z, &contractionSequence](const std::vector<double>& betas_gammas, ExtraData * f_data) {
        //data
        srand(time(NULL));
        std::vector<bool> answerString;
        std::ofstream maxCutCircuitQasm;
        std::ofstream maxCutAnswer(f_data->outputFile);
        maxCutAnswer<<f_data->fileName<<std::endl;
        double currentProb(1.0);

        maxCutCircuitQasm.open("input/tempMaxCut.qasm");
        maxCutCircuitQasm<<f_data->numQubits<<std::endl;
        outputInitialPlusStateToFile(maxCutCircuitQasm,f_data->numQubits);
        applyU_CsThenU_Bs(f_data->pairs,static_cast<ExtraData *>(f_data)->p,betas_gammas,f_data->numQubits,maxCutCircuitQasm);
        maxCutCircuitQasm.close();

        for(int i=0; i<f_data->numQubits; i++)
        {
            std::ofstream measurements("input/measureTest.txt");
            for(int j=0; j<static_cast<ExtraData *>(f_data)->numQubits; j++)
            {
                if(j<answerString.size())
                {
                    if(answerString[j])
                    {
                        measurements <<"1 ";
                    }
                    else
                    {
                        measurements<<"0 ";
                    }
                }
                else if(j==i)
                {
                    measurements<<"0 ";
                }
                else
                {
                    measurements<<"T ";
                }
            }
            measurements.close();
            ContractionTools p ("input/tempMaxCut.qasm","input/measureTest.txt");
            if(contractionSequence.size()==0)
            {
                p.Contract(Stochastic);
            }
            else
            {
                p.ContractGivenSequence(contractionSequence);
            }
            double probNextGivenPrev(p.GetFinalVal().real()/currentProb);
                if (probNextGivenPrev > 0.5)
                {
                    currentProb *= probNextGivenPrev;
                    answerString.push_back(false);
                }
                else if (probNextGivenPrev < 0.5)
                {
                    currentProb *= (1.0-probNextGivenPrev);
                    answerString.push_back(true);
                }
                else
                {
                    int r = rand() % 2;
                    if (r == 1) {
                        answerString.push_back(true);
                    } else {
                        answerString.push_back(false);
                    }
                    currentProb*=0.5;
                }


        }
        int counter(0);
        std::for_each(answerString.begin(),answerString.end(),[&maxCutAnswer](bool b){
           maxCutAnswer<<b<<" ";
        });

        int cutEdgeCount(0);
        std::for_each(f_data->pairs.begin(),f_data->pairs.end(),[&answerString,&cutEdgeCount,&f_data](std::pair<int,int> edge){
            if(answerString[edge.first]!=answerString[edge.second])
            {
                cutEdgeCount++;
            }
        });
        maxCutAnswer<<std::endl<<"Cut edges: "<<cutEdgeCount<<"/"<<f_data->numQubits*3/2<<std::endl;
        maxCutAnswer<<"Time elapsed: "<<z.getElapsed()<<std::endl;
        maxCutAnswer.close();


    };


    calculateFinalString(gAndB,&e);



}

//finds the optimal gammas and betas, and prints them to tempGammasAndBetas.txt
//will occasionally cause a memory error (something to do with the optimization function), so you just have to run it again

void maxcutGetOptimalAngles(std::string& graphFilePath, int p, const std::string& outputPath){
    //the goal of this main class is to run the maxcut algorithm on a graph supplied in the command line arguments
    //the class will use qaoa to solve the maxcut problem, outputting a qasm circuit and simulating it for each iteration of of the maximization algorithm
    //the user will specify the "p" and the graph file in the command line, and this class will run a maxcut simulation based on the graph
    //a graph file will be in .dgf format -- an example can be found @qft8.dgf, but lines should only include e [node#1] [node#2]
    //part of the algorithm will be to generate a .qasm file for each iteration of the minimization (maximization) function
    //additionally, a simple optimization function will be needed

    Timer z;
    z.start();
    std::vector<double> betas_gammas0(2 * p);
    std::fill(betas_gammas0.begin(), betas_gammas0.begin() + betas_gammas0.size() / 2, 0.392699);
    std::fill(betas_gammas0.begin() + betas_gammas0.size() / 2, betas_gammas0.end(), 0.785399);
    ExtraData e(p,graphFilePath.c_str());
    e.outputFile = outputPath;


    auto F_p = [](const std::vector<double>& betas_gammas,std::vector<double>& grad, void * f_data)->double {
        //data
        std::ofstream maxCutCircuitQasm;
        std::ofstream tempGammasAndBetas(static_cast<ExtraData *>(f_data)->outputFile);


        //initialize and apply functions
        double f_pVal(0.0);
        int counter(0);
        std::for_each(static_cast<ExtraData *>(f_data)->pairs.begin(),static_cast<ExtraData *>(f_data)->pairs.end(), [&maxCutCircuitQasm, &f_pVal, f_data, &betas_gammas, &counter](std::pair<int,int> tem) {
            maxCutCircuitQasm.open("input/tempMaxCut.qasm");
            std::ofstream measurements("input/measureTest.txt");
            int numQubits(static_cast<ExtraData *>(f_data)->qubitsNeeded[counter]);
            maxCutCircuitQasm << numQubits << std::endl;
            outputInitialPlusStateToFile(maxCutCircuitQasm, numQubits);
            applyU_CsThenU_Bs(static_cast<ExtraData *>(f_data)->realIterations[counter], static_cast<ExtraData *>(f_data)->p, betas_gammas,numQubits,
                              maxCutCircuitQasm);
            for (int i = 0; i < numQubits; i++) {
                if (i == 0 || i == 1)
                {
                    measurements << "Z ";
                }
                else
                {
                    measurements<<"T ";
                }
            }
            measurements.close();
            maxCutCircuitQasm.close();

            ContractionTools qComputer("input/tempMaxCut.qasm", "input/measureTest.txt");
            qComputer.Contract(Stochastic);


            f_pVal += 0.5 * (1.0 - qComputer.GetFinalVal().real());
            ++counter;
        });
        for(const auto& gORb: betas_gammas) {
            tempGammasAndBetas<<gORb<<" ";
        }
        tempGammasAndBetas.close();
        return f_pVal;
    };



    //optimize(maximize)
    try {

         nlopt::opt optimization(nlopt::LN_COBYLA, 2 * p);
         optimization.set_max_objective(F_p, &e);
         std::vector<double> finalBetas_Gammas(optimization.optimize(betas_gammas0));
        std::cout<<"Took "<<z.getElapsed()<<" seconds"<<std::endl;

    }
    catch (std::runtime_error &error) {
        std::cout << error.what() << std::endl;
        std::cout<<"Took "<<z.getElapsed()<<" seconds"<<std::endl;
        return;
    }
}

//main process

//command line arguments: <GraphFile Path> <p value> <0 for getAngles, 1 for final cut> <file path to either input angle file (if 1) or output angle file (if 0)> <seconds to preprocess for>
int main(int argc, char *argv[]) {
    if(argc<5)
    {
        std::cout<<"Not enough arguments"<<std::endl;
        std::cout<<"arguments: <GraphFile Path> <p value> <0 for getAngles> <file path to output angle file>\n";
        std::cout<<"OR arguments: <GraphFile Path> <p value> <1 for final cut> <file path to input angle file> <file path to output answer file> <seconds to preprocess for (optional)>\n";
        std::cout<<"OR arguments: <GraphFile Path> <p value> <2 for both> <file path to output answer file> <seconds to preprocess for (optional)>\n\n";
        return -1;
    }
    int pVal = atoi(argv[2]);
    int anglesOrFinalCut = atoi(argv[3]);
    if(anglesOrFinalCut==1 && argc<6)
    {
        std::cout<<"Not enough arguments"<<std::endl;
        std::cout<<"arguments: <GraphFile Path> <p value> <1 for final cut> <file path to either input angle file> <file path to output answer file <seconds to preprocess for (optional)>\n";
    }
    std::string graphFilePath(argv[1]);
    mkdir("output",0755);
    mkdir("input",0755);
    int procSec = 60;
    if(anglesOrFinalCut ==1 && argc ==7)
    {
        procSec = atoi(argv[6]);
    }
    if(anglesOrFinalCut==2 && argc ==6)
    {
        procSec = atoi(argv[5]);
    }

    //this part finds the opt string from gammas and betas

    if(anglesOrFinalCut==1) {
        std::ifstream inAngles(argv[4]);
        std::string outfilePath(argv[5]);
        std::vector<double> gammasAndBetas;
        for(int i=0; i<2*pVal;i++)
        {
            double z;
            inAngles>>z;
            gammasAndBetas.push_back(z);
        }
        ExtraData e(pVal, graphFilePath.c_str());
        std::ofstream maxCutCircuitQasm("input/tempMaxCut.qasm");
        maxCutCircuitQasm << e.numQubits << std::endl;
        outputInitialPlusStateToFile(maxCutCircuitQasm, e.numQubits);
        applyU_CsThenU_Bs(e.pairs, pVal, gammasAndBetas, e.numQubits, maxCutCircuitQasm);
        maxCutCircuitQasm.close();
        bool success(false);
        std::vector<std::pair<int, int>> optContract;
        success = preProcess("input/tempMaxCut.qasm", optContract, procSec);
        if (success)
        {
            maxcutGetFinalString(graphFilePath, pVal, optContract, gammasAndBetas, outfilePath);
        }
        else
        {
            std::cout<<"Preprocessing Failed"<<std::endl;
            return -1;
        }
    }
    else if(anglesOrFinalCut==0)
    {
        std::string outputPath(argv[4]);
        maxcutGetOptimalAngles(graphFilePath, pVal,outputPath);
    }
    else if(anglesOrFinalCut==2)
    {
        std::string outfilePath(argv[4]);
        maxcutGetOptimalAngles(graphFilePath, pVal,"tempAngles.txt");
        std::ifstream inAngles("tempAngles.txt");
        std::vector<double> gammasAndBetas;
        for(int i=0; i<2*pVal;i++)
        {
            double z;
            inAngles>>z;
            gammasAndBetas.push_back(z);
        }
        inAngles.close();
        remove("tempAngles.txt");
        ExtraData e(pVal, graphFilePath.c_str());
        std::ofstream maxCutCircuitQasm("input/tempMaxCut.qasm");
        maxCutCircuitQasm << e.numQubits << std::endl;
        outputInitialPlusStateToFile(maxCutCircuitQasm, e.numQubits);
        applyU_CsThenU_Bs(e.pairs, pVal, gammasAndBetas, e.numQubits, maxCutCircuitQasm);
        maxCutCircuitQasm.close();
        bool success(false);
        std::vector<std::pair<int, int>> optContract;
        success = preProcess("input/tempMaxCut.qasm", optContract, procSec);
        if (success)
        {
            maxcutGetFinalString(graphFilePath, pVal, optContract, gammasAndBetas,outfilePath);
        }
        else
        {
            std::cout<<"Preprocessing Failed"<<std::endl;
            return -1;
        }

    }

    return 0;
}

