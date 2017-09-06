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

/*
Main executable for simulating arbitrary qasm circuits.

*/


#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include "Network.h"
#include "Timer.h"
#include "ContractionTools.h"
#include "LineGraph.h"
#include "leviParser.hpp"
#include "Exceptions.h"

using namespace qtorch;


void setInpDefaultsTN(leviParser &parser);
std::string formattedTime(double inp);



int main(int argc, char *argv[]) {
    
    
    if(argc<2) {
        std::cout << "Usage:\nExecutable <input file>\n";
        return -1;
    }
    
    
    // Parse input file, which should include all parameters
    leviParser inpvars;  // Parser that holds input data
    setInpDefaultsTN(inpvars); // *** Set defaults BEFORE reading input file ***
    inpvars.readInputFile( std::string(argv[1]) );  // Read input files
    
    mkdir("output",0755);
    
    
    
    
    
    
    // Make network
    std::cout << "QASM file: " << inpvars.mapString[ "qasm" ] << "\n";
    std::cout << "Meas file: " << inpvars.mapString["measurement"]<<"\n";
    std::cout << "Output file: "<<inpvars.mapString["outputpath"]<<"\n";
    std::ofstream outputFile(inpvars.mapString["outputpath"]);
    if(!outputFile)
    {
        std::cout<<"Invalid Output File Path"<<std::endl;
        outputFile<<"Invalid Output File Path"<<std::endl;
        return -1;
    }
    std::shared_ptr<Network> netw;
    try {
        netw =
                std::make_shared<Network>(inpvars.mapString["qasm"].c_str(),
                                          inpvars.mapString["measurement"].c_str());
    }
    catch (std::exception& e)
    {
        std::cout<<e.what()<<std::endl;
        outputFile<<e.what()<<std::endl;
        return -1;
    }
    
    
    // Set # of threads
    //int threads(2); DEFAULT WAS SET IN setInpDefaultsTN()
    //std::cout<<std::endl<<std::endl;
    
    std::cout<<"========Threading Info========"<<std::endl;
    // (You have more threads than values if #threads > 4^THRESH_RANK_THREAD)
    if(inpvars.mapInt["threads"]<=0||inpvars.mapInt["threads"]>pow(4,THRESH_RANK_THREAD))
    {
        std::cout << "Invalid Number of Threads in Input File. ";
        std::cout << "If it is a large number, try reducing the number of threads. Thread number set to 2."<<std::endl;
        inpvars.mapInt["threads"] = 2;
    }
    std::cout << "Number of Threads set to: " << inpvars.mapInt["threads"]<< std::endl;
    // Set #threads in the network object
    netw->SetNumThreads( inpvars.mapInt["threads"] );
    std::cout<<"=====End of Threading Info===="<<"\n\n";
    
    
    
    // Start timer (uses high_resolution_clock, which should do CPU not wall time)
    Timer tim;
    tim.start();
    
    // Reduce circuit (contracts one-and two-qubit gates)
    netw->ReduceCircuit();
    std::cout << "Throughout, time elapsed after reading in circuit is "
    << "given in { curly brackets }. Time starts after circuit has been read in.\n\n";
    std::cout << "Reduced circuit (removed 1- and 2-qub gates) "
    << formattedTime(tim.getElapsed())  << "\n\n";
    
    // Boolean for successful contraction
    bool succ = false;
    
    // Contract, using specificed method
    std::string contractmeth = inpvars.mapString["contractmethod"];
    std::cout << "Contraction method: " << contractmeth << "\n";
    if( contractmeth=="linegraph-qbb" ) {
    
        // Method:
        // Linegraph - tree decomposition - QuickBB
        std::cout << "Contraction method: Linegraph / tree decomposition\n";
        LineGraph lg(netw);
        
        
        // Run quickbb and get the contraction
        // should be three options: (qbb+contract),qbb-only,contract-only
        // need to make qbb a separate thing in lgcontract.
        
        
        if (inpvars.mapBool["qbbonly"]) {  // Only do qbb ordering, then exit. Don't contract.
            
            try {
                bool sixtyfourbit(true);
                std::cout << "qbbonly=true. Only running qbb on linegraph, not doing contraction.\n";
                std::cout << "quickbbseconds set to: " << inpvars.mapInt["quickbbseconds"] << std::endl;
                if(inpvars.mapBool.find("64bit")!=inpvars.mapBool.end())
                {
                    if(!inpvars.mapBool["64bit"])
                    {
                        sixtyfourbit = false;
                    }
                }
                succ = lg.runQuickBB(inpvars.mapInt["quickbbseconds"], &tim, sixtyfourbit);
                std::cout << "QuickBB has been run. Set qbbonly=false and readqbbresonly=true to contract network. ";
                std::cout << "Exiting.\n";
                return 0;
            }
            catch (std::exception &e) {
                std::cout << e.what() << std::endl;
                outputFile<<e.what()<<std::endl;
            }
            
        } else if (inpvars.mapBool["readqbbresonly"]) {  // Read in qbb ordering from before, and contract.
            
            try {
                std::cout << "readqbbresonly=true. Attempting to read previous qbb result, and contracting network.\n";
                succ = lg.LGContract();
            }
            catch (std::exception &e) {
                std::cout << e.what() << std::endl;
                outputFile<<e.what()<<std::endl;
            }
            if(succ)
            {
                std::cout<<"Result of Contraction (also printed to file): "<<netw->GetFinalValue()<<std::endl;
                outputFile<<"Result of Contraction: "<<netw->GetFinalValue()<<std::endl;
            }
            
        } else {  // Default. Both run qbb and run contraction.
            
            try {
                bool sixtyfourbit(true);
                std::cout << "quickbbseconds set to: " << inpvars.mapInt["quickbbseconds"] << std::endl;
                if(inpvars.mapBool.find("64bit")!=inpvars.mapBool.end())
                {
                    if(!inpvars.mapBool["64bit"])
                    {
                        sixtyfourbit = false;
                    }
                }
                succ = lg.runQuickBB(inpvars.mapInt["quickbbseconds"],&tim,sixtyfourbit);
                succ = lg.LGContract();
            }
            catch (std::exception &e) {
                std::cout << e.what() << std::endl;
                outputFile<<e.what()<<std::endl;
            }
            if(succ)
            {
                std::cout<<"Result of Contraction: "<<netw->GetFinalValue()<<std::endl;
                outputFile<<"Result of Contraction: "<<netw->GetFinalValue()<<std::endl;
            }
        }
        
        
        
        
    }
    else if ( contractmeth=="simple-stoch" ) //simple stochastic method option
    {
        std::shared_ptr<Network> final;
        try {
            ContractionTools p(netw);
            final = p.Contract(Stochastic);
            std::cout << "Result of contraction:\n"
                      << p.GetFinalVal() << "\n";
            outputFile<<"Result of Contraction: "<<p.GetFinalVal()<<std::endl;

        }
        catch(std::exception& e)
        {
            std::cout<<e.what()<<std::endl;
            outputFile<<e.what()<<std::endl;
            return -1;
        }
        if(final!= nullptr)
        {
            succ = true;
        }
    }
    else if(contractmeth == "user-defined") //user - defined contraction option
    {
        std::shared_ptr<Network> final;
        if(inpvars.mapString.find("user-contract-seq")==inpvars.mapString.end()) //if the user forgot to define a user contraction sequence file, contract via stochastic
        {
            std::cout<<"User contraction sequence file was not defined - contracting via simple stochastic"<<std::endl;
            try {
                ContractionTools p(netw);
                final = p.Contract(Stochastic);
                std::cout << "Result of contraction:\n"
                          << p.GetFinalVal() << "\n";
                outputFile<<"Result of Contraction: "<<p.GetFinalVal()<<std::endl;
            }
            catch(std::exception& e)
            {
                std::cout<<e.what()<<std::endl;
                outputFile<<e.what()<<std::endl;
                return -1;
            }
            if(final!= nullptr)
            {
                succ = true;
            }
        }
        else
        {

            try {
                ContractionTools p(netw);
                final = p.ContractUserDefinedSequenceOfWires(inpvars.mapString["user-contract-seq"]);
                std::cout << "Result of contraction:\n"
                          << p.GetFinalVal() << "\n";
                outputFile<<"Result of Contraction: "<<p.GetFinalVal()<<std::endl;
            }
            catch(std::exception& e)
            {
                std::cout<<e.what()<<std::endl;
                outputFile<<e.what()<<std::endl;
                return -1;
            }
            if(final!= nullptr)
            {
                succ = true;
            }

        }

    }
    else {
    
        std::cout << "Error. 'contractmethod' bad option.\n";
        return -1;
    }
    
    

    // Finish
    if(succ) {
        
        // Print compute metrics
        std::cout << "Number of floating point ops in full contraction: "
        << netw->getNumFloatOps() << "\n";
        outputFile << "Number of floating point ops in full contraction: "
                  << netw->getNumFloatOps() << "\n";

        std::cout << "Contraction complete. " << formattedTime(tim.getElapsed())
        << "\n";
        outputFile << "Contraction complete. " << formattedTime(tim.getElapsed())
                  << "\n";
        
    } else {
        std::cout << "ERROR. ABORTING.\n" << formattedTime(tim.getElapsed())
        << "\n";
        outputFile << "ERROR. ABORTING.\n" << formattedTime(tim.getElapsed())
                  << "\n";
        return -1;
    }
    
    
    return 0;
    
}






void setInpDefaultsTN(leviParser &parser) {

    
    // QuickBB runtime default
    parser.mapInt["quickbbseconds"] = 20;
    
    // Set # of threads default
    parser.mapInt["threads"] = 2;
    
    // Defaults for linegraph and qbb
    parser.mapBool["qbbonly"] = false;
    parser.mapBool["readqbbresonly"] = false;
    parser.mapString["outputpath"] = "output/qtorch.out";
    
    
    
    
    
    
    
}


std::string formattedTime(double inp) {

    std::string vStr = std::to_string(inp);
    
    return std::string(" { ") + vStr + std::string(" } ");
    
}





