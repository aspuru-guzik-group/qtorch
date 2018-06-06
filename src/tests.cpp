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


/*This file is a comprehensive testing executable file for the tensor network library. The test can be compiled through
 * the makefile and run through command line execution. Each separate test method is designed to test a certain part of
 * the library. However, in the runTests function, the user can change a map to select which tests to run. The default is all
 *
 * Additionally, the file contains templated helper functions that are necessary for some of the tests. Please do not modify these
 * functions.
 *
 * Please see individual function implementations for specific comments
*/

#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include "Network.h"
#include "Timer.h"
#include "ContractionTools.h"
#include "LineGraph.h"

using namespace qtorch;

bool testArbitraryOneQubit (std::ofstream& out);
bool simpleReduceAndContractTest(std::ofstream& out);
bool testRotationCircuits (std::ofstream& out);
bool testArbitraryTwoQubit (std::ofstream& out);
bool testLineGraph(std::ofstream& out);
bool largeCircuitTest(std::ofstream& out);
bool bellTest(std::ofstream& out);
bool catTest(std::ofstream& out);
bool teleporationTest(std::ofstream& out);
bool tofolliTest(std::ofstream& out);
bool randomCircuitsTest(std::ofstream& out);
bool testUserDefinedSequence (std::ofstream& out);
void removeFile(const std::string& filePath);
const std::string generateQASMWithDiffPureInputState(const std::string& origQASMFilePath, const std::vector<bool>& inputState);
void runTests(const std::string& fileToOutputTo);
std::vector<std::complex<double>> applySimple2qubitSim(const std::string& gateFile, std::ofstream& out);
template <typename T>
void multiplyMatrixToVector(std::vector<std::vector<T>>& matrix,std::vector<T>& psi);
template <typename T>
void measureExpectation(std::vector<T>& psi, std::vector<std::vector<T>>& matrix, T& result);

//main function - creates the input and output directories if they don't already exist.
//Also defines the output test log and runs the tests.
int main()
{
    mkdir("output", 0755);
    mkdir("input", 0755);
    char output[] = {"output/testingoutput.log"};
    runTests(output);
    return 0;
}

//this function takes in a qasm file and an input state as 0's and 1's and generates a new qasm file with the same circuit but
//the new input state. The new file path will be the original file path + <input state> + .qasm.
//Ex. Samples/tofolli.qasm + input state: 011 -> Samples/tofolli011.qasm
//The function returns the new file path
const std::string generateQASMWithDiffPureInputState(const std::string& origQASMFilePath, const std::vector<bool>& inputState)
{
    std::stringstream ss;
    std::string newPath(origQASMFilePath);
    ss<<newPath.substr(0,newPath.find(".qasm"));
    for(bool b: inputState)
    {
        b ? ss<<'1': ss<<'0';
    }
    ss<<".qasm\0";
    std::string numQubits;
    std::ifstream src(origQASMFilePath);
    std::getline(src,numQubits);
    std::ofstream  dst(ss.str());
    dst<<numQubits.c_str()<<'\n';
    for(int i(0); i<inputState.size(); ++i)
    {
        if(inputState[i])
        {
            dst << "X " << i<<'\n';
        }
    }
    dst << src.rdbuf();
    dst.close();
    src.close();
    return ss.str();
}

//this function takes in a filepath as a string and removes the file at the filepath via a command line call, if it exists
void removeFile(const std::string& filePath)
{
    std::stringstream ss;
    ss<<"rm "<<filePath;
    system(ss.str().c_str());
}

//this function takes in a vector of vectors (matrix) and a vector both by reference, and modifies the vector by multiplying
//it by the matrix. Does not check for correct dimensions
template <typename T>
void multiplyMatrixToVector(std::vector<std::vector<T>>& matrix,std::vector<T>& psi) {
    std::vector<T> newpsi(psi.size());
    for (int i = 0; i < matrix.size(); i++)
    {
        T sum;
        for (int j = 0; j < matrix[i].size(); ++j) {
            if(j==0) {
                sum = matrix[i][j] * psi[j];
            }
            else
            {
                sum += matrix[i][j] * psi[j];
            }
        }
        newpsi[i] = sum;
    }
    psi = newpsi;
}


//this function takes in a vector (wavefunction), a matrix, and a result holder
//it measures the expectation value <psi|matrix|psi> of the matrix in the state, psi, and fills the result holder with the result
template <typename T>
void measureExpectation(std::vector<T>& psi, std::vector<std::vector<T>>& matrix, T& result)
{
    std::vector<T> newpsi(psi.size());
    for (int i = 0; i < matrix.size(); i++)
    {
        T sum;
        for (int j = 0; j < matrix[i].size(); ++j) {
            if(j==0) {
                sum = matrix[i][j] * psi[j];
            }
            else
            {
                sum += matrix[i][j] * psi[j];
            }
        }
        newpsi[i] = sum;
    }

    std::vector<T> psidagger(psi.size());
    for(int i =0; i<psi.size();++i)
    {
        psidagger[i] = std::conj(psi[i]);
    }

    for(int i =0; i<psi.size(); ++i)
    {
        if(i==0)
            result = psidagger[i]*newpsi[i];
        else
            result += psidagger[i]*newpsi[i];
    }
}


//this function takes in a gate file and an output stream (for any errors), and for two input states 00 and 11,
//applies the gate specified in the gate file to both states, then measures the states in multiple bases. The results
//of the measurements are returned in a vector. Measurements include all projectional measurements for both two qubit states
//00 and 11, as well as YY, YI, IY measurements for 00 and 11 input states.
std::vector<std::complex<double>> applySimple2qubitSim(const std::string& gateFile, std::ofstream& out)
{
    std::vector<std::complex<double>> psi00 = {{1,0},{0,0},{0,0},{0,0}};
    std::vector<std::complex<double>> psi11 = {{0,0},{0,0},{0,0},{1,0}};
    std::vector<std::vector<std::complex<double>>> gate;
    gate.resize(4);
    std::ifstream inputGate(gateFile);
    if(!inputGate)
    {
        out<<"Error opening file: " <<gateFile<<std::endl;
        return std::vector<std::complex<double>>();
    }
    for(int i=0; i<4; ++i)
    {
        for(int j=0; j<4; ++j)
        {
            std::complex<double> temp;
            inputGate>>temp;
            gate[i].push_back(temp);
        }
    }
    multiplyMatrixToVector(gate,psi00);
    multiplyMatrixToVector(gate,psi11);
    std::vector<std::complex<double>> retVal;
   /* std::cout<<std::endl<<std::endl<<std::endl;
    std::cout<<"================Matrix debug:=============="<<std::endl;
    for(int i(0);i<4;++i) {
        for (int j(0); j < 4; ++j) {
            std::cout<<gate[i][j]<<",   ";
        }
        std::cout<<std::endl;
    }
    std::cout<<std::endl<<std::endl<<std::endl;*/
    for(auto& temp: psi00)
    {
       retVal.push_back(std::abs(temp)*std::abs(temp));
    }
    for(auto& temp: psi11)
    {
        retVal.push_back(std::abs(temp)*std::abs(temp));
    }
    std::vector<std::vector<std::complex<double>>> yy = {{{0,0},{0,0},{0,0},{-1,0}},
                                                         {{0,0},{0,0},{1,0},{0,0}},
                                                         {{0,0},{1,0},{0,0},{0,0}},
                                                         {{-1,0},{0,0},{0,0},{0,0}}};
    std::vector<std::vector<std::complex<double>>> yi = {{{0,0},{0,0},{0,-1},{0,0}},
                                            {{0,0},{0,0},{0,0},{0,-1}},
                                            {{0,1},{0,0},{0,0},{0,0}},
                                            {{0,0},{0,1},{0,0},{0,0}}};

    std::vector<std::vector<std::complex<double>>> iy = {{{0,0},{0,-1},{0,0},{0,0}},
                                            {{0,1},{0,0},{0,0},{0,0}},
                                            {{0,0},{0,0},{0,0},{0,-1}},
                                            {{0,0},{0,0},{0,1},{0,0}}};
    std::complex<double> resYY00;
    std::complex<double> resYY11;
    std::complex<double> resYI00;
    std::complex<double> resYI11;
    std::complex<double> resIY00;
    std::complex<double> resIY11;
    measureExpectation(psi00,yy,resYY00);
    measureExpectation(psi00,yi,resYI00);
    measureExpectation(psi00,iy,resIY00);
    measureExpectation(psi11,yy,resYY11);
    measureExpectation(psi11,yi,resYI11);
    measureExpectation(psi11,iy,resIY11);
    retVal.push_back(resYY00);
    retVal.push_back(resYI00);
    retVal.push_back(resIY00);
    retVal.push_back(resYY11);
    retVal.push_back(resYI11);
    retVal.push_back(resIY11);
    return retVal;
}

//this function is a simple reduce and contract test that checks whether the simple stochastic contraction method is outputting
//correct results. It checks multiple measurements on a test file in the Samples folder: test_JW.qasm
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool simpleReduceAndContractTest(std::ofstream& out)
{
    out<<"Running Simple Reduce and Contract test"<<std::endl<<std::endl;
    bool passed{true};
    int failCount(0);
    std::ofstream generateMeasurement("Samples/measureTest.txt");
    generateMeasurement<<"Z T T T";
    generateMeasurement.close();
    ContractionTools p("Samples/test_JW.qasm", "Samples/measureTest.txt");
   // p.ReduceAndPrintCircuitToVisualGraph("output/visualGraphtest.dgf");
    p.ReduceAndPrintCircuitToTWGraph("Samples/temp.dgf");
    p.Contract(Stochastic);
    if(std::abs(p.GetFinalVal().real()+.620705)>.000001 || std::abs(p.GetFinalVal().imag())>.00001)
    {
       passed= false;
        failCount++;
        out<<"Failed test 1"<<std::endl;
    }


    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"T Z T T";
    generateMeasurement.close();
    ContractionTools p2("Samples/test_JW.qasm", "Samples/measureTest.txt");
   // p2.ReduceAndPrintCircuitToVisualGraph("output/visualGraphtest.dgf");
    p2.ReduceAndPrintCircuitToTWGraph("Samples/temp.dgf");
    p2.Contract(Stochastic);
    if(std::abs(p2.GetFinalVal().real()+.620705)>.000001 || std::abs(p2.GetFinalVal().imag())>.00001)
    {
        passed= false;
        failCount++;
        out<<"Failed test 2"<<std::endl;
    }



    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Y X X Y";
    generateMeasurement.close();
    ContractionTools p3("Samples/test_JW.qasm", "Samples/measureTest.txt");
   // p3.ReduceAndPrintCircuitToVisualGraph("output/visualGraphtest.dgf");
    p3.ReduceAndPrintCircuitToTWGraph("Samples/temp.dgf");
    p3.Contract(Stochastic);
    if(std::abs(p3.GetFinalVal().real()+.784044)>.000001 || std::abs(p3.GetFinalVal().imag())>.00001)
    {
        passed= false;
        failCount++;
        out<<"Failed test 3"<<std::endl;
    }


    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Y X X Y";
    generateMeasurement.close();
    ContractionTools p4("Samples/test_JW.qasm", "Samples/measureTest.txt");
  //  p4.ReduceAndPrintCircuitToVisualGraph("output/visualGraphtest.dgf");
    p4.ReduceAndPrintCircuitToTWGraph("Samples/temp.dgf");
    p4.Contract(Stochastic);
    if(std::abs(p4.GetFinalVal().real()+.784044)>.000001 || std::abs(p4.GetFinalVal().imag())>.00001)
    {
        passed= false;
        failCount++;
        out<<"Failed test 4"<<std::endl;
    }

    out<<"Number of tests failed: "<<failCount<<std::endl;

    removeFile("Samples/measureTest.txt");
    removeFile("Samples/temp.dgf");
    return passed;
}

//this function tests the arbitrary one qubit gate node on a hadamard gate input from the samples folder
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool testArbitraryOneQubit (std::ofstream& out)
{
    out<<"Running Arbitrary One Qubit Gate Test"<<std::endl<<std::endl;

    std::ofstream createTestStream("Samples/test.qasm");
    createTestStream<<"1"<<std::endl;
    createTestStream<<"def1 HadamardTester Samples/hadtest.gate"<<std::endl;
    createTestStream<<"HadamardTester 0"<<std::endl;
    createTestStream.close();
    std::ofstream generateHadamard("Samples/hadtest.gate");
    generateHadamard<<"0.707107 0.707107 0.707107 -0.707107"<<std::endl;
    generateHadamard.close();
    std::ofstream generateMeasurement("Samples/measureTest.txt");
    generateMeasurement<<"0";
    generateMeasurement.close();
    ContractionTools p("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p.Contract(Stochastic);
    out<<"Final Value for |0><0| after Hadamard Gate (read in from file) should be: 1/2 Actual value: "<<p.GetFinalVal()<<std::endl;
    if(std::abs(p.GetFinalVal().real()-0.5)>.00001 || std::abs(p.GetFinalVal().imag())>.00001)
    {
        removeFile("Samples/measureTest.txt");
        removeFile("Samples/test.qasm");
        removeFile("Samples/hadtest.gate");
        return false;
    }


    std::ofstream generateMeasurement2("Samples/measureTest.txt");
    generateMeasurement2<<"1";
    generateMeasurement2.close();
    ContractionTools p2("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p2.Contract(Stochastic);
    out<<"Final Value for |1><1| after Hadamard Gate (read in from file) should be: 1/2 Actual value: "<<p2.GetFinalVal()<<std::endl;
    if(std::abs(p2.GetFinalVal().real()-0.5)>.00001 || std::abs(p2.GetFinalVal().imag())>.00001)
    {
        removeFile("Samples/measureTest.txt");
        removeFile("Samples/test.qasm");
        removeFile("Samples/hadtest.gate");
        return false;
    }




    createTestStream.open("Samples/test.qasm");
    createTestStream<<"1"<<std::endl;
    createTestStream<<"def1 HadamardTester Samples/hadtest.gate"<<std::endl;
    createTestStream<<"X 0"<<std::endl;
    createTestStream<<"HadamardTester 0"<<std::endl;
    createTestStream.close();
    generateHadamard.open("Samples/hadtest.gate");
    generateHadamard<<"0.707107 0.707107 0.707107 -0.707107"<<std::endl;
    generateHadamard.close();
    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0";
    generateMeasurement.close();
    ContractionTools p3("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p3.Contract(Stochastic);
    out<<"Final Value for |0><0| after Hadamard Gate (read in from file) should be: 1/2 Actual value: "<<p3.GetFinalVal()<<std::endl;
    if(std::abs(p3.GetFinalVal().real()-0.5)>.00001 || std::abs(p3.GetFinalVal().imag())>.00001)
    {
        removeFile("Samples/measureTest.txt");
        removeFile("Samples/test.qasm");
        removeFile("Samples/hadtest.gate");
        return false;
    }


    generateMeasurement2.open("Samples/measureTest.txt");
    generateMeasurement2<<"1";
    generateMeasurement2.close();
    ContractionTools p4("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p4.Contract(Stochastic);
    out<<"Final Value for |1><1| after Hadamard Gate (read in from file) should be: 1/2 Actual value: "<<p4.GetFinalVal()<<std::endl;
    if(std::abs(p4.GetFinalVal().real()-0.5)>.00001 || std::abs(p4.GetFinalVal().imag())>.00001)
    {
        removeFile("Samples/measureTest.txt");
        removeFile("Samples/test.qasm");
        removeFile("Samples/hadtest.gate");
        return false;
    }

    removeFile("Samples/measureTest.txt");
    removeFile("Samples/test.qasm");
    removeFile("Samples/hadtest.gate");
    return true;
}


//this test runs both stochastic and LineGraph contraction tests on large circuits to check threading and correct results
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool largeCircuitTest(std::ofstream& out)
{
    out<<"Running Large Circuit Test"<<std::endl<<std::endl;
    try {
        ContractionTools c("Samples/4regRand20Node1-p1.qasm","Samples/measure125.txt");
        c.Contract(Stochastic);
        out<<"Contraction Method: Stochastic; Measurement Result: "<<c.GetFinalVal()<<"; Expected Result: (0.0035757,0)"<<std::endl;
        if(std::abs(c.GetFinalVal().real()-0.0035757)>= 0.00001 || c.GetFinalVal().imag() >= 0.00001)
        {
            return false;
        }

        c.Reset("Samples/4regRand20Node5-p1.qasm","Samples/measure125.txt");
        c.Contract(Stochastic);
        out<<"Contraction Method: Stochastic; Measurement Result: "<<c.GetFinalVal()<<"; Expected Result: (0.00255591,0)"<<std::endl;
        if(std::abs(c.GetFinalVal().real()-0.00255591)>= 0.00001 || c.GetFinalVal().imag()>= 0.00001)
        {
            return false;
        }

        std::shared_ptr<Network> temp1 = std::make_shared<Network>("Samples/4regRand20Node1-p1.qasm","Samples/measure125.txt");
        std::shared_ptr<Network> temp2 = std::make_shared<Network>("Samples/4regRand20Node5-p1.qasm","Samples/measure125.txt");
        LineGraph lg1(temp1);
        LineGraph lg2(temp2);
        lg1.runQuickBB(20);
        lg1.LGContract();
        out<<"Contraction Method: LineGraph; Measurement Result: "<<temp1->GetFinalValue()<<"; Expected Result: (0.0035757,0)"<<std::endl;
        if(std::abs(temp1->GetFinalValue().real()-0.0035757)>= 0.00001 || temp1->GetFinalValue().imag()>= 0.00001)
        {
            return false;
        }

        lg2.runQuickBB(20);
        lg2.LGContract();
        out<<"Contraction Method: LineGraph; Measurement Result: "<<temp2->GetFinalValue()<<"; Expected Result: (0.00255591,0)"<<std::endl;
        if(std::abs(temp2->GetFinalValue().real()-0.00255591)>= 0.00001 || temp2->GetFinalValue().imag()>= 0.00001)
        {
            return false;
        }
    }
    catch(std::exception& e)
    {
        out<<e.what()<<std::endl;
        return false;
    }

    return true;
}

//this function tests the Rx, Ry, and Rz nodes
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool testRotationCircuits (std::ofstream& out)
{
    out<<"Running Rotation Circuit tests"<<std::endl<<std::endl;
    int failed(0);
    bool passed{true};
    //circuit 1
    std::ofstream createTestStream("Samples/test.qasm");
    createTestStream<<"1"<<std::endl;
    createTestStream<<"Ry 0.1 0"<<std::endl;
    createTestStream.close();
    std::ofstream generateMeasurement("Samples/measureTest.txt");
    generateMeasurement<<"X";
    generateMeasurement.close();
    ContractionTools p("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p.Contract(Stochastic);
    if(std::abs(p.GetFinalVal().real()-0.0998334)>.00001 || std::abs(p.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 1"<<std::endl;
        failed++;
        passed=false;
    }


    std::ofstream generateMeasurement2("Samples/measureTest.txt");
    generateMeasurement2<<"Y";
    generateMeasurement2.close();
    ContractionTools p2("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p2.Contract(Stochastic);
    if(std::abs(p2.GetFinalVal().real())>.00001 || std::abs(p2.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 2"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Z";
    generateMeasurement.close();
    ContractionTools p3("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p3.Contract(Stochastic);
    if(std::abs(p3.GetFinalVal().real()-0.995004)>.00001 || std::abs(p3.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 3"<<std::endl;
        failed++;
        passed=false;
    }




    //circuit 2
    createTestStream.open("Samples/test.qasm");
    createTestStream<<"1"<<std::endl;
    createTestStream<<"Ry -0.1 0"<<std::endl;
    createTestStream.close();
    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"X";
    generateMeasurement.close();
    ContractionTools p4("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p4.Contract(Stochastic);
    if(std::abs(p4.GetFinalVal().real()+0.0998334166468)>.00001 || std::abs(p4.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 4"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Y";
    generateMeasurement.close();
    ContractionTools p5("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p5.Contract(Stochastic);
    if(std::abs(p5.GetFinalVal().real())>.00001 || std::abs(p5.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 5"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Z";
    generateMeasurement.close();
    ContractionTools p6("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p6.Contract(Stochastic);
    if(std::abs(p6.GetFinalVal().real()-0.995004165278)>.00001 || std::abs(p6.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 6"<<std::endl;
        failed++;
        passed=false;
    }


    //circuit 3

    createTestStream.open("Samples/test.qasm");
    createTestStream<<"1"<<std::endl;
    createTestStream<<"H 0"<<std::endl;
    createTestStream<<"Rz 0.1 0"<<std::endl;
    createTestStream.close();
    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"X";
    generateMeasurement.close();
    ContractionTools p7("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p7.Contract(Stochastic);
    if(std::abs(p7.GetFinalVal().real()-0.995004165278)>.00001 || std::abs(p7.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 7"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Y";
    generateMeasurement.close();
    ContractionTools p8("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p8.Contract(Stochastic);
    if(std::abs(p8.GetFinalVal().real()-0.0998334166468)>.00001 || std::abs(p8.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 8"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Z";
    generateMeasurement.close();
    ContractionTools p9("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p9.Contract(Stochastic);
    if(std::abs(p9.GetFinalVal().real())>.00001 || std::abs(p9.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 9"<<std::endl;
        failed++;
        passed=false;
    }


    //circuit 4

    createTestStream.open("Samples/test.qasm");
    createTestStream<<"1"<<std::endl;
    createTestStream<<"H 0"<<std::endl;
    createTestStream<<"Rz -0.1 0"<<std::endl;
    createTestStream.close();
    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"X";
    generateMeasurement.close();
    ContractionTools p10("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p10.Contract(Stochastic);
    if(std::abs(p10.GetFinalVal().real()-0.995004165278)>.00001 || std::abs(p10.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 10"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Y";
    generateMeasurement.close();
    ContractionTools p11("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p11.Contract(Stochastic);
    if(std::abs(p11.GetFinalVal().real()+0.0998334166468)>.00001 || std::abs(p11.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 11"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Z";
    generateMeasurement.close();
    ContractionTools p12("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p12.Contract(Stochastic);
    if(std::abs(p12.GetFinalVal().real())>.00001 || std::abs(p12.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 12"<<std::endl;
        failed++;
        passed=false;
    }


    //circuit 5

    createTestStream.open("Samples/test.qasm");
    createTestStream<<"1"<<std::endl;
    createTestStream<<"Rx 0.1 0"<<std::endl;
    createTestStream.close();
    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"X";
    generateMeasurement.close();
    ContractionTools p13("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p13.Contract(Stochastic);
    if(std::abs(p13.GetFinalVal().real())>.00001 || std::abs(p13.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 13"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Y";
    generateMeasurement.close();
    ContractionTools p14("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p14.Contract(Stochastic);
    if(std::abs(p14.GetFinalVal().real()+0.0998334166468)>.00001 || std::abs(p14.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 14"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Z";
    generateMeasurement.close();
    ContractionTools p15("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p15.Contract(Stochastic);
    if(std::abs(p15.GetFinalVal().real()-0.995004165278)>.00001 || std::abs(p15.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 15"<<std::endl;
        failed++;
        passed=false;
    }


    //circuit 6

    createTestStream.open("Samples/test.qasm");
    createTestStream<<"1"<<std::endl;
    createTestStream<<"Rx -0.1 0"<<std::endl;
    createTestStream.close();
    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"X";
    generateMeasurement.close();
    ContractionTools p16("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p16.Contract(Stochastic);
    if(std::abs(p16.GetFinalVal().real())>.00001 || std::abs(p16.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 16"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Y";
    generateMeasurement.close();
    ContractionTools p17("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p17.Contract(Stochastic);
    if(std::abs(p17.GetFinalVal().real()-0.0998334166468)>.00001 || std::abs(p17.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 17"<<std::endl;
        failed++;
        passed=false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"Z";
    generateMeasurement.close();
    ContractionTools p18("Samples/test.qasm","Samples/measureTest.txt");
    //contract and compare
    p18.Contract(Stochastic);
    if(std::abs(p18.GetFinalVal().real()-0.995004165278)>.00001 || std::abs(p18.GetFinalVal().imag())>.00001)
    {
        out<<"Did not pass test 18"<<std::endl;
        failed++;
        passed=false;
    }

    out<<"Number of tests failed: "<<failed<<std::endl;

    removeFile("Samples/measureTest.txt");
    removeFile("Samples/test.qasm");
    return passed;
}



//this function tests the user defined sequence method 10 times.
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool testUserDefinedSequence (std::ofstream& out)
{
    std::ofstream outputSequence;
    std::ofstream tempMeasure("Samples/tempmeasure.txt");
    if(!tempMeasure)
    {
        out<<"Failed to open measure file - make sure Samples folder exists"<<std::endl;
        return false;
    }
    tempMeasure<<"T";
    tempMeasure.close();
    for(int i = 0; i<10; ++i)
    {
        outputSequence.open("Samples/outputseqtemp.txt");
        if(!outputSequence)
        {
            out<<"Failed to open output sequence file  - make sure Samples folder exists"<<std::endl;
            removeFile("Samples/outputseqtemp.txt");
            removeFile("Samples/tempmeasure.txt");
            return false;
        }
        std::shared_ptr<Network> tempptr = std::make_shared<Network>("Samples/test_JW.qasm","Samples/tempmeasure.txt");
        int allNodesSize = tempptr->GetAllNodes().size();
        ContractionTools c(tempptr);
        tempptr = c.Contract(Stochastic);
        std::vector<std::pair<int,int>> tempOrdering;
        std::vector<std::vector<bool>> alreadyAdded(tempptr->GetAllNodes().size(),std::vector<bool>(tempptr->GetAllNodes().size(),false));
        std::unordered_map<int,int> mapToOriginalNode;
        for(const auto& node: tempptr->GetAllNodes())
        {
            if(!(node->mCreatedFrom.first==0 && node->mCreatedFrom.second==0))
            {

                    int temp1 = node->mCreatedFrom.first;
                    int temp2 = node->mCreatedFrom.second;
                    while(mapToOriginalNode.find(temp1)!=mapToOriginalNode.end() || mapToOriginalNode.find(temp2)!=mapToOriginalNode.end())
                    {
                        if(mapToOriginalNode.find(temp1)!=mapToOriginalNode.end())
                        {
                            temp1 = mapToOriginalNode[temp1];
                        }
                        if(mapToOriginalNode.find(temp2)!=mapToOriginalNode.end())
                        {
                            temp2 = mapToOriginalNode[temp2];
                        }
                    }
                    if(alreadyAdded[temp1][temp2]==true || temp1>=allNodesSize || temp2>=allNodesSize)
                    {
                        continue;
                    }
                    else
                    {
                        tempOrdering.push_back({temp1, temp2});
                        alreadyAdded[temp1][temp2] = true;
                        alreadyAdded[temp2][temp1] = true;
                        mapToOriginalNode.insert({node->mID, node->mCreatedFrom.first});
                    }

            }
        }
        for(const auto& pair: tempOrdering)
        {
            outputSequence<<pair.first<<" "<<pair.second<<std::endl;
        }
        outputSequence.close();
        c.Reset("Samples/test_JW.qasm","Samples/tempmeasure.txt");
        try {
            out<<"Contracting arbitrary sequence for trial #: "<<i+1<<std::endl;
            c.ContractUserDefinedSequenceOfWires("Samples/outputseqtemp.txt");
        }
        catch(std::exception& e)
        {
            out<<e.what()<<std::endl;
            removeFile("Samples/outputseqtemp.txt");
            removeFile("Samples/tempmeasure.txt");
            return false;
        }
    }
    removeFile("Samples/outputseqtemp.txt");
    removeFile("Samples/tempmeasure.txt");
    return true;
}

//this function tests the linegraph contraction method
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool testLineGraph(std::ofstream& out)
{
    out<<"Running LineGraph test"<<std::endl<<std::endl;



    std::shared_ptr<Network> netw;

    std::ofstream outputMeasure("Samples/measureTest.txt");
    outputMeasure<<"T";
    outputMeasure.close();

    try {
        netw = std::make_shared<Network>("Samples/test_JW.qasm", "Samples/measureTest.txt");
    }
    catch(std::exception& e)
    {
        std::cout<<e.what()<<std::endl;
        out<<e.what()<<std::endl<<std::endl;
        removeFile("Samples/measureTest.txt");
        return false;
    }



    out << "Network has been made." << std::endl;

    out << netw->GetAllNodes().size() << std::endl;
    //std::cout << netw->GetCurrentNodes().size() << std::endl;
    out << netw->GetUncontractedNodes().size() << std::endl;

    //netw->ReduceCircuit();

    //std::cout << netw->GetCurrentNodes().size() << std::endl;


    out << "Reducing circuit" << std::endl;
    try {
        netw->ReduceCircuit();
    }
    catch(std::exception& e)
    {
        std::cout<<e.what()<<std::endl;
        out<<e.what()<<std::endl<<std::endl;
        removeFile("Samples/measureTest.txt");
        return false;
    }


    out << netw->GetAllNodes().size() << std::endl;
    // std::cout << netw->GetCurrentNodes().size() << std::endl;
    out << netw->GetUncontractedNodes().size() << std::endl;


    LineGraph lg(netw);

    try
    {
        lg.runQuickBB(20);
        lg.LGContract();
    }
    catch(std::exception& e)
    {
        std::cout<<e.what()<<std::endl;
        out<<e.what()<<std::endl<<std::endl;
        removeFile("Samples/measureTest.txt");
        return false;
    }


    return true;

}

//this function tests the arbitrary two qubit gate node on the randomly generated unitaries in Samples/arbitrary2qubitmatrixset
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool testArbitraryTwoQubit (std::ofstream& out)
{
    out<<"Running Arbitrary Two Qubit test"<<std::endl<<std::endl;
    for(int i=0; i<100; i++)
    {
        out<<"Testing matrix: matrix"<<i<<".gate"<<std::endl<<std::endl;
        std::string inPath("Samples/arbitrary2qubitmatrixset/matrix");
        inPath+=std::to_string(i);
        inPath+=".gate";
        std::vector<std::complex<double>> answers(applySimple2qubitSim(inPath,out));
        if(answers.size()==0)
        {
            out<<"Failed to open file "<<std::endl;
            return false;
        }
        std::ofstream outputQasm("Samples/tempQASM.qasm");
        std::ofstream outputMeasure("Samples/testMeasure.txt");
        outputQasm<<2<<std::endl<<"def2 TEST "<<inPath<<std::endl;
        outputQasm<<"TEST 0 1"<<std::endl;
        outputQasm.close();
        outputMeasure<<"0 0";
        outputMeasure.close();
        ContractionTools c("Samples/tempQASM.qasm","Samples/testMeasure.txt");
        c.Contract(Stochastic);
        out<<"Result of tensor contraction (input 00, measure: 00): "<<c.GetFinalVal()<<"  Expected: "<<answers[0]<<std::endl;
        if(std::abs((c.GetFinalVal()-answers[0]).real())>0.00001 || std::abs((c.GetFinalVal()-answers[0]).imag())>0.00001 )
        {
            removeFile("Samples/tempQASM.qasm");
            removeFile("Samples/testMeasure.txt");
            return false;
        }
        outputMeasure.open("Samples/testMeasure.txt");
        outputMeasure<<"1 1";
        outputMeasure.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction (input 00, measure: 11): "<<c.GetFinalVal()<<"  Expected: "<<answers[3]<<std::endl;
        if(std::abs((c.GetFinalVal()-answers[3]).real())>0.00001 || std::abs((c.GetFinalVal()-answers[3]).imag())>0.00001 )
        {
            removeFile("Samples/tempQASM.qasm");
            removeFile("Samples/testMeasure.txt");
            return false;
        }

        outputQasm.open("Samples/tempQASM.qasm");
        outputQasm<<2<<std::endl;
        outputQasm<<"X 0"<<std::endl<<"X 1"<<std::endl;
        outputQasm<<"def2 TEST "<<inPath<<std::endl;
        outputQasm<<"TEST 0 1"<<std::endl;
        outputQasm.close();
        outputMeasure.open("Samples/testMeasure.txt");
        outputMeasure<<"0 0";
        outputMeasure.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction (input 11, measure: 00): "<<c.GetFinalVal()<<"  Expected: "<<answers[4]<<std::endl;
        if(std::abs((c.GetFinalVal()-answers[4]).real())>0.00001 || std::abs((c.GetFinalVal()-answers[4]).imag())>0.00001 )
        {
            removeFile("Samples/tempQASM.qasm");
            removeFile("Samples/testMeasure.txt");
            return false;
        }
        outputMeasure.open("Samples/testMeasure.txt");
        outputMeasure<<"1 1";
        outputMeasure.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction (input 11, measure: 11): "<<c.GetFinalVal()<<"  Expected: "<<answers[7]<<std::endl;
        if(std::abs((c.GetFinalVal()-answers[7]).real())>0.00001 || std::abs((c.GetFinalVal()-answers[7]).imag())>0.00001 )
        {
            removeFile("Samples/tempQASM.qasm");
            removeFile("Samples/testMeasure.txt");
            return false;
        }




        //TEST Y MEASURE
        //INPUT: 00

        outputQasm.open("Samples/tempQASM.qasm");
        outputQasm<<2<<std::endl;
        outputQasm<<"def2 TEST "<<inPath<<std::endl;
        outputQasm<<"TEST 0 1"<<std::endl;
        outputQasm.close();
        outputMeasure.open("Samples/testMeasure.txt");
        outputMeasure<<"Y Y";
        outputMeasure.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction (input 00, measure: YY): "<<c.GetFinalVal()<<"  Expected: "<<answers[8]<<std::endl;
        if(std::abs((c.GetFinalVal()-answers[8]).real())>0.00001 || std::abs((c.GetFinalVal()-answers[8]).imag())>0.00001 )
        {
            removeFile("Samples/tempQASM.qasm");
            removeFile("Samples/testMeasure.txt");
            return false;
        }

        outputMeasure.open("Samples/testMeasure.txt");
        outputMeasure<<"Y T";
        outputMeasure.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction (input 00, measure: YI): "<<c.GetFinalVal()<<"  Expected: "<<answers[9]<<std::endl;
        if(std::abs((c.GetFinalVal()-answers[9]).real())>0.00001 || std::abs((c.GetFinalVal()-answers[9]).imag())>0.00001 )
        {
            removeFile("Samples/tempQASM.qasm");
            removeFile("Samples/testMeasure.txt");
            return false;
        }
        outputMeasure.open("Samples/testMeasure.txt");
        outputMeasure<<"T Y";
        outputMeasure.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction (input 00, measure: IY): "<<c.GetFinalVal()<<"  Expected: "<<answers[10]<<std::endl;
        if(std::abs((c.GetFinalVal()-answers[10]).real())>0.00001 || std::abs((c.GetFinalVal()-answers[10]).imag())>0.00001 )
        {
            removeFile("Samples/tempQASM.qasm");
            removeFile("Samples/testMeasure.txt");
            return false;
        }



        //INPUT: 11
        outputQasm.open("Samples/tempQASM.qasm");
        outputQasm<<2<<std::endl;
        outputQasm<<"X 0"<<std::endl<<"X 1"<<std::endl;
        outputQasm<<"def2 TEST "<<inPath<<std::endl;
        outputQasm<<"TEST 0 1"<<std::endl;
        outputQasm.close();
        outputMeasure.open("Samples/testMeasure.txt");
        outputMeasure<<"Y Y";
        outputMeasure.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction (input 11, measure: YY): "<<c.GetFinalVal()<<"  Expected: "<<answers[11]<<std::endl;
        if(std::abs((c.GetFinalVal()-answers[11]).real())>0.00001 || std::abs((c.GetFinalVal()-answers[11]).imag())>0.00001 )
        {
            removeFile("Samples/tempQASM.qasm");
            removeFile("Samples/testMeasure.txt");
            return false;
        }
        outputMeasure.open("Samples/testMeasure.txt");
        outputMeasure<<"Y T";
        outputMeasure.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction (input 11, measure: YI): "<<c.GetFinalVal()<<"  Expected: "<<answers[12]<<std::endl;
        if(std::abs((c.GetFinalVal()-answers[12]).real())>0.00001 || std::abs((c.GetFinalVal()-answers[12]).imag())>0.00001 )
        {
            removeFile("Samples/tempQASM.qasm");
            removeFile("Samples/testMeasure.txt");
            return false;
        }
        outputMeasure.open("Samples/testMeasure.txt");
        outputMeasure<<"T Y";
        outputMeasure.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction (input 11, measure: IY): "<<c.GetFinalVal()<<"  Expected: "<<answers[13]<<std::endl;
        if(std::abs((c.GetFinalVal()-answers[13]).real())>0.00001 || std::abs((c.GetFinalVal()-answers[13]).imag())>0.00001 )
        {
            removeFile("Samples/tempQASM.qasm");
            removeFile("Samples/testMeasure.txt");
            return false;
        }





        out<<std::endl;
    }
    removeFile("Samples/tempQASM.qasm");
    removeFile("Samples/testMeasure.txt");
    return true;
}

//This function generates a bell pair and runs multiple measurements to test for correctness
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool bellTest(std::ofstream& out)
{
    out<<"Running Bell Pair test"<<std::endl<<std::endl;

    //input: 00
    std::ofstream generateMeasurement("Samples/measureTest.txt");
    generateMeasurement<<"0 0";
    generateMeasurement.close();
    ContractionTools c("Samples/bell_pair.qasm","Samples/measureTest.txt");
    c.Contract(Stochastic);
    out<<"Input: |00>; Measurement: 00; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0.5,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0 1";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |00>; Measurement: 01; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 0";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |00>; Measurement: 10; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 1";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |00>; Measurement: 11; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0.5,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }


    //input: 01

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0 0";
    generateMeasurement.close();
    std::string newPath(generateQASMWithDiffPureInputState("Samples/bell_pair.qasm",{0,1}));
    c.Reset(newPath,"Samples/measureTest.txt");
    c.Contract(Stochastic);
    out<<"Input: |01>; Measurement: 00; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0 1";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |01>; Measurement: 01; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0.5,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 0";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |01>; Measurement: 10; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0.5,0)).real())>0.00001)
    {
        return false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 1";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |01>; Measurement: 11; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }
    removeFile(newPath);



    //input: 10

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0 0";
    generateMeasurement.close();
    newPath = generateQASMWithDiffPureInputState("Samples/bell_pair.qasm",{1,0});
    c.Reset(newPath,"Samples/measureTest.txt");
    c.Contract(Stochastic);
    out<<"Input: |10>; Measurement: 00; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0.5,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0 1";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |10>; Measurement: 01; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }


    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 0";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |10>; Measurement: 10; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }


    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 1";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |10>; Measurement: 11; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0.5,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }
    removeFile(newPath);



    //input: 11

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0 0";
    generateMeasurement.close();
    newPath = generateQASMWithDiffPureInputState("Samples/bell_pair.qasm",{1,1});
    c.Reset(newPath,"Samples/measureTest.txt");
    c.Contract(Stochastic);
    out<<"Input: |11>; Measurement: 00; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0 1";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |11>; Measurement: 01; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0.5,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }


    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 0";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |11>; Measurement: 10; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0.5,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }


    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 1";
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    out<<"Input: |11>; Measurement: 11; Probability: "<<c.GetFinalVal().real()<<std::endl;
    if(std::abs((c.GetFinalVal() - std::complex<double>(0,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }
    removeFile(newPath);
    removeFile("Samples/measureTest.txt");
    return true;

}

//this function generates both an 8 qubit cat state and a 100 qubit cat state and checks for correctness with specific measurements
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool catTest(std::ofstream& out)
{

    out<<"Running Cat State test"<<std::endl<<std::endl;
    std::ofstream generateMeasurement;
    ContractionTools c("Samples/catStateEightQubits.qasm","Samples/measureTest.txt");
    out<<"Checking 8 qubit cat state..."<<std::endl<<std::endl;

    c.Reset();
    generateMeasurement.open("Samples/measureTest.txt");
    bool temp = false;
    std::stringstream ss;
    for(int i =0; i<8; ++i) {
        temp ? generateMeasurement << "1 ": generateMeasurement<<"0 ";
        temp ? ss << "1": ss<<"0";
    }
    generateMeasurement.close();

    c.Contract(Stochastic);
    out << "Input: |00000000>; Measurement: ";
    out<<ss.str()<<" ; Probability: " << c.GetFinalVal().real() << std::endl;
    if (std::abs((c.GetFinalVal() - std::complex<double>(0.5, 0)).real()) > 0.00001) {
        removeFile("Samples/measureTest.txt");
        return false;
    }

    c.Reset();
    generateMeasurement.open("Samples/measureTest.txt");
    temp = true;
    ss = std::stringstream();
    for(int i =0; i<8; ++i) {
        temp ? generateMeasurement << "1 ": generateMeasurement<<"0 ";
        temp ? ss << "1": ss<<"0";
    }
    generateMeasurement.close();

    c.Contract(Stochastic);
    out << "Input: |00000000>; Measurement: ";
    out<<ss.str()<<" ; Probability: " << c.GetFinalVal().real() << std::endl;
    if (std::abs((c.GetFinalVal() - std::complex<double>(0.5, 0)).real()) > 0.00001) {
        removeFile("Samples/measureTest.txt");
        return false;
    }


    out<<std::endl;
    out<<"Checking 100 qubit cat state circuit..."<<std::endl<<std::endl;
    c.Reset("Samples/catState100Qubits.qasm","Samples/measureTest.txt");
    temp = false;
    generateMeasurement.open("Samples/measureTest.txt");
    for(int i(0);i<100;++i) {
        temp ? generateMeasurement << "1 ": generateMeasurement<<"0 ";
    }
    generateMeasurement.close();
    c.Contract(Stochastic);
    if(!temp) {
        out << "Input: |0> tensor 100 times; Measurement: ";
        out << "All Zeros"<<" ; Probability: " << c.GetFinalVal().real() << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(0.5, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTest.txt");
            return false;
        }

    }


    temp = true;
    generateMeasurement.open("Samples/measureTest.txt");
    for(int i(0);i<100;++i) {
        temp ? generateMeasurement << "1 ": generateMeasurement<<"0 ";
    }
    generateMeasurement.close();
    c.Reset();
    c.Contract(Stochastic);
    if(temp) {
        out << "Input: |0> tensor 100 times; Measurement: ";
        out<<"All Ones"<<" ; Probability: " << c.GetFinalVal().real() << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(0.5, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTest.txt");
            return false;
        }
    }

    removeFile("Samples/measureTest.txt");
    return true;
}

//this function creates a quantum teleportation circuit and teleports arbitrary 1 qubit states, checking for correctness with projectional measurements
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool teleporationTest(std::ofstream& out)
{
    out<<"Running Teleportation Circuit test"<<std::endl<<std::endl;
    std::random_device randomDevice;
    std::mt19937 randGen(randomDevice());
    std::uniform_real_distribution<double> dist(-1,1);
    for(int i=0; i<10; i++) {
        double temp1 = dist(randGen);
        double temp2 = dist(randGen);
        double temp3 = dist(randGen);
        double temp4 = dist(randGen);
        double magnitude = sqrt(temp1*temp1 + temp2 *temp2 + temp3*temp3 + temp4 * temp4);
        std::complex<double> alpha(temp1/magnitude, temp2/magnitude);
        std::complex<double> beta(temp3/magnitude,temp4/magnitude);
        double normalization = 4*std::abs(alpha)+4*std::abs(beta);
        out<<"Teleporting State: "<<alpha<<" |0> + "<<beta<<" |1>"<<std::endl<<std::endl;
        std::ofstream gateOutput("Samples/testTeleportGate.gate");
        gateOutput <<alpha<<" "<<0<<" "<<beta<<" "<<0;
        gateOutput.close();
        std::ofstream qasmOut("Samples/teleportTest.qasm");
        qasmOut<<3<<std::endl;
        qasmOut<<"def1 AB Samples/testTeleportGate.gate"<<std::endl<<"AB 0"<<std::endl;
        qasmOut<<"H 1\n"
                <<"CNOT 1 2\n"
                <<"CNOT 0 1\n"
                <<"H 0\n";
        qasmOut.close();
        std::ofstream measureOut("Samples/tempMeasure.txt");
        measureOut<<"0 0 0";
        measureOut.close();
        ContractionTools c("Samples/teleportTest.qasm","Samples/tempMeasure.txt");
        c.Contract(Stochastic);
        out<<"Result of tensor contraction: "<<c.GetFinalVal()<<"  Expected: "<<std::abs(alpha/2.0)*std::abs(alpha/2.0)<<std::endl;
        if(std::abs((c.GetFinalVal().real()- std::abs(alpha/2.0)*std::abs(alpha/2.0)))>0.00001)
        {
            removeFile("Samples/testTeleportGate.gate");
            removeFile("Samples/tempMeasure.txt");
            removeFile("Samples/teleportTest.qasm");
            return false;
        }

        measureOut.open("Samples/tempMeasure.txt");
        measureOut<<"0 0 1";
        measureOut.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction: "<<c.GetFinalVal()<<"  Expected: "<<std::abs(beta/2.0)*std::abs(beta/2.0)<<std::endl;
        if(std::abs((c.GetFinalVal().real()- std::abs(beta/2.0)*std::abs(beta/2.0)))>0.00001)
        {
            removeFile("Samples/testTeleportGate.gate");
            removeFile("Samples/tempMeasure.txt");
            removeFile("Samples/teleportTest.qasm");
            return false;
        }

        measureOut.open("Samples/tempMeasure.txt");
        measureOut<<"1 0 0";
        measureOut.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction: "<<c.GetFinalVal()<<"  Expected: "<<std::abs(alpha/2.0)*std::abs(alpha/2.0)<<std::endl;
        if(std::abs((c.GetFinalVal().real()- std::abs(alpha/2.0)*std::abs(alpha/2.0)))>0.00001)
        {
            removeFile("Samples/testTeleportGate.gate");
            removeFile("Samples/tempMeasure.txt");
            removeFile("Samples/teleportTest.qasm");
            return false;
        }

        measureOut.open("Samples/tempMeasure.txt");
        measureOut<<"1 0 1";
        measureOut.close();
        c.Reset();
        c.Contract(Stochastic);
        out<<"Result of tensor contraction: "<<c.GetFinalVal()<<"  Expected: "<<std::abs(beta/2.0)*std::abs(beta/2.0)<<std::endl;
        if(std::abs((c.GetFinalVal().real()- std::abs(beta/2.0)*std::abs(beta/2.0)))>0.00001)
        {
            removeFile("Samples/testTeleportGate.gate");
            removeFile("Samples/tempMeasure.txt");
            removeFile("Samples/teleportTest.qasm");
            return false;
        }



        out<<std::endl;


    }
    removeFile("Samples/testTeleportGate.gate");
    removeFile("Samples/tempMeasure.txt");
    removeFile("Samples/teleportTest.qasm");
    return true;
}

//this function creates a toffoli gate circuit out of 1 and 2 qubit gates and runs the circuit on all possible 3 qubit inputs
//it checks for correctness with projectional measurements
//returns true on success or false on failure
//input ofstream is for printing errors/results
bool tofolliTest(std::ofstream& out)
{
    out<<"Running Toffoli Gate test"<<std::endl<<std::endl;

    //input: 000
    std::ofstream generateMeasurement("Samples/measureTest.txt");
    generateMeasurement<<"0 0 0";
    generateMeasurement.close();
    ContractionTools c("Samples/tofolli.qasm","Samples/measureTest.txt");
    c.Contract(Stochastic);
    if(std::abs((c.GetFinalVal() - std::complex<double>(1,0)).real())>0.00001)
    {
        removeFile("Samples/measureTest.txt");
        return false;
    }
    out<<"Input: |000>; Measurement: 000; Probability: "<<c.GetFinalVal().real()<<std::endl;

    //input: 001

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0 0 1";
    generateMeasurement.close();
    std::string newPath(generateQASMWithDiffPureInputState("Samples/tofolli.qasm",{0,0,1}));
    c.Reset(newPath,"Samples/measureTest.txt");
    c.Contract(Stochastic);
    if(std::abs((c.GetFinalVal() - std::complex<double>(1,0)).real())>0.00001)
    {
        removeFile(newPath);
        removeFile("Samples/measureTest.txt");
        return false;
    }
    out<<"Input: |001>; Measurement: 001; Probability: "<<c.GetFinalVal().real()<<std::endl;
    removeFile(newPath);


    //input: 010

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0 1 0";
    generateMeasurement.close();
    newPath = generateQASMWithDiffPureInputState("Samples/tofolli.qasm",{0,1,0});
    c.Reset(newPath,"Samples/measureTest.txt");
    c.Contract(Stochastic);
    if(std::abs((c.GetFinalVal() - std::complex<double>(1,0)).real())>0.00001)
    {
        removeFile(newPath);
        removeFile("Samples/measureTest.txt");
        return false;
    }
    out<<"Input: |010>; Measurement: 010; Probability: "<<c.GetFinalVal().real()<<std::endl;
    removeFile(newPath);


    //input: 011

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"0 1 1";
    generateMeasurement.close();
    newPath = generateQASMWithDiffPureInputState("Samples/tofolli.qasm",{0,1,1});
    c.Reset(newPath,"Samples/measureTest.txt");
    c.Contract(Stochastic);
    if(std::abs((c.GetFinalVal() - std::complex<double>(1,0)).real())>0.00001)
    {
        removeFile(newPath);
        removeFile("Samples/measureTest.txt");
        return false;
    }
    out<<"Input: |011>; Measurement: 011; Probability: "<<c.GetFinalVal().real()<<std::endl;
    removeFile(newPath);
    //input: 100

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 0 0";
    generateMeasurement.close();
    newPath = generateQASMWithDiffPureInputState("Samples/tofolli.qasm",{1,0,0});
    c.Reset(newPath,"Samples/measureTest.txt");
    c.Contract(Stochastic);
    if(std::abs((c.GetFinalVal() - std::complex<double>(1,0)).real())>0.00001)
    {
        removeFile(newPath);
        removeFile("Samples/measureTest.txt");
        return false;
    }
    out<<"Input: |100>; Measurement: 100; Probability: "<<c.GetFinalVal().real()<<std::endl;
    removeFile(newPath);
    //input: 101

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 0 1";
    generateMeasurement.close();
    newPath = generateQASMWithDiffPureInputState("Samples/tofolli.qasm",{1,0,1});
    c.Reset(newPath,"Samples/measureTest.txt");
    c.Contract(Stochastic);
    if(std::abs((c.GetFinalVal() - std::complex<double>(1,0)).real())>0.00001)
    {
        removeFile(newPath);
        removeFile("Samples/measureTest.txt");
        return false;
    }
    out<<"Input: |101>; Measurement: 101; Probability: "<<c.GetFinalVal().real()<<std::endl;
    removeFile(newPath);
    //input: 110

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 1 1";
    generateMeasurement.close();
    newPath = generateQASMWithDiffPureInputState("Samples/tofolli.qasm",{1,1,0});
    c.Reset(newPath,"Samples/measureTest.txt");
    c.Contract(Stochastic);
    if(std::abs((c.GetFinalVal() - std::complex<double>(1,0)).real())>0.00001)
    {
        removeFile(newPath);
        removeFile("Samples/measureTest.txt");
        return false;
    }
    out<<"Input: |110>; Measurement: 111; Probability: "<<c.GetFinalVal().real()<<std::endl;
    removeFile(newPath);
    //input: 111

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement<<"1 1 0";
    generateMeasurement.close();
    newPath = generateQASMWithDiffPureInputState("Samples/tofolli.qasm",{1,1,1});
    c.Reset(newPath,"Samples/measureTest.txt");
    c.Contract(Stochastic);
    if(std::abs((c.GetFinalVal() - std::complex<double>(1,0)).real())>0.00001)
    {
        removeFile(newPath);
        removeFile("Samples/measureTest.txt");
        return false;
    }
    removeFile(newPath);
    out<<"Input: |111>; Measurement: 110; Probability: "<<c.GetFinalVal().real()<<std::endl;
    removeFile("Samples/measureTest.txt");
    return true;

}

/*
 * This function tests for correct measurement results for randomly generated circuits with Rx, Ry, Rz gates at random angles and
 * CNOT gates. Three circuits are tested, each with three measurements for both linegraph and stochastic contraction methods
 * Returns true on success and false on failure
 * Takes as input an ofstream for printing errors and results
 */

bool randomCircuitsTest(std::ofstream& out)
{
    out<<"Running Random Circuit test"<<std::endl<<std::endl;
    try {
        //circuit 1 (only Rx gates)
        //TESTING CIRCUIT 1:::::::::::::::::::::::::::::::::::::::::::::
        std::ofstream generateMeasurement("Samples/measureTestXI.txt");
        generateMeasurement << "X T X T X T X T";
        generateMeasurement.close();
        generateMeasurement.open("Samples/measureTestYI.txt");
        generateMeasurement << "Y T Y T Y T Y T";
        generateMeasurement.close();
        generateMeasurement.open("Samples/measureTestZI.txt");
        generateMeasurement << "Z T Z T Z T Z T";
        generateMeasurement.close();
        out << "Circuit: " << "Samples/rand-nq6-cn2-d10_rx.qasm" << std::endl;
        ContractionTools c("Samples/rand-nq6-cn2-d10_rx.qasm", "Samples/measureTestXI.txt");
        std::shared_ptr<Network> temp = c.Contract(Stochastic);
        out << "Scheme: Stochastic; Measurement: XI; Probability: " << c.GetFinalVal() << " Expected: " << "(0,0)"
            << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(0, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }
        temp->Reset();
        LineGraph lg(temp);
        lg.runQuickBB(20);
        lg.LGContract();
        out << "Scheme: Linegraph; Measurement: XI; Probability: " << temp->GetFinalValue() << " Expected: " << "(0,0)"
            << std::endl;
        if (std::abs((temp->GetFinalValue() - std::complex<double>(0, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }

        c.Reset("Samples/rand-nq6-cn2-d10_rx.qasm", "Samples/measureTestYI.txt");
        temp = c.Contract(Stochastic);
        out << "Scheme: Stochastic; Measurement: YI; Probability: " << c.GetFinalVal() << " Expected: "
            << "(0.0817020577919,0)" << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(0.0817020577919, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }
        temp->Reset();
        lg.Reset(temp);
        lg.LGContract();
        out << "Scheme: Linegraph; Measurement: YI; Probability: " << temp->GetFinalValue() << " Expected: "
            << "(0.0817020577919,0)" << std::endl;
        if (std::abs((temp->GetFinalValue() - std::complex<double>(0.0817020577919, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }

        c.Reset("Samples/rand-nq6-cn2-d10_rx.qasm", "Samples/measureTestZI.txt");
        temp = c.Contract(Stochastic);
        out << "Scheme: Stochastic; Measurement: ZI; Probability: " << c.GetFinalVal() << " Expected: "
            << "(0.97151682702,0)" << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(0.97151682702, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }
        temp->Reset();
        lg.Reset(temp);
        lg.LGContract();
        out << "Scheme: Linegraph; Measurement: ZI; Probability: " << temp->GetFinalValue() << " Expected: "
            << "(0.97151682702,0)" << std::endl;
        if (std::abs((temp->GetFinalValue() - std::complex<double>(0.97151682702, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }

        //TESTING CIRCUIT 2:::::::::::::::::::::::::::::::::::::::::::::
        //Circuit: Rx, Rz gates only
        out << "Circuit: " << "Samples/rand-nq6-cn2-d10_rxz.qasm" << std::endl;
        c.Reset("Samples/rand-nq6-cn2-d10_rxz.qasm", "Samples/measureTestXI.txt");
        temp = c.Contract(Stochastic);
        out << "Scheme: Stochastic; Measurement: XI; Probability: " << c.GetFinalVal() << " Expected: " << "(0,0)"
            << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(0, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }
        temp->Reset();
        lg.Reset(temp);
        lg.runQuickBB(20);
        lg.LGContract();
        out << "Scheme: Linegraph; Measurement: XI; Probability: " << temp->GetFinalValue() << " Expected: " << "(0,0)"
            << std::endl;
        if (std::abs((temp->GetFinalValue() - std::complex<double>(0, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }


        c.Reset("Samples/rand-nq6-cn2-d10_rxz.qasm", "Samples/measureTestYI.txt");
        temp = c.Contract(Stochastic);
        out << "Scheme: Stochastic; Measurement: YI; Probability: " << c.GetFinalVal() << " Expected: "
            << "(-5.22598463817e-09,0)" << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(-0.00000000522598463817, 0)).real()) > 0.00000001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }
        temp->Reset();
        lg.Reset(temp);
        lg.LGContract();
        out << "Scheme: Linegraph; Measurement: YI; Probability: " << temp->GetFinalValue() << " Expected: "
            << "(-5.22598463817e-09,0)" << std::endl;
        if (std::abs((temp->GetFinalValue() - std::complex<double>(-0.00000000522598463817, 0)).real()) > 0.00000001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }


        c.Reset("Samples/rand-nq6-cn2-d10_rxz.qasm", "Samples/measureTestZI.txt");
        temp = c.Contract(Stochastic);
        out << "Scheme: Stochastic; Measurement: ZI; Probability: " << c.GetFinalVal() << " Expected: "
            << "(0.993037531105,0)" << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(0.993037531105, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }
        temp->Reset();
        lg.Reset(temp);
        lg.LGContract();
        out << "Scheme: Linegraph; Measurement: ZI; Probability: " << temp->GetFinalValue() << " Expected: "
            << "(0.993037531105,0)" << std::endl;
        if (std::abs((temp->GetFinalValue() - std::complex<double>(0.993037531105, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }



        //TESTING CIRCUIT 3:::::::::::::::::::::::::::::::::::::::::::::
        //Circuit with Rx, Ry, and Rz gates
        out << "Circuit: " << "Samples/rand-nq6-cn2-d10_rxyz.qasm" << std::endl;
        c.Reset("Samples/rand-nq6-cn2-d10_rxyz.qasm", "Samples/measureTestXI.txt");
        temp = c.Contract(Stochastic);
        out << "Scheme: Stochastic; Measurement: XI; Probability: " << c.GetFinalVal() << " Expected: "
            << "(0.000364239511924,0)" << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(0.000364239511924, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }
        temp->Reset();
        lg.Reset(temp);
        lg.runQuickBB(20);
        lg.LGContract();
        out << "Scheme: Linegraph; Measurement: XI; Probability: " << temp->GetFinalValue() << " Expected: "
            << "(0.000364239511924,0)" << std::endl;
        if (std::abs((temp->GetFinalValue() - std::complex<double>(0.000364239511924, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }


        c.Reset("Samples/rand-nq6-cn2-d10_rxyz.qasm", "Samples/measureTestYI.txt");
        temp = c.Contract(Stochastic);
        out << "Scheme: Stochastic; Measurement: YI; Probability: " << c.GetFinalVal() << " Expected: "
            << "(0.000104923558503,0)" << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(0.000104923558503, 0)).real()) > 0.00000001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }
        temp->Reset();
        lg.Reset(temp);
        lg.LGContract();
        out << "Scheme: Linegraph; Measurement: YI; Probability: " << temp->GetFinalValue() << " Expected: "
            << "(0.000104923558503,0)" << std::endl;
        if (std::abs((temp->GetFinalValue() - std::complex<double>(0.000104923558503, 0)).real()) > 0.00000001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }


        c.Reset("Samples/rand-nq6-cn2-d10_rxyz.qasm", "Samples/measureTestZI.txt");
        temp = c.Contract(Stochastic);
        out << "Scheme: Stochastic; Measurement: ZI; Probability: " << c.GetFinalVal() << " Expected: "
            << "(0.988514404931,0)" << std::endl;
        if (std::abs((c.GetFinalVal() - std::complex<double>(0.988514404931, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }
        temp->Reset();
        lg.Reset(temp);
        lg.LGContract();
        out << "Scheme: Linegraph; Measurement: ZI; Probability: " << temp->GetFinalValue() << " Expected: "
            << "(0.988514404931,0)" << std::endl;
        if (std::abs((temp->GetFinalValue() - std::complex<double>(0.988514404931, 0)).real()) > 0.00001) {
            removeFile("Samples/measureTestXI.txt");
            removeFile("Samples/measureTestYI.txt");
            removeFile("Samples/measureTestZI.txt");
            return false;
        }


        removeFile("Samples/measureTestXI.txt");
        removeFile("Samples/measureTestYI.txt");
        removeFile("Samples/measureTestZI.txt");
        return true;

    }
    catch(std::exception& e)
    {
        out<<e.what()<<std::endl;
        removeFile("Samples/measureTestXI.txt");
        removeFile("Samples/measureTestYI.txt");
        removeFile("Samples/measureTestZI.txt");
        return false;
    }
}


bool unconnectedCircuitsTest(std::ofstream& out)
{
    out<<"Running Unconnected Circuits Test"<<std::endl<<std::endl;
	
	std::ofstream generateMeasurement("Samples/measureTest.txt");
    generateMeasurement << "T T T T T T";
    generateMeasurement.close();
    std::ofstream generateCircuit("Samples/unconnected.qasm");
    generateCircuit<<"3"<<std::endl<<"H 0"<<std::endl;
    generateCircuit.close();
    ContractionTools c("Samples/unconnected.qasm", "Samples/measureTest.txt");
    try{
    	c.Contract(Stochastic);
    }
    catch(InvalidTensorNetwork & error){
    	out<<"Passed Incorrect Number of Qubits Test"<<std::endl;
    }
    catch(std::exception & e)
    {
    	removeFile("Samples/measureTest.txt");
        removeFile("Samples/unconnected.qasm");
        out<<"Failed Test with exception: "<<e.what()<<std::endl;
        return false;
    }

    generateCircuit.open("Samples/unconnected.qasm");
    generateCircuit<<"3"<<std::endl<<"H 2"<<std::endl<<"H 1"<<std::endl<<"H 0"<<std::endl;
    generateCircuit.close();
    c.Reset();
     try{
    	c.Contract(Stochastic);
    }
    catch(std::exception & e)
    {
    	removeFile("Samples/measureTest.txt");
        removeFile("Samples/unconnected.qasm");
        out<<"Failed Test with exception: "<<e.what()<<std::endl;
        return false;
    }
    if (std::abs((c.GetFinalVal() - std::complex<double>(1, 0)).real()) > 0.00000001) {
            removeFile("Samples/measureTest.txt");
            removeFile("Samples/unconnected.qasm");
            out<<"Failed Unconnected Test 1"<<std::endl;
            return false;
    }
    out<<"Passed Unentangled Test 1"<<std::endl;

    generateMeasurement.open("Samples/measureTest.txt");
    generateMeasurement << "1 1 0 T T T";
    generateMeasurement.close();
    generateCircuit.open("Samples/unconnected.qasm");
    generateCircuit<<"3"<<std::endl<<"H 2"<<std::endl<<"H 0"<<std::endl<<"CNOT 0 1"<<std::endl;
    generateCircuit.close();
    c.Reset();
     try{
    	c.Contract(Stochastic);
    }
    catch(std::exception & e)
    {
    	removeFile("Samples/measureTest.txt");
        removeFile("Samples/unconnected.qasm");
        out<<"Failed Test with exception: "<<e.what()<<std::endl;
        return false;
    }
    if (std::abs((c.GetFinalVal() - std::complex<double>(0.25, 0)).real()) > 0.00000001) {
            removeFile("Samples/measureTest.txt");
            removeFile("Samples/unconnected.qasm");
            out<<"Failed Unconnected Test 2"<<std::endl;
            out<<"Expected 0.25+1j, received: "<<c.GetFinalVal()<<std::endl;
            return false;
    }
    out<<"Passed Unentangled Test 2"<<std::endl;
    
    
    removeFile("Samples/measureTest.txt");
    removeFile("Samples/unconnected.qasm");
    
    return true;
    
    
}

//this function runs all selected tests
//to modify which tests are run, simply change the flag from true to false in the testsToRun map.
//the function takes in a string, which is the testing log file path
void runTests(const std::string& fileToOutputTo)
{
    std::unordered_map<bool(*)(std::ofstream&),bool> testsToRun;
    testsToRun.insert({
                              {simpleReduceAndContractTest,true},
                              {testArbitraryOneQubit, true},
                              {testRotationCircuits, true},
                              {testLineGraph,true},
                              {tofolliTest,true},
                              {bellTest,true},
                              {testArbitraryTwoQubit,true},
                              {catTest,true},
                              {teleporationTest,true},
                              {testUserDefinedSequence,true},
                              {largeCircuitTest, true},
                              {randomCircuitsTest,true},
                              {unconnectedCircuitsTest,true}
                      });











    int failCount(0);
    std::ofstream output(fileToOutputTo);
    Timer t;
    t.start();
    double tempTime = 0;
    for(auto& test: testsToRun)
    {
        if(test.second)
        {
            if(test.first(output))
            {
                output<<std::endl<<"Passed"<<std::endl;
            }
            else
            {
                output<<std::endl<<"Failed"<<std::endl;
                failCount++;
            }
            output<<"Time Taken For Test: "<<t.getCPUElapsed()-tempTime<<" seconds."<<std::endl<<"------------------------------------------------------"<<std::endl;
            tempTime = t.getCPUElapsed();
        }
    }
    output<<"============================== Test Summary ================================"<<std::endl;
    output<<"TOTAL TEST FAILURE COUNT: "<<failCount<<". Please check above output for failed test if applicable"<<std::endl;
    output<<"Testing time: "<<t.getCPUElapsed()<<" seconds."<<std::endl;
}



