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

#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <sstream>

//using namespace std;
namespace qtorch {
    class leviParser {
    public:

        std::map<std::string, std::string> mapString;
        std::map<std::string, bool> mapBool;
        std::map<std::string, int> mapInt;
        std::map<std::string, double> mapDouble;


        leviParser() {}

        leviParser(const std::string &fname) {
            readInputFile(fname);
        }

        bool readInputFile(const std::string &fname) {

            std::ifstream f(fname.c_str());

            if (f.is_open()) {

                std::string line;

                while (f.good()) {

                    getline(f, line);
                    std::string datatype;
                    std::string membername;
                    if (line[0] == '>') {

                        std::stringstream ss;

                        ss << line;
                        ss >> datatype;
                        ss >> membername;
                        if (datatype == ">string") {
                            std::string s;
                            ss >> s;
                            mapString[membername] = s;
                        } else if (datatype == ">bool") {
                            std::string s;
                            ss >> s;
                            if (s == "1" || s == "true" || s == "True" || s == "yes" || s == "Yes") {
                                mapBool[membername] = true;
                            } else if (s == "0" || s == "false" || s == "False" || s == "no" || s == "No") {
                                mapBool[membername] = false;
                            } else {
                                std::cout << "Error in leviParser, " << membername
                                          << ". bool inputs must be in one of the following forms: ";
                                std::cout << "1, true, True, yes, or Yes." << std::endl;
                            }
                        } else if (datatype == ">int") {
                            int i;
                            ss >> i;
                            //std::cout << "int, " << membername << " " << i << std::endl;
                            mapInt[membername] = i;
                        } else if (datatype == ">double") {
                            double d;
                            ss >> d;
                            mapDouble[membername] = d;
                        } else {
                            std::cout << "Error. Only the following types are supported in leviParser: ";
                            std::cout << "string, bool, int, double." << std::endl;
                        }

                    }

                }

                f.close();

            } else {
                std::cout << "Unable to open file.";
                return false;
            }

            return true;

        }


    };

}




