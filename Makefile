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

SHELL := /bin/sh
CC = g++
DEBUG = -g
LFLAGS = -std=c++11 -pthread
LMAXCUTFLAGS = -L$(HOME)/usr/local/lib -lnlopt -lm
INCLUDE = -I. -I$(HOME)/usr/local/include
SOURCE = src/
BUILD = build/
BIN = bin/


all:
	@printf "\n\n Compiling All Executables - Note that nlopt-2.4.2 must be installed successfully to compile.....\n\n"
	@mkdir -p input output
	@make qtorch
	@make testdry
	@make cut

qtorch:
	@printf "\n\n Compiling qtorch .....\n\n"
	@g++ -g $(LFLAGS) $(SOURCE)main.cpp -o $(BIN)qtorch $(INCLUDE)
	@cp $(BIN)qtorch $(HOME)/usr/local/bin
	@chmod +x $(HOME)/usr/local/bin/qtorch

test:
	@mkdir -p tests
	@printf "\n\n Compiling Test Executable ....."
	@g++ -g $(LFLAGS) $(SOURCE)tests.cpp -o tests/tester $(INCLUDE)
	@printf "\n"
	@printf "\n"
	@printf "\n"
	@printf "=======Tests executing -> check output/testingoutput.log for results======"
	@printf "\n"
	@printf "\n"
	@printf "\n"
	@mkdir -p tests/input tests/output && chmod +x tests/tester
	@./tests/tester > tests/output/tester.stdout

testdry:
	@mkdir -p tests
	@printf "\n\n Compiling Test Executable ....."
	@g++ -g $(LFLAGS) $(SOURCE)tests.cpp -o tests/tester $(INCLUDE)
	@mkdir -p tests/input tests/output && chmod +x tests/tester
	@printf "\n"

cut:
	@printf "\n\n Compiling QAOA MaxCut - Note that nlopt-2.4.2 must be installed successfully to compile .....\n\n"
	@g++ -g $(LFLAGS) $(SOURCE)maxcut.cpp -o $(BIN)maxcutQAOA $(INCLUDE) $(LMAXCUTFLAGS)
	@mkdir -p input output
	@cp $(BIN)maxcutQAOA $(HOME)/usr/local/bin
	@chmod +x $(HOME)/usr/local/bin/maxcutQAOA

clean:
	@printf "\nRemoving Executables -> Please Recompile\n\n"
	@\rm -rf $(SOURCE)*.dSYM
	@\rm -rf $(BIN)*.dSYM
	@\rm -rf tests/*.dSYM
	@\rm -f $(BIN)qtorch
	@\rm -f $(BIN)maxcutQAOA
	@\rm -f $(HOME)/usr/local/bin/qtorch
	@\rm -f $(HOME)/usr/local/bin/maxcutQAOA
	@\rm -f $(SOURCE)tester
	@\rm -f tests/tester

install:
	@-apt-get update  # To get the latest package lists
	@-apt-get install gcc -y
	@-apt-get install libtool
	@-brew install gcc
	@-brew upgrade gcc
	@-brew install libtool
	@-brew upgrade libtool
	@mkdir -p input
	@mkdir -p output
	@mkdir -p build
	@chmod +x $(BIN)quickbb_64
	@chmod +x $(BIN)quickbb_32
	@mkdir -p /usr/local/bin
	@mkdir -p $(HOME)/usr/local/bin
	@cp $(BIN)quickbb_64 /usr/local/bin
	@cp $(BIN)quickbb_32 /usr/local/bin
	@printf "TO INSTALL THE NLOPT 2.4.2 PACKAGE\n"
	@cd nlopt-2.4.2/ && make distclean
	@cd nlopt-2.4.2 && ./configure
	@cd nlopt-2.4.2/ && make && sudo make install #installs the nlopt-2.4.2 package (root privileges necessary)
	@mkdir -p /usr/local/include/qtorch
	@cp $(SOURCE)*.h /usr/local/include/qtorch
	@cp $(SOURCE)leviParser.hpp /usr/local/include/qtorch
	@cp $(SOURCE)qtorch.hpp /usr/local/include
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Node.h -o $(BUILD)Node.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Wire.h -o $(BUILD)Wire.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Network.h -o $(BUILD)Network.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)ContractionTools.h -o $(BUILD)ContractionTools.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)leviParser.hpp -o $(BUILD)leviParser.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)LineGraph.h -o $(BUILD)LineGraph.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) -I/usr/local/include $(SOURCE)maxcut.h -o $(BUILD)maxcut.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)preprocess.h -o $(BUILD)preprocess.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Timer.h -o $(BUILD)Timer.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Exceptions.h -o $(BUILD)Exceptions.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)zconf.h -o $(BUILD)zconf.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) -I/usr/local/include $(SOURCE)qtorch.hpp -o $(BUILD)qtorch.lo
	@-glibtool --mode=link --tag=CXX g++ -g -O -o $(BUILD)libqtorch.a $(BUILD)*.o -rpath /usr/local/lib
	@-glibtool --mode=link --tag=CXX g++ -g -O -o $(BUILD)libqtorch.la $(BUILD)*.lo -rpath /usr/local/lib
	@-glibtool --mode=install cp $(BUILD)libqtorch.a /usr/local/lib/libqtorch.a
	@-glibtool --mode=install cp $(BUILD)libqtorch.la /usr/local/lib/libqtorch.la
	@-glibtool  --mode=finish /usr/local/lib
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Node.h -o $(BUILD)Node.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Wire.h -o $(BUILD)Wire.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Network.h -o $(BUILD)Network.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)ContractionTools.h -o $(BUILD)ContractionTools.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)leviParser.hpp -o $(BUILD)leviParser.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)LineGraph.h -o $(BUILD)LineGraph.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) -I/usr/local/include $(SOURCE)maxcut.h -o $(BUILD)maxcut.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)preprocess.h -o $(BUILD)preprocess.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Timer.h -o $(BUILD)Timer.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Exceptions.h -o $(BUILD)Exceptions.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)zconf.h -o $(BUILD)zconf.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) -I/usr/local/include $(SOURCE)qtorch.hpp -o $(BUILD)qtorch.lo
	@-libtool --mode=link --tag=CXX g++ -g -O -o $(BUILD)libqtorch.a $(BUILD)*.o -rpath /usr/local/lib
	@-libtool --mode=link --tag=CXX g++ -g -O -o $(BUILD)libqtorch.la $(BUILD)*.lo -rpath /usr/local/lib
	@-libtool --mode=install cp $(BUILD)/.libs/libqtorch.a /usr/local/lib/libqtorch.a
	@-libtool --mode=install cp $(BUILD)libqtorch.la /usr/local/lib/libqtorch.la
	@-libtool  --mode=finish /usr/local/lib
	@rm $(BUILD)*.o
	@rm $(BUILD)*.lo
	@printf "\n\n SUCCESSFULLY INSTALLED QTORCH.\n INCLUDE THE LINE: #include "qtorch.hpp" AT THE TOP OF ANY FILES,\n AND LINK WITH THE -lqtorch COMMAND.\n NOTE THAT SOME COMPILERS MAY BE PICKY AND REQUIRE THIS FLAGS AT THE END OF THE COMPILE COMMAND. \n\n"

installlocal:
	@-apt-get update  # To get the latest package lists
	@-apt-get install gcc -y
	@-apt-get install libtool
	@-brew install gcc
	@-brew upgrade gcc
	@-brew install libtool
	@-brew upgrade libtool
	@mkdir -p input
	@mkdir -p output
	@mkdir -p build
	@chmod +x $(BIN)quickbb_64
	@chmod +x $(BIN)quickbb_32
	@mkdir -p $(HOME)/usr/local/bin
	@mkdir -p $(HOME)/usr/local/lib
	@mkdir -p $(HOME)/usr/local/include
	@cp $(BIN)quickbb_64 $(HOME)/usr/local/bin
	@cp $(BIN)quickbb_32 $(HOME)/usr/local/bin
	@printf "TO INSTALL THE NLOPT 2.4.2 PACKAGE\n"
	@-cd nlopt-2.4.2/ && make distclean
	@cd nlopt-2.4.2 && ./configure --prefix=$(HOME)/usr/local
	@cd nlopt-2.4.2/ && make && make install #installs the nlopt-2.4.2 package
	@mkdir -p $(HOME)/usr/local/include/qtorch
	@cp $(SOURCE)*.h $(HOME)/usr/local/include/qtorch
	@cp $(SOURCE)leviParser.hpp $(HOME)/usr/local/include/qtorch
	@cp $(SOURCE)qtorch.hpp $(HOME)/usr/local/include
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Node.h -o $(BUILD)Node.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Wire.h -o $(BUILD)Wire.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Network.h -o $(BUILD)Network.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)ContractionTools.h -o $(BUILD)ContractionTools.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)leviParser.hpp -o $(BUILD)leviParser.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)LineGraph.h -o $(BUILD)LineGraph.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) -I $(HOME)/usr/local/include $(SOURCE)maxcut.h -o $(BUILD)maxcut.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)preprocess.h -o $(BUILD)preprocess.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Timer.h -o $(BUILD)Timer.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Exceptions.h -o $(BUILD)Exceptions.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)zconf.h -o $(BUILD)zconf.lo
	@-glibtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) -I $(HOME)/usr/local/include $(SOURCE)qtorch.hpp -o $(BUILD)qtorch.lo
	@-glibtool --mode=link --tag=CXX g++ -g -O -o $(BUILD)libqtorch.a $(BUILD)*.o -rpath $(HOME)/usr/local/lib
	@-glibtool --mode=link --tag=CXX g++ -g -O -o $(BUILD)libqtorch.la $(BUILD)*.lo -rpath $(HOME)/usr/local/lib
	@-glibtool --mode=install cp $(BUILD)libqtorch.a $(HOME)/usr/local/lib/libqtorch.a
	@-glibtool --mode=install cp $(BUILD)libqtorch.la $(HOME)/usr/local/lib/libqtorch.la
	@-glibtool  --mode=finish $(HOME)/usr/local/lib
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Node.h -o $(BUILD)Node.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Wire.h -o $(BUILD)Wire.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Network.h -o $(BUILD)Network.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)ContractionTools.h -o $(BUILD)ContractionTools.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)leviParser.hpp -o $(BUILD)leviParser.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)LineGraph.h -o $(BUILD)LineGraph.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) -I $(HOME)/usr/local/include $(SOURCE)maxcut.h -o $(BUILD)maxcut.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)preprocess.h -o $(BUILD)preprocess.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Timer.h -o $(BUILD)Timer.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)Exceptions.h -o $(BUILD)Exceptions.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) $(SOURCE)zconf.h -o $(BUILD)zconf.lo
	@-libtool --mode=compile --tag=CXX g++ -g -O2 $(LFLAGS) -I $(HOME)/usr/local/include $(SOURCE)qtorch.hpp -o $(BUILD)qtorch.lo
	@-libtool --mode=link --tag=CXX g++ -g -O -o $(BUILD)libqtorch.a $(BUILD)*.o -rpath $(HOME)/usr/local/lib
	@-libtool --mode=link --tag=CXX g++ -g -O -o $(BUILD)libqtorch.la $(BUILD)*.lo -rpath $(HOME)/usr/local/lib
	@-libtool --mode=install cp $(BUILD)/.libs/libqtorch.a $(HOME)/usr/local/lib/libqtorch.a
	@-libtool --mode=install cp $(BUILD)libqtorch.la $(HOME)/usr/local/lib/libqtorch.la
	@-libtool  --mode=finish $(HOME)/usr/local/lib
	@rm $(BUILD)*.o
	@rm $(BUILD)*.lo
	@printf "\n\n SUCCESSFULLY INSTALLED QTORCH.\n INCLUDE THE LINE: #include "qtorch.hpp" AT THE TOP OF ANY FILES,\n AND LINK WITH THE '-I$(HOME)/usr/local/include -L$(HOME)/usr/local/lib' FLAGS.\n NOTE THAT SOME COMPILERS MAY BE PICKY AND REQUIRE THESE FLAGS AT THE END OF THE COMPILE COMMAND. \n\n"
