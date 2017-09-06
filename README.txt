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



==================The Tensor Library================

There are three options to install the library for use: global 
installation, local installation, and docker. Here, we provide
a tutorial for each. gcc/g++ is the compiler used to compile and run the
library. The version of gcc/g++ must be recent enough support the C++14 
standard library. Otherwise, errors like: "g++: error: unrecognized 
command line option '-std=c++14'" may appear. In this case, we suggest
the user upgrade their gcc/g++ compiler to a more recent version.



::::::: Global Installation (Admin Permissions Required) :::::::::::


------- INSTALLING THE LIBRARY (REQUIRED FOR ANY TYPE OF USE) -------
Please type the command: "sudo make install" in a Unix command line to
install globally for all users.

This next step is optional, but it will make it easier to write and link
custom .cpp files that extend the library. If the user doesn't want to 
change any environment variables, they can simply run "make all", 
"make cut", or "make qtorch", then run the generated executables using 
./bin/qtorch or ./bin/maxcutQAOA

Optional: The user should add the directories: /usr/local/bin and 
$HOME/usr/local/bin to the PATH environment variable. For bash shell users,
this means modifying the $HOME/.bash_rc, $HOME/.bash_profile, or the 
$HOME/.profile file, so that every time the user logs in and opens a shell 
script, the PATH environment variable will change. Additionally, we recommend
setting the LIBRARY_PATH environment variable as well to reference the 
directories that we install the library in.

-----------------------------------------------------------------------------



----------- MAKING IT SO YOUR COMPILER CAN FIND THE EXECUTABLES ---------------

Simple commands for bash user (default shell) - please copy and paste into shell:

--THESE WILL SET THE PATH FOR THIS AND ALL FUTURE SHELL SESSIONS:--

echo 'export PATH=$PATH:/usr/local/bin:$HOME/usr/local/bin' >> $HOME/.bash_profile
source $HOME/.bash_profile

--FOR ONLY CURRENT SHELL SESSION:--

export PATH=$PATH:/usr/local/bin:$HOME/usr/local/bin

--SET LIBRARY_PATH VARIABLE ONLY FOR CURRENT SHELL SESSION:--

export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib:$HOME/usr/local/lib

--SET LIBRARY PATH FOR THIS AND ALL FUTURE SHELL SESSIONS:--

echo 'export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/lib:$HOME/usr/local/lib' >> $HOME/.bash_profile
source $HOME/.bash_profile
--------------------------------------------------------------------------------



---------- LINKING THE LIBRARY -------------------------------------------------
For users with a customized shell startup script, we ask that you add 
/usr/local/bin and $HOME/usr/local/bin to your PATH environment variable. 
Additionally, add /usr/local/lib and $HOME/usr/local/lib to your LIBRARY_PATH
environment variable.

If you set the LIBRARY_PATH variable, you can link the library with
the compiler flag: -ltncontr, else link using -L/usr/local/lib -ltncontr
--------------------------------------------------------------------------------



---------------- INCLUDING THE LIBRARY ----------------------------
Make sure to include the line: "#include "tncontr.hpp" at the top of 
any files that use the library, and link with the above linker flags.
--------------------------------------------------------------------------------


::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::



::::::::::: Local Installation (Admin Permissions NOT NEEDED) ::::::::::::::::::


 ----------- INSTALLING THE LIBRARY (REQUIRED FOR ANY TYPE OF USE) -------------
 Please type the command: "make installlocal" in a Unix command line to install 
 globally for all users.

 This next step is optional, but it will make it easier to write and link custom 
 .cpp files that extend the library. If the user doesn't want to change any 
 environment variables, they can simply run "make all", "make cut", or "make qtorch",
 then run the generated executables using ./bin/qtorch or 
 ./bin/maxcutQAOA

 Optional: The user should add the directory: $HOME/usr/local/bin to the PATH 
 environment variable. For bash shell users, this means modifying the $HOME/.bash_rc,
 $HOME/.bash_profile, or the $HOME/.profile file, so that every time the user logs in
 and opens a shell script, the PATH environment variable will change. Additionally, 
 we recommend setting the LIBRARY_PATH environment variable as well to reference 
 the directories that we install the library in.

 -------------------------------------------------------------------------------



----------- MAKING IT SO YOUR COMPILER CAN FIND THE EXECUTABLES ---------------

Simple commands for bash user (default shell) - please copy and paste into shell:

--THESE WILL SET THE PATH FOR THIS AND ALL FUTURE SHELL SESSIONS:--

echo 'export PATH=$PATH:$HOME/usr/local/bin' >> $HOME/.bash_profile
source $HOME/.bash_profile

--FOR ONLY CURRENT SHELL SESSION:--

export PATH=$PATH:$HOME/usr/local/bin

--SET LIBRARY_PATH VARIABLE ONLY FOR CURRENT SHELL SESSION:--

export LIBRARY_PATH=$LIBRARY_PATH:$HOME/usr/local/lib

--SET LIBRARY PATH FOR THIS AND ALL FUTURE SHELL SESSIONS:--

echo 'export LIBRARY_PATH=$LIBRARY_PATH:$HOME/usr/local/lib' >> $HOME/.bash_profile
source $HOME/.bash_profile
--------------------------------------------------------------------------------



---------- LINKING THE LIBRARY -------------------------------------------------

For users with a customized shell startup script, we ask that you add 
$HOME/usr/local/bin to your PATH environment variable. Additionally, add 
$HOME/usr/local/lib to your LIBRARY_PATH environment variable.

Link using: -L$(HOME)/usr/local/lib -I$(HOME)/usr/local/include

--------------------------------------------------------------------------------


---------------- INCLUDING THE LIBRARY ----------------------------
Make sure to include the line: "#include "tncontr.hpp" at the top of any files 
that use the library, and link with the above linker flags.
--------------------------------------------------------------------------------

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


::::::::::::::::::::::::::::::::: Docker Installation :::::::::::::::::::::::::::
An alternative way of running qTorch is to run it inside a Docker container. 
This allows the user to use qTorch regardless of the local environment 
(operating system, package dependencies etc) of her machine. We provide the
following Docker image on DockerHub: therealcaoyudong/qtorch, which contains
the qTorch source code as well as the necessary libraries needed for 
building qTorch.

Here, we provide a step-by-step instruction on how to use the Docker image, 
assuming no prior experience with Docker.

1. Install Docker on your computer. Also VirtualBox is needed;

2. In a command terminal, run: 

docker-machine ls

which returns a list of virtual machines. If the list is empty, 
create a virtual machine called default by running: 

docker-machine create ---driver virtualbox default

3. Configure the shell to the virtual machine by running: 

docker-machine env default

Run the command suggested by the returned message.

4. Now everything is set up, run the image using:

docker run -it therealcaoyudong/qtorch

It may take a while to download the image. When the download is complete, 
the shell header will look like "root@c7d9e3be4c53#". The hash string 
after @ is the identifier of the Docker container started by the "docker run" 
command just executed. To see a list of running Docker containers, run: 

docker ps

Note that in addition to a hash string, each container is also labels 
with a nickname (such as happy_einstein).

5. The Docker container is effectively an Ubuntu environment with qTorch 
installed and executable from any directory. The source code for qTorch 
can be located at "/root/qtorch". The user is free to re-build qTorch 
using "sudo make install", then "make all", "make qtorch", "make test", 
or "make cut". To see if qTorch is correctly installed, run:

qtorch

The output should ask for an input script file.

:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::


The library is then ready for use! See software guide for more details

====================================================
