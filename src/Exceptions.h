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

#include <exception>

namespace qtorch {

    class InvalidFile : public std::exception {
        const char *what() const noexcept override { return "Invalid Input or Output File Path"; }
    };

    class InvalidFileFormat : public std::exception {
        const char *what() const noexcept override { return "Invalid File Format."; }
    };
    
    class InvalidTensorNetwork : public std::exception {
        const char *what() const noexcept override { return "QASM File Does Not Entangle All Qubits - Please Add 2 qubit gates or check the specified number of qubits"; }
    };

    class ContractionFailure : public std::exception {
        const char *what() const noexcept override { return "Contraction Failed."; }
    };

    class InvalidUserContractionSequence : public std::exception {
        const char *what() const noexcept override { return "Invalid User Defined Contraction Sequence."; }
    };

    class InvalidFunctionInput : public std::exception {
        const char *what() const noexcept override { return "Input to Function Invalid."; }
    };

    class NumWiresVsNodeRank : public std::exception {
        const char *what() const noexcept override { return "Number of wires different than tensor rank."; }
    };

    class InvalidContractionMethod : public std::exception {
        const char *what() const noexcept override { return "Invalid Contraction Method."; }
    };

    class QbbFailure : public std::exception {
        const char *
        what() const noexcept override { return "Quick BB Failure.\n Details:\n quickbb_64 Executable: ELF 64-bit LSB executable,\n x86-64, version 1 (GNU/Linux), statically linked,\n for GNU/Linux 2.6.24, not stripped.\n Please check that your system is Linux and meets the requirements to run this binary.\n Otherwise, use the simple stochastic contraction method."; }
    };
    
}