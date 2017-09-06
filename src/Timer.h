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

#include <chrono>

namespace qtorch {

	class Timer {
	public:
		void start();

		void reset();

		double getElapsed();

		double getCPUElapsed();

		std::chrono::high_resolution_clock::time_point mStart;
		std::clock_t mCPUClockStart;
		bool mStarted{false};
	};

	void Timer::start() {
		mStart = std::chrono::high_resolution_clock::now();
		mCPUClockStart = std::clock();
		mStarted = true;
	}

	double Timer::getElapsed() {
		if (!mStarted) {
			return 0.0;
		}
		auto end = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - mStart).count();
		return duration / 1000000000.0;
	}

	double Timer::getCPUElapsed() {
		return (std::clock() - mCPUClockStart) / (double) CLOCKS_PER_SEC;
	}

	void Timer::reset() {
		mStarted = false;
	}


}