#!/bin/bash
g++ -std=c++11 -I /usr/local/Cellar/eigen/3.3.9/include/eigen3 -I ./lib calc/main.cpp -o exe/main
if [ $? -eq 0 ]; then
	./exe/main 2>&1 | tee run.log
fi
