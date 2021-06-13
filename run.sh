#!/bin/bash
g++ -std=c++11 -I /usr/local/Cellar/eigen/3.3.9/include/eigen3/ main.cpp -o main
if [ $? -eq 0 ]; then
	./main > run.log
fi
