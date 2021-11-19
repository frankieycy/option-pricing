#!/bin/bash
path_eigen="/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3"
if [ -z "$1" ]; then
	file="main"
else
	file=$1
fi
g++ -std=c++11 -I ${path_eigen} -I ./lib calc/${file}.cpp -o exe/${file}
if [ $? -eq 0 ]; then
	./exe/${file} 2>&1 | tee run.log
fi
