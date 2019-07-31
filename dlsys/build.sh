#!/bin/bash
g++ -o src/main.so -shared -fPIC -fopenmp src/main.cpp -lopenblas -O4