#!/bin/bash
gcc -o src/main.so -shared -fPIC -fopenmp src/main.c -lopenblas -O4