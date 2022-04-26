#pragma once

#include <cstdio>
#include <cmath>
#include <ctime>
#include <cstring>
#include <conio.h>

#include <io.h>//access
#include <direct.h>//getcwd

#include "outputDisplay_functions.h"
#include "init.h"

// MC integration parameters ----------------------------------------------------------------------------------------------
const int dim = 6;
const long long int numRun = 1e8;
const long long int N = 256 * 1024;// NUM_SM* MAX_THREADS_PER_SM;//CUDA_CORES * dim * 10; // total number of points (~random numbers) to be thrown; preferably about numSM * maxThreadsPerSM
const int numPoints = N / dim; // number of dim-dimensional points

const double tol = 0.0005; // tolerance for integral error; program terminates once tol is reached
