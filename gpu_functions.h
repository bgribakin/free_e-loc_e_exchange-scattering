#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include "main.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "\n\nGPUassert: %s %s %d\n\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
__device__ double psi_loc(double a0, double L, double r);
__device__ double psi_free(double L, double S, double z_e);
__device__ double Coul_pot(double r);
__global__ void initRand(unsigned int seed, int runCounter, curandState_t* states);
__global__ void intMC_J_ee_exch(curandState_t* states, constants* constants_SI, double* gpu_f, double* gpu_f2, const int dim, double k, double phi_k);
