/*
  bla bla 
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "curand.h"
#include "curand_kernel.h"

#include <cstdio>
#include <cmath>
#include <ctime>
#include <cstring>
#include <conio.h>

#include <io.h>//access
#include <direct.h>//getcwd

#include "main.h"
#include "gpu_functions.h"
#include "gpuReductionSum.h"

int main() {

	time_t tic;
	time_t toc;
	time_t seed = clock() + time(0);

	constants a = {};
	constants* constants_SI, * gpu_constants_SI;
	constants_SI = &a;

	double m_e = 0.11; // eff e mass in CdTe, units of m0
	double eps = 10.2; // static dielectric constant in CdTe
	double a0 = 3e-9; // Bohr radius of the localized electron, m
	double L = 20e-9; // QW width, m
	double V_MC = init_constants_SI(constants_SI, m_e, eps, a0, L); // initializes struct pointer with constants: pi, hbar, e2eps, m_e, and a0; also calculates V_MC

	gpuErrchk(cudaMalloc((void**)&gpu_constants_SI, sizeof(constants)));
	gpuErrchk(cudaMemcpy(gpu_constants_SI, constants_SI, sizeof(constants), cudaMemcpyHostToDevice));

	char filename[] = "integral__.dat";
	f_head_display(filename);
	head_display();

	int blockSize = 384;
	int numBlocks = (N + blockSize) / blockSize;

	curandState_t* states;
	gpuErrchk(cudaMalloc((void**)&states, N * sizeof(curandState_t))); // space for random states

	tic = clock();
	initRand << <numBlocks, blockSize >> > (seed, 0, states); // invoke the GPU to initialize all of the random states
	gpuErrchk(cudaDeviceSynchronize());

	double cpu_f; // variable for sum of integrand function values inside a run on cpu
	double cpu_f2; // var for the sum of it's squares (to calculate error later) on cpu

	double cpu_f_sum = 0.0; // vars to accumulate final values across all runs
	double cpu_f2_sum = 0.0;

	double temp_res; // for storing integral estimates in real-time 
	double temp_err;

	double* gpu_f; // array for integrand function values at N random points on gpu
	double* gpu_f2; // array for it's squares (to calculate error later) on gpu
	gpuErrchk(cudaMalloc((void**)&gpu_f, numPoints * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&gpu_f2, numPoints * sizeof(double)));

	double* gpu_f_out;
	double* gpu_f2_out;
	gpuErrchk(cudaMalloc((void**)&gpu_f_out, numPoints * sizeof(double)));
	gpuErrchk(cudaMalloc((void**)&gpu_f2_out, numPoints * sizeof(double)));


	double k = 0;
	double phi_k = 0;

	// main loop to accumulate integral estimate and error
	long long int runCounter;
	for (runCounter = 0; runCounter < numRun; runCounter++) {

		intMC_J_ee_exch << <numBlocks, blockSize >> > (states, gpu_constants_SI, gpu_f, gpu_f2, dim, k, phi_k); // accumulate func and func^2 evaluations in gpu_f and gpu_f2

		gpuErrchk(cudaPeekAtLastError());
		gpuErrchk(cudaDeviceSynchronize());

		// efficient parallel reduction sum algorithm
		sumGPUDouble(gpu_f, gpu_f_out, numPoints); 
		sumGPUDouble(gpu_f2, gpu_f2_out, numPoints);

		// copy back
		gpuErrchk(cudaMemcpy(&cpu_f, gpu_f, sizeof(double), cudaMemcpyDeviceToHost));
		gpuErrchk(cudaMemcpy(&cpu_f2, gpu_f2, sizeof(double), cudaMemcpyDeviceToHost));

		cpu_f_sum += cpu_f;
		cpu_f2_sum += cpu_f2;

		if (runCounter % 5 == 0) { //  we lose speed if we printf on every run
			toc = live_control_and_display(filename, tic, runCounter, V_MC, cpu_f_sum, cpu_f2_sum);
		}
	}

	f_data_display(filename, temp_res, temp_err, runCounter, tic, toc);

	gpuErrchk(cudaFree(states));
	gpuErrchk(cudaFree(gpu_f));
	gpuErrchk(cudaFree(gpu_f2));
	gpuErrchk(cudaFree(gpu_f_out));
	gpuErrchk(cudaFree(gpu_f2_out));

	finish_display();
	
	return 1;
}