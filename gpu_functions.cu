#include "gpu_functions.h"

// ---------------------------------- __device__ functions (helper funcs, wave funcs, potential etc.) ------------------------------------------- 

__device__ double psi_loc(double a0, double r) {

	const double pi = 3.14159265359;
	double psi = exp(-r / a0)  / sqrt( pow(a0, 3) * pi);

	return psi;
}

__device__ double psi_loc_quantized(double a0, double L, double rho, double z) {

	const double pi = 3.14159265359;
	double psi = sqrt(8 / (L * pow(a0, 2))) * exp(-rho / a0);

	return psi;
}

__device__ double psi_free(double L, double S, double z) {

	const double pi = 3.14159265359;
	double psi;
	if (abs(z) <= L / 2)
		psi = sqrt(2 / L / S) * cos(pi / L * z);
	else
		psi = 0;
	return psi;
}

// Coulomb attraction potential without dimensional coefficients
__device__ double Coul_pot(double r) {	
	return  1 / r;
}

// ------------------------------------------------ __global__ kernel functions ------------------------------------------------------------------ 

// used to initialize the random states */
__global__ void initRand(unsigned int seed, int runCounter, curandState_t* states) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	// nvidia recommends using the same seed but monotonically increasing sequence numbers when dealing with multiple kernel launches
	// but that is much much slower, so best to leave runCounter=0
	for (int i = index; i < N; i += stride)
		curand_init(seed, N * runCounter + i, 0, &states[i]);
}

// calculates J_exch(q) using MC method
// stores numPoints function values in gpu_f and as much squares in gpu_f2
__global__ void intMC_J_ee_exch(curandState_t* states, constants* gpu_constants_SI, double* gpu_f, double* gpu_f2, const int dim, double k, double phi_k) {
	
	double rho, Rho, phi, Phi, z, Z; // (assume vectors) R = 1/2 (r_1 + r_2), r = r_1 - r_2 --> double cylindrical

	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	double* rands = new double[dim];

	for (int i = index; i < numPoints; i += stride) {
		rands[0] = curand_uniform_double(&states[dim * i + 0]);
		rands[1] = curand_uniform_double(&states[dim * i + 1]);
		rands[2] = curand_uniform_double(&states[dim * i + 2]);
		rands[3] = curand_uniform_double(&states[dim * i + 3]);
		rands[4] = curand_uniform_double(&states[dim * i + 4]);
		rands[5] = curand_uniform_double(&states[dim * i + 5]);
		
		rho = rands[0] * gpu_constants_SI->maxRho;
		Rho = rands[1] * gpu_constants_SI->maxRho;
		phi = rands[2] * 2 * gpu_constants_SI->pi;
		Phi = rands[3] * 2 * gpu_constants_SI->pi;
		z = rands[4] * gpu_constants_SI->maxZ - gpu_constants_SI->maxZ / 2;
		Z = rands[5] * gpu_constants_SI->maxZ - gpu_constants_SI->maxZ / 2;
				
		// evalute k-factor
		double k_factor_arg = -k * (2 * Rho * sin(phi_k / 2) * sin(Phi - phi_k / 2) 
									  + rho * cos(phi_k / 2) * cos(phi - phi_k / 2));//rho_2 * cos(phi_2) - rho_1 * cos(phi_1 - phi_k)); //= k . rho_2 - k'. rho_1 
		double k_factor_real = cos(k_factor_arg);
		//double k_factor_im = sin(k_factor_arg); // leads to 0 due to symmetry [checked]
		
		/*
		// evaluate coords for wave functions & potential
		double R2 = pow(Rho, 2) + pow(Z, 2), R = sqrt(R2);
		double r2 = pow(rho, 2) + pow(z, 2), r = sqrt(r2);
		double rRcos = R * r * cos(Phi - phi);

		double r_1 = sqrt(R2 + r2 / 4 + rRcos); // WRONG! this expression is for polar (2D) coordinates. 
		double r_2 = sqrt(R2 + r2 / 4 - rRcos); // In 3D cyl. coordinates you have to sum (\vec{Rho} + \vec{rho}/2)^2 + (Z + z / 2)^2

		double z_1 = Z + z / 2, z_2 = Z - z / 2;

		// evaluate wave functions
		double psi_free_1 = psi_free(gpu_constants_SI->L, gpu_constants_SI->S, z_1);
		double psi_free_2 = psi_free(gpu_constants_SI->L, gpu_constants_SI->S, z_2);

		double psi_loc_1 = psi_loc(gpu_constants_SI->a0, r_1);
		double psi_loc_2 = psi_loc(gpu_constants_SI->a0, r_2);
		*/
		
		// evaluate coords for wave functions & potential
		/* for spherical localized electron */
		double R2 = pow(Rho, 2) + pow(Z, 2), R = sqrt(R2);
		double r2 = pow(rho, 2) + pow(z, 2), r = sqrt(r2);
		double rhoRhocos = Rho * rho * cos(Phi - phi);

		double r_1 = sqrt(R2 + r2 / 4 + rhoRhocos + z * Z);
		double r_2 = sqrt(R2 + r2 / 4 - rhoRhocos - z * Z);

		/* for quantized localized electron
		double Rho2 = pow(Rho, 2);
		double rho2 = pow(rho, 2);
		double rhoRhocos = Rho * rho * cos(Phi - phi);

		double r_1 = sqrt(Rho2 + rho2 / 4 + rhoRhocos);
		double r_2 = sqrt(Rho2 + rho2 / 4 - rhoRhocos);
		*/

		double z_1 = Z + z / 2, z_2 = Z - z / 2;

		// evaluate wave functions
		double psi_free_1 = psi_free(gpu_constants_SI->L, gpu_constants_SI->S, z_1);
		double psi_free_2 = psi_free(gpu_constants_SI->L, gpu_constants_SI->S, z_2);

		//double psi_loc_1 = psi_loc_quantized(gpu_constants_SI->a0, gpu_constants_SI->L, rho_1, z_1);
		//double psi_loc_2 = psi_loc_quantized(gpu_constants_SI->a0, gpu_constants_SI->L, rho_2, z_2);

		double psi_loc_1 = psi_loc(gpu_constants_SI->a0, r_1);
		double psi_loc_2 = psi_loc(gpu_constants_SI->a0, r_2);
		

		// evaluate the V_I potential
		double V = gpu_constants_SI->e2eps * Coul_pot(r);

		// don't forget about the jacobian
		double detTheta = rho * Rho;// jacobi determinant of new double cylindrical coordinates

		// now we simply evaluate the complete integrand
		gpu_f[i] = gpu_constants_SI->S * detTheta * k_factor_real * psi_free_1 * psi_loc_2 * V * psi_free_2 * psi_loc_1 / gpu_constants_SI->e * 1e6 * 1e12; // in micro eV * mum2
		gpu_f2[i] = gpu_f[i] * gpu_f[i]; // here we store their squares to get <f^2> -> int error
	}
	delete[] rands;
}
