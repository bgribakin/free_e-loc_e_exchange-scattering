#pragma once
#include "init.h"

// simple func to initialize the constants structure with specific eps and a0 to accomodate different materials
// returns the Monte-Carlo volume V_MC
double init_constants_SI(constants* constants_SI, double m_e, double eps, double a0, double L) {

	constants_SI->pi = 3.14159265359;
	constants_SI->hbar = 1.054571817e-34; // J*s
	constants_SI->e = 1.60217663e-19; // coul.
	constants_SI->e2eps = pow(1.60217663e-19, 2) / (4 * 3.14159265359 * eps * 8.85418781e-12); // e2eps = e^2 / (4*pi*eps*eps0) (SI)
	constants_SI->m_e = 9.109383561e-31 * m_e; // eff e mass in CdTe
	constants_SI->a0 = a0;
	constants_SI->L = L;
	constants_SI->maxRho = a0 * 3;
	constants_SI->maxZ = L;
	constants_SI->S = 1; // this shouldn't affect anything, but it feels better to take it into account

	double V_MC = pow(2 * constants_SI->pi * constants_SI->maxRho * constants_SI->maxZ, 2);
	return V_MC;
}

