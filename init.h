#pragma once
#include "main.h"

typedef struct constants {
	double pi;
	double hbar;
	double e;
	double e2eps;
	double m_e;
	double a0;
	double L;
	double maxRho;
	double maxZ;
	double S;
} constants;

double init_constants_SI(constants* constants_SI, double m_e, double eps, double a0, double L);
