#pragma once
#include "main.h"

void f_head_display(char* filename);
void f_data_display(char* filename, double k, double phi_k, double cpu_f_sum, double cpu_f2_sum, int runCounter, double V_MC, time_t tic, time_t toc);
void head_display();
void data_display(double k, double phi_k, double temp_res, double temp_err, int runCounter, time_t tic, time_t toc);
int keyboard_control(char* filename, double k, double phi_k, double temp_res, double temp_err, int runCounter, time_t tic, time_t toc);
int live_control_and_display(char* filename, time_t tic, long long int runCounter, double V_MC, double cpu_f_sum, double cpu_f2_sum, double k, double phi_k);
void finish_display();
