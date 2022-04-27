#include "outputDisplay_functions.h"

/* f_head_display prints initial info to a file, including table head
*/
void f_head_display(char* filename) {

	FILE* F = fopen(filename, "a");
	if (F == NULL) {
		printf("Failed opening file \"%s\"! \n", filename);
		exit(0);
	}
	fprintf(F, "\n       ______________________________________________________________________\n");
	fprintf(F, "       J_X-e   &   error, mueV*mum^2  |  total points  |   elapsed time\n");
	fclose(F);
}

void f_data_display(char* filename, double temp_res, double temp_err, int runCounter, time_t tic, time_t toc) {
	
	FILE* F = fopen(filename, "a");
	if (F == NULL) {
		printf("Failed opening file \"%s\"! \n", filename);
		exit(0);
	}
	fprintf(F, "\t%13e\t%12e", temp_res, temp_err);
	fprintf(F, "\t %9e", (double)(runCounter + 1) * numPoints);
	fprintf(F, "\t  %7e", double(toc - tic) / CLOCKS_PER_SEC);

	fclose(F);
}

/* head_display prints initial info on the screen, including table head
*/
void head_display() {

	printf("\n--------------------------------------------------------------------------------------------\n");
	printf("  Calculation controls:\n");
	printf("  \t'p' -- pause\n");
	printf("  \t'n' -- skip to next filename\n");
	printf("  \t'b' -- break\n");
	printf("  Program will terminate when the error is less than tol = %e\n", tol);

	printf("       ______________________________________________________________________\n");
	printf("       J_X-e   &   error, mueV*mum^2  |  total points  |   elapsed time\n");
}

void data_display(double temp_res, double temp_err, int runCounter, time_t tic, time_t toc) {

	for (int bCount = 0; bCount < 150; bCount++) // erase old line
		printf("\b");
	printf("\t%13e\t%12e", temp_res, temp_err);
	printf("\t %9e", (double)(runCounter + 1) * numPoints);
	printf("\t  %7e", double(toc - tic) / CLOCKS_PER_SEC);
}

int keyboard_control(char* filename, double temp_res, double temp_err, int runCounter, time_t tic, time_t toc) {

	if (_kbhit()) {
		char kb = _getch(); // consume the char from the buffer, otherwise _kbhit remains != 0

		// next
		if (kb == 'n') {			
			printf("\n=============================================================================================================================\n\n");
			printf(" Skipping to next calculation...\n\n");
			printf("=============================================================================================================================\n\n\n");
			return 0;
		}

		// pause-unpause
		else if (kb == 'p') {
			FILE* F = fopen(filename, "a");
			if (F == NULL)
				printf("Failed opening file \"%s\"! \n", filename);

			fprintf(F, "\t%13e\t %12e", temp_res, temp_err);
			fprintf(F, "\t %9e", (double)(runCounter + 1) * numPoints);
			fprintf(F, "\t  %7e s\n", double(toc - tic) / CLOCKS_PER_SEC);
			fprintf(F, "--------------------------------------------------------------------------------------------\n");
			fclose(F);

			printf("\n\n Program paused: intermediate results appended to file \"%s\".\n", filename);
			printf(" To continue, press any key.\n\n");

			_getch(); // wait for a second key press to continue calculation

			printf("       ______________________________________________________________________\n");
			printf("       J_e-e   &   error, mueV*mum^2  |  total points  |   elapsed time\n");
			return 1;
		}

		// exit program
		else if (kb == 'b') {
			printf("\n=============================================================================================================================\n\n");
			printf(" Program stopped.\n\n");
			printf("=============================================================================================================================\n\n\n");
			exit(10);
		}
	}
}

int live_control_and_display(char* filename, time_t tic, long long int runCounter, double V_MC, double cpu_f_sum, double cpu_f2_sum) {

	double temp_res, temp_err;

	temp_res = V_MC * cpu_f_sum / ((runCounter + 1) * numPoints);
	temp_err = 3 * V_MC / sqrt((runCounter + 1) * numPoints)
		* sqrt(cpu_f2_sum / ((runCounter + 1) * numPoints) - cpu_f_sum * cpu_f_sum / ((runCounter + 1) * numPoints) / ((runCounter + 1) * numPoints));

	time_t toc = clock();
	data_display(temp_res, temp_err, runCounter, tic, toc);

	if (temp_err < tol * temp_res) {
		return 1;
	}
	// keyboard control
	keyboard_control(filename, temp_res, temp_err, runCounter, tic, toc);
}

void finish_display() {
	printf("\n=============================================================================================================================\n\n\n");
	printf("\n\n\n\t\tAll calculations processed.\n\n");
}
