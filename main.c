#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>

#include "utils.h"
#include "NN.h"

void sec_main(char *param) {
	
	size_t arch[] = {2, 2, 1};
	NN nn = nn_alloc(arch, ARRAY_LEN(arch));



	double data_in[] = {
		0,0,
		0,1,
		1,0,
		1,1,
		};

	double data_out[] = {0,1,1,0};

	Mat ti = {.rows = 4, .cols = 2, .data = data_in};
	Mat to = {.rows = 4, .cols = 1, .data = data_out};

	
	double eps = 1e-1;
	double rate = 2e-1;

	do {
	note("cost : %lf", nn_cost(nn, ti, to));
	FOR(5000)
	// nn_learn(nn, eps, rate, ti, to);
		my_learn(nn, rate, ti, to);
	note("cost : %lf", nn_cost(nn, ti, to));
	
	FOR(2) {
		FORR(2) {
			MAT_AT(nn_in(nn), 0, 0) = i;
			MAT_AT(nn_in(nn), 0, 1) = ii;
			nn_ileri(nn);
			note("%zu ^ %zu = %lf",i, ii, MAT_AT(nn_out(nn), 0, 0));
		}
	}
	} while(0);
}


int main(int argc, char *argv[]) {
	in();
	sec_main(argv[1]);
	out();
	return 0;
}
