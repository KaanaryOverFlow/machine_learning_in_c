#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "utils.h"
#include "memory.h"
#include "nn.h"

struct {
	int fd;	
} *ed;


void default_app() {
	in();

	double input[] = {
		0,0,
		0,1,
		1,0,
		1,1,
	};

	double output[] = {
		0,
		1,
		1,
		0,
	};


	Mat in = {.rows = 4, .cols = 2, .data = input};
	Mat out = {.rows = 4, .cols = 1, .data = output};

	mat_p(in);
	mat_p(out);
	

	size_t arch[] =  {2, 1};
	NN nn = nn_alloc(2, arch, ARRAY_LEN(arch));
	
	
	plf(nn_cost(nn, in, out));


	for_print_percent(i, 1000 * 600)
		nn_delta(nn, mat_row(in, 0), mat_row(out, 0));
		nn_learn(nn, 1e-1);
	}
	plf(nn_cost(nn, in, out));

/*
	for_print_percent(i, 1000 * 1000)
#if 1
		nn_delta(nn, in, out);
		nn_learn(nn, 1e-1);
#else
		Fnn_delta(nn, in, out);
#endif
	}

	plf(nn_cost(nn, in, out));
*/


	out();
}

void sec_main(char *param) {
	if (!param) {
		default_app();
	} else {
		die("invalid parameter");
	}

}


int main(int argc, char *argv[]) {
	in();
	sec_main(argv[1]);
	out();
	return 0;
}
