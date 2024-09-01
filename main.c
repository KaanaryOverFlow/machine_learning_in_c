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

	Mat in = mat_alloc(1, 2);
	MAT_AT(in, 0, 0) = 1;
	MAT_AT(in, 0, 1) = 1;
	
	Mat out = mat_alloc(1, 1);
	MAT_AT(out, 0, 0) = 0;


	

	size_t arch[] =  {3, 5, 9, 1};
	NN nn = nn_alloc(2, arch, ARRAY_LEN(arch));
	mat_p(nn.layer[nn.layer_count].output);
	mat_copy(nn.input, in);
	nn_forward(nn);
	mat_p(nn.layer[nn.layer_count].output);

	nn_cost(nn, in, out);


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
