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

void setup() {
	note("\t\t###################################################");
	note("\t\t# 盲目人間はこの世界で色がありませんに思うか？    #");
	note("\t\t# この人は思うと道がありません。人間は思うるです。#");
	note("\t\t###################################################");
	note("\t\t				    Zedeleyici.1337");
	ed = mmap(NULL, 4096 * 2, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (ed == MAP_FAILED) die("inital map of setup");


}

void default_app() {
	in();

	// TODO: implement image upsacle in this base.

	double in[] = {0,1,2,3,4};
	double out[] = {4, 8};
	Mat input = {.rows = 1, .cols = 5, .data = in};
	Mat output = {.rows = 1, .cols = 2, .data = out};
	Dense network[] = {
		dense_init(5, 20),
		dense_init(20, 13),
		dense_init(13, 2),
	};
	
	plf(network_cost(network, ARRAY_LEN(network), input, output));
	mat_p(network_forward(network, ARRAY_LEN(network), input));
	FOR(i, 5100)
		network_backward(network, ARRAY_LEN(network), 0.0000001f, input, output);

	plf(network_cost(network, ARRAY_LEN(network), input, output));
	mat_p(network_forward(network, ARRAY_LEN(network), input));

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
	setup();
	sec_main(argv[1]);
	out();
	return 0;
}
