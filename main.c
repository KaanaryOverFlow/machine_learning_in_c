#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>

#include "utils.h"

typedef struct {
	size_t rows;
	size_t cols;
	double *data;

} Mat;

#define MAT_AT(m, r, c) (m).data[(r)*(m).cols + (c)]

double rand_double(void) {
	return (double) rand() / (double) RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols) {
	double h = 2;
	double l = 0;
	Mat m;
	m.rows = rows;
	m.cols = cols;
	m.data = malloc(sizeof(*m.data)*rows*cols);
	if (m.data == NULL) die("out of memory");
	srand(time(NULL));
	FOR(m.rows) {
		FORR(m.cols) {
			MAT_AT(m, i, ii) = rand_double() * (h - l) + l;
		}
	}

	return m;
}


void mat_print(Mat m, char *name) {
	printf("%s = [\n", name);
	FOR(m.rows) {
		FORR(m.cols) {
			printf("\t%lf", MAT_AT(m, i, ii));
		}
		printf("\n");
	}
	printf("]\n");
}

#define mat_p(m) mat_print(m, #m)
#define phex(m) note("%s : %lf", #m, m)
#define plx(m) note("%s : %lx", #m, m)

void mat_sum(Mat a, Mat b) {
	if (a.cols != b.cols || a.rows != b.rows) die("invalid matrix to sum");
	FOR(a.rows) {
		FORR(a.cols) {
			MAT_AT(a, i, ii) += MAT_AT(b, i, ii);
		}
	}

}

void mat_fill(Mat m, double x);

void mat_dot(Mat dest, Mat a, Mat b) {
	if (a.cols != b.rows || dest.rows != a.rows || dest.cols != b.cols) die("invalid matrix to dot");
	size_t inner = a.cols;
	
	mat_fill(dest, 0);

	FOR(dest.rows) {
		FORR(dest.cols) {
			MAT_AT(dest, i, ii) = 0;
			for (unsigned int iii = 0; iii < inner; iii++) {
				
				MAT_AT(dest, i, ii) += MAT_AT(a, i, iii) * MAT_AT(b, iii, ii);

			}
		}
	}
}

double sigmoidf(double x) {
	return 1 / (1 + expf(-x));
}

void mat_sig(Mat m) {
	FOR(m.rows) {
		FORR(m.cols) {
			MAT_AT(m, i, ii) = sigmoidf(MAT_AT(m, i, ii));
		}
	}
}

void mat_fill(Mat m, double x) {
	FOR(m.rows) {
		FORR(m.cols) {
			MAT_AT(m, i, ii) = x;
		}
	}

}



typedef struct {
	Mat a0;
	Mat w1, b1, a1;
	Mat w2, b2, a2;
} Xor;

double ileri(Xor xor) {

	/*
	Mat bir = mat_alloc(3,5);
	Mat iki = mat_alloc(5,7);
	Mat r = mat_alloc(3,7);

	mat_fill(bir, 3);
	mat_fill(iki, 5);
	mat_dot(r, bir, iki);

	mat_p(bir);
	mat_p(iki);
	mat_p(r);

	die("X");

	*/
	
	mat_dot(xor.a1, xor.a0, xor.w1);
	mat_sum(xor.a1, xor.b1);
	mat_sig(xor.a1);
	
	mat_dot(xor.a2, xor.a1, xor.w2);
	mat_sum(xor.a2, xor.b2);
	mat_sig(xor.a2);
	return MAT_AT(xor.a2, 0, 0);

}

double cost(Xor xor, Mat ti, Mat to) {
	size_t tc = ti.rows;

	double cost = 0;
	FOR(tc) {
		MAT_AT(xor.a0, 0, 0) = MAT_AT(ti, i, 0);
		MAT_AT(xor.a0, 0, 1) = MAT_AT(ti, i, 1);
		
		ileri(xor);
		double d = MAT_AT(to, i, 0) - MAT_AT(xor.a2, 0, 0);
		cost += d*d;
	}
	return cost/tc;
}


Xor xor_alloc() {
	
	Xor xor;
	xor.a0 = mat_alloc(1,2);
	
	xor.w1 = mat_alloc(2,2);
	xor.b1 = mat_alloc(1,2);
	xor.a1 = mat_alloc(1,2);
	
	xor.w2 = mat_alloc(2,1);
	xor.b2 = mat_alloc(1,1);
	xor.a2 = mat_alloc(1,1);
	return xor;

}

void learn(Xor g, Xor m, Mat ti, Mat to) {

	double eps = 1e-1;
	double rate = 1e-1;
	double saved = 0;
	double c = cost(m, ti, to);

	FOR(m.w1.rows) {
		FORR(m.w1.cols) {
			saved = MAT_AT(m.w1, i, ii);
			MAT_AT(m.w1, i, ii) += eps;
			MAT_AT(g.w1, i, ii) = (cost(m, ti, to) - c)/eps;
			// phex(MAT_AT(g.w1, i, ii));
			MAT_AT(m.w1, i, ii) = saved;
		}
	}
	
	FOR(m.w2.rows) {
		FORR(m.w2.cols) {
			saved = MAT_AT(m.w2, i, ii);
			MAT_AT(m.w2, i, ii) += eps;
			MAT_AT(g.w2, i, ii) = (cost(m, ti, to) - c)/eps;
			MAT_AT(m.w2, i, ii) = saved;
		}
	}
	
	FOR(m.b1.rows) {
		FORR(m.b1.cols) {
			saved = MAT_AT(m.b1, i, ii);
			MAT_AT(m.b1, i, ii) += eps;
			MAT_AT(g.b1, i, ii) = (cost(m, ti, to) - c)/eps;
			MAT_AT(m.b1, i, ii) = saved;
		}
	}
	
	FOR(m.b2.rows) {
		FORR(m.b2.cols) {
			saved = MAT_AT(m.b2, i, ii);
			MAT_AT(m.b2, i, ii) += eps;
			MAT_AT(g.b2, i, ii) = (cost(m, ti, to) - c)/eps;
			MAT_AT(m.b2, i, ii) = saved;
		}
	}
	
	FOR(m.w1.rows) {
		FORR(m.w1.cols) {
			MAT_AT(m.w1, i, ii) -= rate*MAT_AT(g.w1, i, ii);
		}
	}
	
	FOR(m.w2.rows) {
		FORR(m.w2.cols) {
			MAT_AT(m.w2, i, ii) -= rate*MAT_AT(g.w2, i, ii);
		}
	}
	
	FOR(m.b1.rows) {
		FORR(m.b1.cols) {
			MAT_AT(m.b1, i, ii) -= rate*MAT_AT(g.b1, i, ii);
		}
	}
	
	FOR(m.b2.rows) {
		FORR(m.b2.cols) {
			MAT_AT(m.b2, i, ii) -= rate*MAT_AT(g.b2, i, ii);
		}
	}

}


void sec_main(char *param) {
	
	Xor xor = xor_alloc();
	Xor xor_d = xor_alloc();

	double data_in[] = {
		0,0,
		0,1,
		1,0,
		1,1,
		};

	double data_out[] = {0,1,1,0};

	Mat ti = {.rows = 4, .cols = 2, .data = data_in};
	Mat to = {.rows = 4, .cols = 1, .data = data_out};

	

	// the problem is in the ileri. resut is chancing with same shits
	note("before");
	FOR(2) {
		FORR(2) {
			MAT_AT(xor.a0, 0, 0) = i;
			MAT_AT(xor.a0, 0, 1) = ii;
			note("%zu ^ %zu = %lf",i, ii, ileri(xor));
		}
	}

	note("cost : %lf", cost(xor, ti, to));
	FOR(100 * 1000)
		learn(xor_d, xor, ti, to);
	note("cost : %lf", cost(xor, ti, to));
	note("after");
	FOR(2) {
		FORR(2) {
			MAT_AT(xor.a0, 0, 0) = i;
			MAT_AT(xor.a0, 0, 1) = ii;
			note("%zu ^ %zu = %lf",i, ii, ileri(xor));
		}
	}

}


int main(int argc, char *argv[]) {
	in();
	sec_main(argv[1]);
	out();
	return 0;
}
