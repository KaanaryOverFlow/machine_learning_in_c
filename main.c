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
	double h = 1;
	double l = 0;
	Mat m;
	m.rows = rows;
	m.cols = cols;
	m.data = malloc(sizeof(*m.data)*rows*cols);
	if (m.data == NULL) die("out of memory");
	srand(time(NULL));
	FOR(m.rows) {
		FORR(m.cols) {
			MAT_AT(m, i, ii) = rand_double() * (h - l) + l + 1e-1;
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
	return 1 / (1 + exp(-x));
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

Mat mat_row(Mat m, size_t row) {
	return (Mat) {
		.rows = 1,
		.cols = m.cols,
		.data = &MAT_AT(m, row, 0)
	};
}

void mat_copy(Mat a, Mat b) {
	if (a.cols != b.cols || a.rows != b.rows) die("invalid mat to copy");

	FOR(a.rows) {
		FORR(a.cols) {
			MAT_AT(a, i, ii) = MAT_AT(b, i, ii);
		}

	}
}



typedef struct {
	size_t count;
	Mat *ws;
	Mat *bs;
	Mat *as;

	Mat *dw;
	Mat *db;
	Mat *gas;
} NN;

#define ARRAY_LEN(x) sizeof(x)/sizeof(x[0])

NN nn_alloc(size_t *arch, size_t count) {
	NN nn;
	nn.count = count - 1;
	nn.ws = malloc(sizeof(Mat) * nn.count);
	nn.bs = malloc(sizeof(Mat) * nn.count);
	nn.as = malloc(sizeof(Mat) * count);
	
	nn.dw = malloc(sizeof(Mat) * nn.count);
	nn.db = malloc(sizeof(Mat) * nn.count);
	nn.gas = malloc(sizeof(Mat) * count);

	nn.as[0] = mat_alloc(1, arch[0]);
	nn.gas[0] = mat_alloc(1, arch[0]);
	mat_fill(nn.gas[0], 0);

	FOR(nn.count) {
		nn.ws[i] = mat_alloc(nn.as[i].cols, arch[i+1]);
		nn.bs[i] = mat_alloc(1 , arch[i+1]);
		nn.as[i+1] = mat_alloc(1 , arch[i+1]);
		
		nn.dw[i] = mat_alloc(nn.as[i].cols, arch[i+1]);
		nn.db[i] = mat_alloc(1 , arch[i+1]);
		nn.gas[i+1] = mat_alloc(1 , arch[i+1]);


		mat_fill(nn.dw[i], 0);
		mat_fill(nn.db[i], 0);
		mat_fill(nn.gas[i + 1], 0);
		
	}

	return nn;
}

void nn_print(NN nn, char *name) {
	note("dumping %s NN", name);
	FOR(nn.count) {
		mat_p(nn.ws[i]);
		mat_p(nn.bs[i]);
	}
}

#define nn_p(a) nn_print(a, #a);
#define nn_in(nn) nn.as[0]
#define nn_out(nn) nn.as[nn.count]

void nn_ileri(NN nn) {
	FOR(nn.count) {
		mat_dot(nn.as[i + 1], nn.as[i], nn.ws[i]);
		mat_sum(nn.as[i + 1], nn.bs[i]);
		mat_sig(nn.as[i + 1]);
	}
}

double nn_cost(NN nn, Mat ti, Mat to) {
	double cost = 0;
	if (ti.rows != to.rows || to.cols != nn_out(nn).cols) die("invalied ti and to to cost");
	
	FOR(ti.rows) {
		Mat in = mat_row(ti, i);
		Mat out = mat_row(to, i);
		mat_copy(nn_in(nn), in);
		nn_ileri(nn);
		FORR(nn_out(nn).cols) {
			double d = MAT_AT(nn_out(nn), 0, ii) - MAT_AT(out, 0, ii);
			cost += d * d;
		}

	}
	
	return cost/ti.rows;
}

void nn_learn(NN nn, double eps, double rate, Mat ti, Mat to) {
	double saved = 0;
	double cost = nn_cost(nn, ti, to);

	FOR(nn.count) {
		FORR(nn.ws[i].rows) {
			for (unsigned long iii = 0; iii < nn.ws[i].cols; iii++) {
				saved = MAT_AT(nn.ws[i], ii, iii);
				MAT_AT(nn.ws[i], ii, iii) += eps;
				MAT_AT(nn.dw[i], ii, iii) = (nn_cost(nn, ti, to) - cost)/eps;
				MAT_AT(nn.ws[i], ii, iii) = saved;
			}
		}
		
		FORR(nn.bs[i].rows) {
			for (unsigned long iii = 0; iii < nn.bs[i].cols; iii++) {
				saved = MAT_AT(nn.bs[i], ii, iii);
				MAT_AT(nn.bs[i], ii, iii) += eps;
				MAT_AT(nn.db[i], ii, iii) = (nn_cost(nn, ti, to) - cost)/eps;
				MAT_AT(nn.bs[i], ii, iii) = saved;
			}
		}
	}
	
	FOR(nn.count) {
		FORR(nn.ws[i].rows) {
			for (unsigned long iii = 0; iii < nn.ws[i].cols; iii++) {
				MAT_AT(nn.ws[i], ii, iii) -= rate * MAT_AT(nn.dw[i], ii, iii);
			}
		}
		
		FORR(nn.bs[i].rows) {
			for (unsigned long iii = 0; iii < nn.bs[i].cols; iii++) {
				MAT_AT(nn.bs[i], ii, iii) -= rate * MAT_AT(nn.db[i], ii, iii);
			}
		}
	}
}

void commit_diff(NN nn, double rate) {
	FOR(nn.count) {
		for (size_t row = 0; row < nn.ws[i].rows; row++) {
			for (size_t col = 0; col < nn.ws[i].cols; col++) {
				MAT_AT(nn.ws[i], row, col) -= rate *  MAT_AT(nn.dw[i], row, col);
			}
		}
		
		for (size_t row = 0; row < nn.bs[i].rows; row++) {
			for (size_t col = 0; col < nn.bs[i].cols; col++) {
				MAT_AT(nn.bs[i], row, col) -= rate *  MAT_AT(nn.db[i], row, col);
			}
		}
	}
}


void nn_backprop(NN nn, double rate, Mat input, Mat output) {
	
	mat_fill(nn.gas[0], 0);
	
	FOR(nn.count) {
		mat_fill(nn.dw[i], 0);
		mat_fill(nn.db[i], 0);
		mat_fill(nn.gas[i + 1], 0);
	}
	
	FOR(input.rows) {

		mat_copy(nn_in(nn), mat_row(input, i));
		nn_ileri(nn);
		
		FORR(nn.count) {
			mat_fill(nn.gas[ii], 0);

		}
		
		FORR(output.cols) {
			MAT_AT(nn.gas[nn.count], 0, ii) = MAT_AT(nn_out(nn), 0, ii) - MAT_AT(output, i, ii);
		}
		for (size_t layer = nn.count; layer > 0; layer--) {
			for (size_t col = 0; col < nn.as[layer].cols; col++) {
				double a = MAT_AT(nn.as[layer], 0, col);
				double da = MAT_AT(nn.gas[layer], 0, col);
				MAT_AT(nn.db[layer - 1], 0, col) += 2 * da * a * (1 - a);
				for (size_t row = 0; row < nn.as[layer - 1].cols; row++) {
					double pa = MAT_AT(nn.as[layer - 1], row, col);
					double w = MAT_AT(nn.ws[layer - 1], 0, row);
					MAT_AT(nn.dw[layer - 1], row, col) += 2 * da * a * (1 - a) * pa;
					MAT_AT(nn.gas[layer - 1], 0, row) += 2 * da * a * (1 - a) * w;

				}


			}
		}
	}
	

	FOR(nn.count) {
		for (size_t row = 0; row < nn.ws[i].rows; row++) {
			for (size_t col = 0; col < nn.ws[i].cols; col++) {
				MAT_AT(nn.dw[i], row, col) /= input.rows;
			}
		}
		
		for (size_t row = 0; row < nn.bs[i].rows; row++) {
			for (size_t col = 0; col < nn.bs[i].cols; col++) {
				MAT_AT(nn.db[i], row, col) /= input.rows;
			}
		}
	}
	
	commit_diff(nn, rate);

	
}



void my_learn(NN nn, double rate, Mat input, Mat output) {
	
	mat_fill(nn.gas[0], 0);
	
	FOR(nn.count) {
		mat_fill(nn.dw[i], 0);
		mat_fill(nn.db[i], 0);
		mat_fill(nn.gas[i + 1], 0);
	}
	
	FOR(input.rows) {

		mat_copy(nn_in(nn), mat_row(input, i));
		nn_ileri(nn);
	
		mat_fill(nn.gas[0], 0);
		
		FORR(nn.count) {
			mat_fill(nn.gas[ii], 0);

		}

		
		FORR(output.cols) {
			MAT_AT(nn.gas[nn.count], 0, ii) = MAT_AT(nn_out(nn), 0, ii) - MAT_AT(output, i, ii);
		}
		
		for (size_t layer = nn.count; layer > 0; layer--) {
			for (size_t col = 0; col < nn.as[layer].cols; col++) { // çıktıda ne kadar nöron varsa o kadar dön
				double delta_predict_cost = 2 * MAT_AT(nn.gas[layer], 0, col);
				double delta_sigmoid = MAT_AT(nn.as[layer], 0, col) * (1 - MAT_AT(nn.as[layer], 0, col));
				// phex(delta_predict_cost);
				// phex(delta_sigmoid);
				if (!delta_predict_cost) {
					note("delta_predict_cost sıfır 2 * %lf", MAT_AT(nn.gas[nn.count], 0, col));
					mat_p(nn.gas[nn.count]);
					getchar();
				}
				if (!delta_sigmoid) {
					note("delta_sigmoid sıfır");
					getchar();
				}
				MAT_AT(nn.db[layer - 1], 0, col) += delta_predict_cost * delta_sigmoid;
				// phex(MAT_AT(nn.db[layer - 1], 0, col));
				for (size_t input_count = 0; input_count < nn.as[layer - 1].cols; input_count++) { // mevcut nöron kaç tane input almışsa o kadar dön
					double in = MAT_AT(nn.as[layer - 1], 0, input_count);
					double w = MAT_AT(nn.ws[layer - 1], input_count, col);
					
					MAT_AT(nn.dw[layer - 1], input_count, col) += delta_predict_cost * delta_sigmoid * in;
					MAT_AT(nn.gas[layer - 1], 0, input_count) += delta_predict_cost * delta_sigmoid * w;
				
				}

			}
		}
		
	}

	FOR(nn.count) {
		for (size_t row = 0; row < nn.ws[i].rows; row++) {
			for (size_t col = 0; col < nn.ws[i].rows; col++) {
				// MAT_AT(nn.dw[i], row, col) /= input.rows;
			}
		}
		for (size_t row = 0; row < nn.bs[i].rows; row++) {
			for (size_t col = 0; col < nn.bs[i].rows; col++) {
				// MAT_AT(nn.db[i], row, col) /= input.rows;
			}
		}
	}

	commit_diff(nn, rate);

}

void print_delta(NN nn) {
	mat_p(nn.gas[0]);
	
	FOR(nn.count) {
		mat_p(nn.dw[i]);
		mat_p(nn.db[i]);
		mat_p(nn.gas[i + 1]);
	}
}

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
	double rate = 1e-1;

	do {
	note("cost : %lf", nn_cost(nn, ti, to));
	FOR(3000)
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
