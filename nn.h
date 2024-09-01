#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <time.h>
#include <math.h>
#include <assert.h>

typedef struct {
	size_t rows;
	size_t cols;
	double *data;

} Mat;

typedef struct {
	Mat weight;
	Mat bias;
	Mat input;
	Mat output;
} Layer;

typedef struct {
	size_t layer_count;
	Mat input;
	Layer *layer;
} NN;

#define MAT_AT(m, r, c) (m).data[(r)*(m).cols + (c)]
#define ARRAY_LEN(x) sizeof(x)/sizeof(x[0])
#define nn_p(a) nn_print(a, #a);

#define nn_out(nn) (nn).layer[(nn).layer_count].output

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
	FOR(i, m.rows)
		FOR(ii, m.cols)
			MAT_AT(m, i, ii) = rand_double() * (h - l) + l + 1e-1;
		}
	}

	return m;
}


void mat_print(Mat m, char *name) {
	printf("%s = [\n", name);
	For(i, m.rows) {
		For(ii, m.cols) {
			printf("\t%lf", MAT_AT(m, i, ii));
		}
		printf("\n");
	}
	printf("]\n");
}

#define mat_p(m) mat_print(m, #m)

void mat_sum(Mat a, Mat b) {
	if (a.cols != b.cols || a.rows != b.rows) die("invalid matrix to sum");
	FOR(i, a.rows)
		FOR(ii, a.cols)
			MAT_AT(a, i, ii) += MAT_AT(b, i, ii);
		}
	}

}

void mat_fill(Mat m, double x);

void mat_dot(Mat dest, Mat a, Mat b) {
	if (a.cols != b.rows || dest.rows != a.rows || dest.cols != b.cols) die("invalid matrix to dot");
	size_t inner = a.cols;
	
	mat_fill(dest, 0);

	FOR(i, dest.rows)
		FOR(ii, dest.cols)
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
	FOR(i, m.rows)
		FOR(ii, m.cols)
			MAT_AT(m, i, ii) = sigmoidf(MAT_AT(m, i, ii));
		}
	}
}

void mat_fill(Mat m, double x) {
	FOR(i, m.rows)
		FOR(ii, m.cols)
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

	FOR(i, a.rows)
		FOR(ii, a.cols)
			MAT_AT(a, i, ii) = MAT_AT(b, i, ii);
		}

	}
}



Layer layer_alloc(size_t a, size_t b) {

	Layer l;

	l.weight = mat_alloc(a, b);
	l.output = mat_alloc(1, b);
	l.input = mat_alloc(1, a);
	l.bias = mat_alloc(1, b);


	return l;

}

NN nn_alloc(size_t input_count, size_t *arch, size_t len) {
	NN nn;

	nn.layer = malloc(len * sizeof(Layer));
	assert(nn.layer != NULL);
	
	nn.input = mat_alloc(1, input_count);


	nn.layer[0] = layer_alloc(input_count, arch[0]);


	For(i, len - 1) {
		nn.layer[i + 1] = layer_alloc(arch[i], arch[i + 1]);
	}
	nn.layer_count = len - 1;
	return nn;
}

void layer_forward(Layer l, Mat layer_input) {

	mat_copy(l.input, layer_input);
	mat_dot(l.output, l.input, l.weight);
	mat_sum(l.output, l.bias);
	mat_sig(l.output);


}

void nn_forward(NN nn) {

	
	layer_forward(nn.layer[0], nn.input);

	For(i, nn.layer_count ) {
	
		layer_forward(nn.layer[i + 1], nn.layer[i].output);
	
	}
	
}

double nn_cost(NN nn, Mat in, Mat out) {

	assert(out.cols == nn_out(nn).cols);

	mat_copy(nn.input, in);

	nn_forward(nn);

	double cost = 0.0f;
	FOR(i, out.cols)
		double distance = MAT_AT(nn_out(nn), 0, i) - MAT_AT(out, 0, i);
		cost += distance * distance;
	}

	plf(cost);



	return cost;
}


