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

	Mat delta_weight;
	Mat Particular_delta_weight;
	Mat delta_bias;
	Mat Particular_delta_bias;
	Mat error;
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
	FOR(i, m.rows) {
		FOR(ii, m.cols) {
			MAT_AT(m, i, ii) = rand_double() * (h - l) + l + 1e-1;
		}
	}

	return m;
}


void mat_print(Mat m, char *name) {
	printf("%s = [\n", name);
	FOR(i, m.rows) {
		FOR(ii, m.cols) {
			printf("\t%lf", MAT_AT(m, i, ii));
		}
		printf("\n");
	}
	printf("]\n");
}

#define mat_p(m) mat_print(m, #m)

void mat_sum(Mat a, Mat b) {
	if (a.cols != b.cols || a.rows != b.rows) die("invalid matrix to sum");
	FOR(i, a.rows) {
		FOR(ii, a.cols) {
			MAT_AT(a, i, ii) += MAT_AT(b, i, ii);
		}
	}

}

void mat_sub(Mat a, Mat b) {
	if (a.cols != b.cols || a.rows != b.rows) die("invalid matrix to sum");
	FOR(i, a.rows) {
		FOR(ii, a.cols) {
			MAT_AT(a, i, ii) -= MAT_AT(b, i, ii);
		}
	}

}


void mat_fill(Mat m, double x);

void mat_dot(Mat dest, Mat a, Mat b) {
	if (a.cols != b.rows || dest.rows != a.rows || dest.cols != b.cols) die("invalid matrix to dot");
	size_t inner = a.cols;
	
	mat_fill(dest, 0);

	FOR(i, dest.rows) {
		FOR(ii, dest.cols) {
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
	FOR(i, m.rows) {
		FOR(ii, m.cols) {
			MAT_AT(m, i, ii) = sigmoidf(MAT_AT(m, i, ii));
		}
	}
}

void mat_derivative_sig(Mat m) {
	FOR(i, m.rows) {
		FOR(ii, m.cols) {
			MAT_AT(m, i, ii) = MAT_AT(m, i, ii) * (1 - MAT_AT(m, i, ii));
		}
	}
}


void mat_fill(Mat m, double x) {
	FOR(i, m.rows) {
		FOR(ii, m.cols) {
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

	FOR(i, a.rows) {
		FOR(ii, a.cols) {
			MAT_AT(a, i, ii) = MAT_AT(b, i, ii);
		}

	}
}

Mat mat_T(Mat m) {
	return (Mat) {
		.rows = m.cols,
		.cols = m.rows,
		.data = m.data
	};
	// size_t rows_backup = m->rows;
	// m->rows = m->cols;
	// m->cols = rows_backup;
}

// class Dense(Layer):
//     def __init__(self, input_size, output_size):
//         self.weights = np.random.randn(output_size, input_size)
//         self.bias = np.random.randn(output_size, 1)
// 
//     def forward(self, input):
//         self.input = input
//         return np.dot(self.weights, self.input) + self.bias
// 
//     def backward(self, output_gradient, learning_rate):
//         weights_gradient = np.dot(output_gradient, self.input.T)
//         input_gradient = np.dot(self.weights.T, output_gradient)
//         self.weights -= learning_rate * weights_gradient
//         self.bias -= learning_rate * output_gradient
//         return input_gradient
// 


typedef struct {
	Mat weight;
	Mat bias;
	
	Mat input;
	Mat output;

	Mat weight_gradient;
	Mat bias_gradient;
	Mat input_gradient;
} Dense;

Dense dense_init(size_t input_size, size_t output_size) {
	Dense dense;
	dense.weight = mat_alloc(output_size, input_size);
	dense.bias = mat_alloc(1, output_size);
	dense.input = mat_alloc(1, input_size);
	dense.output = mat_alloc(1, output_size);
	
	dense.weight_gradient = mat_alloc(output_size, input_size);
	dense.bias_gradient = mat_alloc(1, output_size);
	dense.input_gradient = mat_alloc(1, input_size);
#if 0
	mat_p(dense.weight);
	mat_p(dense.bias);
	mat_p(dense.input);
	mat_p(dense.output);
	mat_p(dense.weight_gradient);
	mat_p(dense.bias_gradient);
	mat_p(dense.input_gradient);
#endif
	return dense;
}


void dense_forward(Dense dense, Mat input) {
	mat_copy(dense.input, input);
	mat_dot(dense.output, dense.input, mat_T(dense.weight));
	mat_sum(dense.output, dense.bias);
}

void dense_backward(Dense dense, Mat gradient, double rate) {
	mat_dot(dense.weight_gradient, mat_T(gradient), dense.input);
	mat_dot(dense.input_gradient, gradient, dense.weight);
	FOR(i, dense.weight.rows) {
		FOR(j, dense.weight.cols) {
			MAT_AT(dense.weight, i, j) -= MAT_AT(dense.weight_gradient, i, j) * rate;
		}
	}
	FOR(i, dense.bias.rows) {
		FOR(j, dense.bias.cols) {
			MAT_AT(dense.bias, i, j) -= MAT_AT(dense.bias_gradient, i, j) * rate;
		}
	}
}

Mat network_forward(Dense *dense, size_t dense_count, Mat in) {
	dense_forward(dense[0], in);
 	FOR(dc, dense_count - 1) {
 		dense_forward(dense[dc + 1], dense[dc].output);
 	}
	return dense[dense_count - 1].output;
}

void network_backward(Dense *dense, size_t dense_count, Mat in, Mat out) {
	double error = 0.0f;
	double rate = 0.0000001f;

 	dense_forward(dense[0], in);
 	FOR(dc, dense_count - 1) {
 		dense_forward(dense[dc + 1], dense[dc].output);
 	}
 	// mat_p(dense[dense_count - 1].output);
	FOR(i, dense[dense_count - 1].output.cols) {
		double d = MAT_AT(dense[dense_count - 1].output, 0, i) - MAT_AT(out, 0, i);
		error += d*d;
		MAT_AT(dense[dense_count - 1].output, 0, i) = 2 * (MAT_AT(dense[dense_count - 1].output, 0, i) - MAT_AT(out, 0, i)); // calculate gradient of output layer
	}
	// plf(error);
	error = 0.0f;

	dense_backward(dense[dense_count - 1], dense[dense_count - 1].output, rate); // this will set dense[dense_count - 1].input_gradient
	for (size_t dc = dense_count - 1; dc > 0; dc--) {
		dense_backward(dense[dc - 1], dense[dc].input_gradient, rate);
	}
}

