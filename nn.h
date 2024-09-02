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

void mat_T(Mat *m) {
	size_t rows_backup = m->rows;
	m->rows = m->cols;
	m->cols = rows_backup;
}


Layer layer_alloc(size_t a, size_t b) {

	Layer l;

	l.weight = mat_alloc(a, b);
	l.output = mat_alloc(1, b);
	l.input = mat_alloc(1, a);
	l.bias = mat_alloc(1, b);
	
	l.delta_weight = mat_alloc(a, b);
	l.Particular_delta_weight = mat_alloc(a, b);
	
	l.delta_bias = mat_alloc(1, b);
	l.Particular_delta_bias = mat_alloc(1, b);
	
	l.error = mat_alloc(1, b);



	return l;

}

NN nn_alloc(size_t input_count, size_t *arch, size_t len) {
	NN nn;

	nn.layer = malloc(len * sizeof(Layer));
	assert(nn.layer != NULL);
	
	nn.input = mat_alloc(1, input_count);


	nn.layer[0] = layer_alloc(input_count, arch[0]);


	FOR(i, len - 1) {
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

	FOR(i, nn.layer_count ) {
	
		layer_forward(nn.layer[i + 1], nn.layer[i].output);
	
	}
	
}

double nn_cost(NN nn, Mat in, Mat out) {

	assert(out.cols == nn_out(nn).cols);
	assert(in.rows == out.rows);

	double cost = 0.0f;
	
	FOR(count, in.rows) {
	
		mat_copy(nn.input, mat_row(in, count));
	
		nn_forward(nn);
	
		FOR(i, out.cols) {
			double distance = MAT_AT(nn_out(nn), 0, i) - MAT_AT(out, count, i);
			cost += distance * distance;
		}

	}


	return cost / in.rows;
}

void nn_delta(NN nn, Mat in, Mat out) {
	

	assert(in.rows == out.rows);

	FOR(lc, nn.layer_count) {
		mat_fill(nn.layer[lc].delta_weight, 0);
		mat_fill(nn.layer[lc].delta_bias, 0);
	}

	// FOR(input_count, in.rows) {

		mat_copy(nn.input, in); // mat_row(in, input_count));
	
		nn_forward(nn);
		
		for(size_t lc = nn.layer_count; lc + 1 > 0; lc--) {
			
			mat_fill(nn.layer[lc].error, 0);

			if ( lc == nn.layer_count) {
			FOR(i, nn.layer[lc].output.cols) {
			
				
				MAT_AT(nn.layer[lc].error, 0, i) = 2 * (MAT_AT(nn.layer[lc].output, 0, i) - MAT_AT(out, 0, i)) * MAT_AT(nn.layer[lc].output, 0, i) * ( 1 - MAT_AT(nn.layer[lc].output, 0, i)); // sigma_o
			}
			} else {
				mat_T(&nn.layer[lc + 1].weight);
				mat_dot(nn.layer[lc].error, nn.layer[lc + 1].error, nn.layer[lc + 1].weight);
				mat_T(&nn.layer[lc + 1].weight);
				

				mat_derivative_sig(nn.layer[lc + 1].input);
				assert(nn.layer[lc].error.cols == nn.layer[lc + 1].input.cols);
				FOR(j, nn.layer[lc].error.cols)
					MAT_AT(nn.layer[lc].error, 0, j) *= MAT_AT(nn.layer[lc + 1].input, 0, j);
			}
		

		 	mat_T(&nn.layer[lc].input);
			mat_dot(nn.layer[lc].delta_weight, nn.layer[lc].input, nn.layer[lc].error);
			mat_T(&nn.layer[lc].input);
		
			

			mat_fill(nn.layer[lc].Particular_delta_bias, 0);
			FOR(i, nn.layer[lc].delta_bias.cols) {
				FOR(j, nn.layer[lc].delta_weight.rows)
					MAT_AT(nn.layer[lc].delta_bias, 0, i) += MAT_AT(nn.layer[lc].Particular_delta_weight, j, i);
			}

			// mat_sum(nn.layer[lc].delta_weight, nn.layer[lc].Particular_delta_weight);
			// mat_sum(nn.layer[lc].delta_bias, nn.layer[lc].Particular_delta_bias);

		}

	return;

	// }

	FOR(lc, nn.layer_count) {
		FOR(r, nn.layer[lc].weight.rows) {
			FOR(c, nn.layer[lc].weight.cols) {
				MAT_AT(nn.layer[lc].delta_weight, r, c) /= in.rows;
			}
		}
		FOR(r, nn.layer[lc].bias.rows) {
			FOR(c, nn.layer[lc].bias.cols) {
				MAT_AT(nn.layer[lc].delta_bias, r, c) /= in.rows;
			}
		}
	}



}

void nn_learn(NN nn, double rate) {
	FOR(lc, nn.layer_count) {
		FOR(r, nn.layer[lc].weight.rows) {
			FOR(c, nn.layer[lc].weight.cols) {
				MAT_AT(nn.layer[lc].weight, r, c) -= rate * MAT_AT(nn.layer[lc].delta_weight, r, c);
			}
		}
	FOR(r, nn.layer[lc].bias.rows) {
			FOR(c, nn.layer[lc].bias.cols) {
				MAT_AT(nn.layer[lc].bias, r, c) -= rate * MAT_AT(nn.layer[lc].delta_bias, r, c);
			}
		}
	}
}

void Fnn_delta(NN nn, Mat inp, Mat out) {

	double eps = 1e-3;
	double rate = 1;
	



	double current_cost = nn_cost(nn, inp, out);


	FOR(i, nn.layer_count) {
		Layer l = nn.layer[i];
		mat_fill(l.delta_weight, 0);
		mat_fill(l.delta_bias, 0);
		FOR(r, l.weight.rows) {
			FOR(c, l.weight.cols) {
				double saved = MAT_AT(l.weight, r, c);
				MAT_AT(l.weight, r, c) += eps;
				MAT_AT(l.delta_weight, r, c) = (nn_cost(nn, inp, out) - c)/eps;
				MAT_AT(l.weight, r, c) = saved;
				
				saved = MAT_AT(l.bias, r, c);
				MAT_AT(l.bias, r, c) += eps;
				MAT_AT(l.delta_bias, r, c) = (nn_cost(nn, inp, out) - c)/eps;
				MAT_AT(l.bias, r, c) = saved;



			}
		}
	}

	FOR(i, nn.layer_count) {
		Layer l = nn.layer[i];
		FOR(r, l.weight.rows) {
			FOR(c, l.weight.cols) {
				MAT_AT(l.weight, r, c) -= rate * MAT_AT(l.delta_weight, r, c);
			}
		}
	}



}

