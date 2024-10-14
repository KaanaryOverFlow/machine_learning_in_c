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

#define MAT_AT(m, r, c) (m).data[(r)*(m).cols + (c)]
#define ARRAY_LEN(x) sizeof(x)/sizeof(x[0])



double rand_double(void) {
	return (double) rand() / (double) RAND_MAX;
}

Mat mat_alloc(size_t rows, size_t cols) {
	double h = 1.0f;
	double l = -1.0f;
	Mat m;
	m.rows = rows;
	m.cols = cols;
	m.data = malloc(sizeof(*m.data)*rows*cols);
	if (m.data == NULL) die("out of memory");
	srand(time(NULL));
	FOR(i, m.rows) {
		FOR(ii, m.cols) {
			MAT_AT(m, i, ii) = rand_double() * (h - l) + l;
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
			for (unsigned int iii = 0; iii < inner; iii++) {
				
				MAT_AT(dest, i, ii) += MAT_AT(a, i, iii) * MAT_AT(b, iii, ii);

			}
		}
	}
}

double sigmoidf(double x) {
	return 1 / (1 + exp(-x));
}

#define LowerValue 0.001f

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

#define RELU_VALUE 0.01f

void mat_relu(Mat m) {
	FOR(i, m.rows) {
		FOR(ii, m.cols) {
			double x = MAT_AT(m, i, ii);
			if (x <= 0.0f)
				MAT_AT(m, i, ii) = x * RELU_VALUE; 
			else
				MAT_AT(m, i, ii) = x; 
		}
	}
}

void mat_derivative_relu(Mat m) {
	FOR(i, m.rows) {
		FOR(ii, m.cols) {
			double x = MAT_AT(m, i, ii);
			if (x > 0) {
				MAT_AT(m, i, ii) = 1; 
			} else {
				MAT_AT(m, i, ii) = RELU_VALUE; 
			}
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
}

void mat_shuffle_one(Mat m) {
        FOR(i, m.rows) {
                size_t r = i + rand()%(m.rows - i);
                if (i != r) {
                        FOR(ii, m.cols) {
                                double temp = MAT_AT(m, i, ii);
                                MAT_AT(m, i, ii) = MAT_AT(m, r, ii);
                                MAT_AT(m, r, ii) = temp;
                 	}               
                }
        }
}


void mat_shuffle(Mat m, Mat n) {
        FOR(i, m.rows) {
                size_t r = i + rand()%(m.rows - i);
                if (i != r) {
                        FOR(ii, m.cols) {
                                double temp = MAT_AT(m, i, ii);
                                MAT_AT(m, i, ii) = MAT_AT(m, r, ii);
                                MAT_AT(m, r, ii) = temp;
                 	}               
                        FOR(ii, n.cols) {
                                double temp = MAT_AT(n, i, ii);
                                MAT_AT(n, i, ii) = MAT_AT(n, r, ii);
                                MAT_AT(n, r, ii) = temp;
                        }
                }
        }
}

int is_mat_nan(Mat m) {

        FOR(i, m.rows) {

        	FOR(ii, m.cols) {
                                if (MAT_AT(m, i, ii) != MAT_AT(m, i, ii)) return 1;

		}
	}
	return 0;
}

typedef enum {
	SIGMOID,
	RELU,
} ACTS;

typedef struct {
	
	
	// dense part
	Mat weight;
	Mat bias;
	
	Mat input;
	Mat output;


	Mat dw;
	Mat db;
	
	Mat input_gradient;
	Mat temp_dw;
	Mat temp_db;

	// act part
	ACTS act;
	void (*activate)(Mat m);
	void (*derivative_activate)(Mat m);


} Dense;

Dense create_dense(size_t input_size, size_t output_size, ACTS act) {
	Dense dense;

	dense.weight = mat_alloc(input_size, output_size);
	dense.bias = mat_alloc(1, output_size);
	
	dense.input = mat_alloc(1, input_size);
	dense.output = mat_alloc(1, output_size);
	
	dense.dw = mat_alloc(input_size, output_size);
	dense.db = mat_alloc(1, output_size);
	
	dense.input_gradient = mat_alloc(1, input_size);
	dense.temp_dw = mat_alloc(input_size, output_size);
	dense.temp_db = mat_alloc(1, output_size);
	
	
	switch (act) {
		case SIGMOID:
			dense.activate = &mat_sig;
			dense.derivative_activate = &mat_derivative_sig;
			break;
		case RELU:
			dense.activate = &mat_relu;
			dense.derivative_activate = &mat_derivative_relu;
			break;
		default:
			die("wrong action");
			break;

	}
	dense.act = act;
	
	return dense;
}

void dense_forward(Dense dense, Mat input) {
	mat_copy(dense.input, input);
	mat_dot(dense.output, dense.input, dense.weight);
	mat_sum(dense.output, dense.bias);

	dense.activate(dense.output);

}

void dense_backward(Dense dense, Mat grad) {
	dense.derivative_activate(dense.output);

	FOR(i, grad.cols)
		MAT_AT(grad, 0, i) *= MAT_AT(dense.output, 0, i);
	
	mat_dot(dense.temp_dw, mat_T(dense.input), grad);
	mat_copy(dense.temp_db, grad);
	mat_dot(dense.input_gradient, grad, mat_T(dense.weight));
	

	mat_sum(dense.dw, dense.temp_dw);
	mat_sum(dense.db, dense.temp_db);
}

void dense_apply(Dense dense, double lr) {
	FOR(r, dense.weight.rows) {
		FOR(c, dense.weight.cols) {
			MAT_AT(dense.weight, r, c) -= MAT_AT(dense.dw, r, c) * lr;
		}
	}


	FOR(r, dense.bias.rows) {
		FOR(c, dense.bias.cols) {
			MAT_AT(dense.bias, r, c) -= MAT_AT(dense.db, r, c) * lr;
		}
	}
	

}

typedef struct {
	size_t count;
	Dense *dense;
	Mat grad;
} NN;

void nn_rand(NN nn) {
	double h = 1.0f;
	double l = -1.0f;
	
	FOR(i, nn.count) {

		FOR(r, nn.dense[i].weight.rows) {
			FOR(c, nn.dense[i].weight.cols) {
				MAT_AT(nn.dense[i].weight, r, c) =rand_double() * (h - l) + l; 
			}
		}
	
		FOR(r, nn.dense[i].bias.rows) {
			FOR(c, nn.dense[i].bias.cols) {
				MAT_AT(nn.dense[i].bias, r, c) =rand_double() * (h - l) + l; 
			}
		}

	}
}

void nn_load(NN nn, char *name) {
	int fd = open(name, O_CREAT | O_RDWR);
	if (fd < 0) die("no saved network to load");
	FOR(i, nn.count) {
		FOR(r, nn.dense[i].weight.rows) {
			FOR(c, nn.dense[i].weight.cols) {
				read(fd,
						&MAT_AT(nn.dense[i].weight, r, c)
					,8);
			}
		}
		FOR(r, nn.dense[i].bias.rows) {
			FOR(c, nn.dense[i].bias.cols) {
				read(fd,
						&MAT_AT(nn.dense[i].bias, r, c)
					,8);
			}
		}

	}
	close(fd);
}


void nn_save(NN nn, char *name) {
	int fd = open(name, O_CREAT | O_RDWR, 0666);
	FOR(i, nn.count) {
		FOR(r, nn.dense[i].weight.rows) {
			FOR(c, nn.dense[i].weight.cols) {
				write(fd,
						&MAT_AT(nn.dense[i].weight, r, c)
					,8);
			}
		}
		FOR(r, nn.dense[i].bias.rows) {
			FOR(c, nn.dense[i].bias.cols) {
				write(fd,
						&MAT_AT(nn.dense[i].bias, r, c)
					,8);
			}
		}

	}
	close(fd);
}

#define nn_out(nn) (nn).dense[(nn).count - 1].output

void nn_forward(NN nn, Mat input) {

	FOR(dense_index, nn.count) {
		if(dense_index == 0) {
			dense_forward(nn.dense[dense_index], input);
		} else {
			dense_forward(nn.dense[dense_index], nn.dense[dense_index - 1].output);
		}
	}
}

void nn_backward(NN nn, Mat grad) {
	for (size_t index = nn.count; index > 0; index--) {
		if (index == nn.count) {
			dense_backward(nn.dense[index - 1], nn.grad);
		} else {
			dense_backward(nn.dense[index - 1], nn.dense[index].input_gradient);
		}
		
	}
}

double nn_cost(NN network, Mat input, Mat output) {
		double err = 0;
		FOR(ic, input.rows) {
			nn_forward(network, mat_row(input, ic));
			FOR(i, nn_out(network).cols) {
				double t = MAT_AT(nn_out(network), 0, i) - MAT_AT(output, ic, i);
				err += t * t;
			}
		}
		
		err /= input.rows;
		return err;
	
}

void nn_update(NN network, double lr, Mat input, Mat output) {
	
	FOR(index, network.count) {
		mat_fill(network.dense[index].dw, 0);
		mat_fill(network.dense[index].db, 0);
	}
	
	// double err = 0;
	FOR(ic, input.rows) {
		nn_forward(network, mat_row(input, ic));

	
		FOR(i, nn_out(network).cols) {
			double t = MAT_AT(nn_out(network), 0, i) - MAT_AT(output, ic, i);
			// err += t * t;
			if (network.dense[network.count - 1].act == RELU)
				MAT_AT(network.grad, 0, i) = t;
			else if (network.dense[network.count -1].act == SIGMOID)
				MAT_AT(network.grad, 0, i) = 2 * t;
			else
				die("invalid activation function");
		}
		
		nn_backward(network, network.grad);
	}
	
	// err /= input.rows;
	// plf(err);
	

	FOR(index, network.count) {
		FOR(r, network.dense[index].weight.rows) {
			FOR(c, network.dense[index].weight.cols) {
				MAT_AT(network.dense[index].dw, r, c) /= input.rows;
			}
			
		}
		
		FOR(r, network.dense[index].bias.rows) {
			FOR(c, network.dense[index].bias.cols) {
				MAT_AT(network.dense[index].db, r, c) /= input.rows;
			}
			
		}
	}


	FOR(index, network.count) {
		dense_apply(network.dense[index], lr);
	}

}


void nn_fit_callback(NN network, size_t epoch, double lr, size_t batch_size, void (*callback)(NN net, size_t x, size_t y, size_t u, size_t bc, Mat in, Mat out), Mat inputs, Mat outputs) {
	size_t n = inputs.rows;
	
	assert(batch_size <= n);
	
	size_t last = n % batch_size;

	size_t batch_count = n / batch_size;

	if (last)
		batch_count++;

		
	

	FOR(epoch_, epoch) {

		mat_shuffle_one(inputs);

		FOR(k, batch_count) {
				size_t size = batch_size;
				if (last && (k * batch_size + batch_size) > n ){
					size = last;
				}
	
				Mat mini_batch_in = {
					.rows = size, 
					.cols = inputs.cols,
					.data = &MAT_AT(inputs, k * batch_size, 0)
				};
		
				Mat mini_batch_out = {
					.rows = size, 
					.cols = outputs.cols,
					.data = &MAT_AT(outputs, k * batch_size, 0)
				};
	
			nn_update(network, lr, mini_batch_in, mini_batch_out);
			callback(
					network,
					epoch_,
					epoch,
					k,
					batch_count,
					mini_batch_in, 
					mini_batch_out
				);
		}
	}

}

void nn_fit(NN network, size_t epoch, double lr, size_t batch_size, Mat inputs, Mat outputs) {
	
	size_t n = inputs.rows;
	
	assert(batch_size <= n);
	
	size_t last = n % batch_size;

	size_t batch_count = n / batch_size;

	if (last)
		batch_count++;

		
	

	FOR(epoch_, epoch) {
		
		// mat_shuffle(inputs, outputs);

			
		/* For autoencoder */ mat_shuffle_one(inputs);


		FOR(k, batch_count) {
			size_t size = batch_size;
			if (last && (k * batch_size + batch_size) > n ){
				size = last;
			}

			Mat mini_batch_in = {
				.rows = size, 
				.cols = inputs.cols,
				.data = &MAT_AT(inputs, k * batch_size, 0)
			};
	
			Mat mini_batch_out = {
				.rows = size, 
				.cols = outputs.cols,
				.data = &MAT_AT(outputs, k * batch_size, 0)
			};
			
		// 	plu(size);
		// 	mat_p(mini_batch_in);
		// 	mat_p(mini_batch_out);
		// 	getchar();

			
			nn_update(network, lr, mini_batch_in, mini_batch_out);
		}
#ifdef NN_DEBUG
		note("epoch %lu / %lu\t\tloss : %lf",epoch_, epoch, nn_cost(network, inputs, outputs));
#endif

	}

}

void nn_train(NN network, size_t epoch, double lr, Mat input, Mat output) {

	FOR(epoch_, epoch) {

		nn_update(network, lr, input, output);
		continue;
	
		FOR(index, network.count) {
			mat_fill(network.dense[index].dw, 0);
			mat_fill(network.dense[index].db, 0);
		}
		
		double err = 0;
		FOR(ic, input.rows) {
			nn_forward(network, mat_row(input, ic));
	
		
			FOR(i, nn_out(network).cols) {
				double t = MAT_AT(nn_out(network), 0, i) - MAT_AT(output, ic, i);
				err += t * t;
				MAT_AT(network.grad, 0, i) = 2 * t;
			}
			
			nn_backward(network, network.grad);
		}
		
		err /= input.rows;
		plf(err);
		
	
		FOR(index, network.count) {
			FOR(r, network.dense[index].weight.rows) {
				FOR(c, network.dense[index].weight.cols) {
					MAT_AT(network.dense[index].dw, r, c) /= input.rows;
				}
				
			}
			
			FOR(r, network.dense[index].bias.rows) {
				FOR(c, network.dense[index].bias.cols) {
					MAT_AT(network.dense[index].db, r, c) /= input.rows;
				}
				
			}
		}
	
	
		FOR(index, network.count) {
			dense_apply(network.dense[index], lr);
		}
	}

}

Mat average_pooling_1D(size_t sample_size, Mat input) {
	
	size_t new_cols = input.cols / sample_size;
	
	// assert(input.rows == 1);
	assert(input.cols % sample_size == 0);
	
	
	Mat output = mat_alloc(input.rows, new_cols);

	FOR(row, input.rows) {

		FOR(i, new_cols) {
			double temp = 0;
			FOR(j, sample_size) {
				temp += MAT_AT(input, row, i * sample_size + j);
			}
			MAT_AT(output, row, i) = temp / sample_size;
		}

	}



	return output;

}

Mat upsample(Mat input, size_t sample_size) {
	
	size_t new_cols = input.cols * sample_size;
	
	Mat output = mat_alloc(input.rows, new_cols);

	FOR(row, input.rows) {
	
		FOR(i, output.cols) {
			MAT_AT(output, row, i) = MAT_AT(input, row, i / sample_size);
		}
	}

	return output;

}
