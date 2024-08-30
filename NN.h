


typedef struct {
	size_t rows;
	size_t cols;
	double *data;

} Mat;

typedef struct {
	size_t count;
	Mat *ws;
	Mat *bs;
	Mat *as;

	Mat *dw;
	Mat *db;
	Mat *gas;
} NN;

#define MAT_AT(m, r, c) (m).data[(r)*(m).cols + (c)]
#define ARRAY_LEN(x) sizeof(x)/sizeof(x[0])
#define nn_p(a) nn_print(a, #a);
#define nn_in(nn) nn.as[0]
#define nn_out(nn) nn.as[nn.count]




Mat mat_alloc(size_t rows, size_t cols);
void mat_print(Mat m, char *name);
void mat_sum(Mat a, Mat b);
void mat_fill(Mat m, double x);
void mat_dot(Mat dest, Mat a, Mat b);
double sigmoidf(double x);
void mat_sig(Mat m);
Mat mat_row(Mat m, size_t row);
void mat_copy(Mat a, Mat b);
NN nn_alloc(size_t *arch, size_t count);
void nn_print(NN nn, char *name);
double nn_cost(NN nn, Mat ti, Mat to);
void nn_learn(NN nn, double eps, double rate, Mat ti, Mat to);
void commit_diff(NN nn, double rate);
void my_learn(NN nn, double rate, Mat input, Mat output);
void print_delta(NN nn);
void nn_ileri(NN nn);
