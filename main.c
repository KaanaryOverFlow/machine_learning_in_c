#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <fcntl.h>

#include "utils.h"
#include "nn.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define GYM_IMPLEMENTATION
#include "gym.h"

struct {
	int fd;	
} *ed;

void setup() {
	DEBUG;
}


void img2mat(char *name, Mat inputs, Mat outputs) {

	int image_width, image_height, image_comp;
	unsigned char *data = stbi_load(name, &image_width, &image_height, &image_comp, 0 );
	if (data == NULL) die("parse failed");
	
	size_t check = image_width * image_height;
	if (inputs.rows != check || outputs.rows != check) {
		note("The row should be %ld * %ld = %ld", image_width, image_height, check);
		die("invalid matrix to fill");
	}
	
	for (size_t line = 0; line < image_height; line++) {
		for (size_t height = 0; height < image_width; height++) {
			
			size_t index = line * image_width +  height;
			if (index > check) die("wrong index");
			
			double y = (double)line;
			double x = (double)height;
			double ny = y / (image_width - 1);
			double nx = x / (image_height - 1);
			FOR(i, image_comp) {
				double nval = ((double) data[index + i]); // / 255.f;
				nval /= 255;
				MAT_AT(outputs, index, i) = nval ;
			}



			MAT_AT(inputs, index, 0) = ny;
			MAT_AT(inputs, index, 1) = nx;

			
		}
	}
}

void nn2img(NN nn, char *name, double w, double h, size_t byte) {

	



	
	unsigned char *output_data = malloc(h * w * byte);


	for (size_t y = 0; y < h; y++) { 
		for (size_t x = 0; x < w; x++) { 
			
			size_t index = x * w +  y;
			
			double normal_height = (double) y / (h - 1);
			double normal_width = (double) x / (w - 1);

			double data[] =  {normal_height, normal_width };

			Mat input = {.rows = 1, .cols = 2, .data = data};

			nn_forward(nn, input);

			output_data[index] = (unsigned char) ( 255.f *  MAT_AT(nn_out(nn), 0, 0));
			
			printf("%3u ", output_data[index]);
		}
		printf("\n");

	}
}

void from_lib(void);

void default_app(void) {
	in();
	
	Mat inputs = mat_alloc(28*28, 2); 
	Mat outputs = mat_alloc(28*28, 1); 

	img2mat("../src/10264.png", inputs, outputs);

	
	mat_shuffle(inputs, outputs);

	Dense network[] = {
#if 0
		create_dense(2, 8, RELU),
		create_dense(8, 8, RELU),
		create_dense(8, 1, RELU),
#else	
		create_dense(2, 11, SIGMOID),
		create_dense(11, 11, SIGMOID),
		create_dense(11, 11, SIGMOID),
		create_dense(11, 1, SIGMOID),
#endif
	};

	NN nn = {
		.count = ARRAY_LEN(network),
		.dense = network,
		.grad = mat_alloc(1,1),
	};
	
	plf(nn_cost(nn, inputs, outputs));
     	


	Gym_Plot plot = {0};
	plot.count = 0;
	
	size_t WINDOW_FACTOR = 80;
	size_t WINDOW_WIDTH = (16*WINDOW_FACTOR);
	size_t WINDOW_HEIGHT = (9*WINDOW_FACTOR);
	
	SetConfigFlags(FLAG_WINDOW_RESIZABLE);
	InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "xor");
	SetTargetFPS(60);


	int paused = 0;
	while (!WindowShouldClose()) {
		if (IsKeyPressed(KEY_SPACE)) {
			paused = !paused;
			note("paused");
		}
		if (IsKeyPressed(KEY_S)) {
			nn_save(nn, "dense.network");
		}
		if (IsKeyPressed(KEY_R)) {
			mat_shuffle(inputs, outputs);
			nn_rand(nn);
			plot.count = 0;
			paused = 0;
		}
		if (IsKeyPressed(KEY_P)) {
			nn2img(nn, NULL, 28, 28, 1);
		}
		if (IsKeyPressed(KEY_Q)) die("Q");
     		if (!paused) {
			nn_fit(nn, 50, 0.7, 28, inputs, outputs);
			da_append(&plot, nn_cost(nn, inputs, outputs));
		}
		BeginDrawing();

		ClearBackground(GYM_BACKGROUND);
	        {
	            int w = GetRenderWidth();
	            int h = GetRenderHeight();
	
	            Gym_Rect r;
	            r.w = w;
	            r.h = h*2/3;
	            r.x = 0;
	            r.y = h/2 - r.h/2;
	
	            gym_layout_begin(GLO_HORZ, r, 3, 10);
	                gym_plot(plot, gym_layout_slot(), RED);
			gym_render_nn(nn, gym_layout_slot());
	            gym_layout_end();
	
	        }
	        EndDrawing();
	}

	nn2img(nn, NULL, 28, 28, 1);
	
	out();
}
void sec_main(char *param) {
	if (!param) {
		default_app();
	} else {
		die("invalid parameter");
	}

}

#ifdef SHARED_LIBRARY
void __attribute__ ((constructor)) _setup(void) {
	in();
	setup();
	default_app();
	out();
}

#else

int main(int argc, char *argv[]) {
	in();
	setup();
	sec_main(argv[1]);
	out();
	return 0;
}
#endif
