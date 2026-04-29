#ifdef __cplusplus
extern "C" {
#endif

//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	INFORMATION
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	UPDATE
//======================================================================================================================================================150

//	2009.12 Lukasz G. Szafaryn
//		-- entire code written

//======================================================================================================================================================150
//	DESCRIPTION
//======================================================================================================================================================150

// Description

//======================================================================================================================================================150
//	USE
//======================================================================================================================================================150

// How to run

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>					// (in path known to compiler)			needed by printf
#include <stdlib.h>					// (in path known to compiler)			needed by malloc
#include <stdbool.h>				// (in path known to compiler)			needed by true/false
#include <string.h>
#include <time.h>
#include <math.h>

//======================================================================================================================================================150
//	UTILITIES
//======================================================================================================================================================150

#include "./util/timer/timer.h"			// (in path specified here)
#include "./util/num/num.h"				// (in path specified here)

//======================================================================================================================================================150
//	MAIN FUNCTION HEADER
//======================================================================================================================================================150

#include "./main.h"						// (in the current directory)

//======================================================================================================================================================150
//	KERNEL
//======================================================================================================================================================150

#include "./kernel/kernel_gpu_opencl_wrapper.h"	// (in library path specified here)

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

int platform_id_inuse = 0;            // platform id in use (default: 0)
int device_id_inuse = 0;              // device id in use (default : 0)

#define LAVAMD_ABS_TOL 1e-3f
#define LAVAMD_REL_TOL 1e-3f

typedef struct {
	int boxes1d;
	unsigned int seed;
	const char *output_file;
	const char *ref_file;
	const char *save_ref_file;
} run_options;

static int parse_uint_arg(const char *value, unsigned int *out_value) {
	char *end = NULL;
	unsigned long parsed = strtoul(value, &end, 10);
	if (value[0] == '\0' || end == NULL || *end != '\0') {
		return 0;
	}
	*out_value = (unsigned int)parsed;
	return 1;
}

static int almost_equal(fp actual, fp expected) {
	fp diff = fabsf(actual - expected);
	fp scale = fmaxf(fabsf(actual), fabsf(expected));
	return diff <= LAVAMD_ABS_TOL || diff <= LAVAMD_REL_TOL * scale;
}

static int write_results_file(const char *path, int boxes1d, unsigned int seed,
		long space_elem, FOUR_VECTOR *fv_cpu) {
	long i;
	FILE *fptr = fopen(path, "w");
	if (fptr == NULL) {
		fprintf(stderr, "ERROR: Failed to open %s for writing\n", path);
		return 1;
	}
	fprintf(fptr, "# boxes1d=%d seed=%u count=%ld\n", boxes1d, seed, space_elem);
	for (i = 0; i < space_elem; i++) {
		fprintf(fptr, "%.9g, %.9g, %.9g, %.9g\n",
			fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
	}
	fclose(fptr);
	return 0;
}

static int verify_reference(const char *path, int boxes1d, unsigned int seed,
		long space_elem, FOUR_VECTOR *fv_cpu) {
	long i;
	int ref_boxes1d = 0;
	unsigned int ref_seed = 0;
	long ref_count = 0;
	FILE *fptr = fopen(path, "r");
	if (fptr == NULL) {
		fprintf(stderr, "ERROR: Failed to open %s for reading\n", path);
		return 1;
	}
	if (fscanf(fptr, "# boxes1d=%d seed=%u count=%ld\n", &ref_boxes1d, &ref_seed, &ref_count) != 3) {
		fprintf(stderr, "ERROR: Invalid reference header in %s\n", path);
		fclose(fptr);
		return 1;
	}
	if (ref_boxes1d != boxes1d || ref_seed != seed || ref_count != space_elem) {
		fprintf(stderr,
			"ERROR: Reference metadata mismatch in %s (boxes1d=%d seed=%u count=%ld, expected %d %u %ld)\n",
			path, ref_boxes1d, ref_seed, ref_count, boxes1d, seed, space_elem);
		fclose(fptr);
		return 1;
	}
	for (i = 0; i < space_elem; i++) {
		fp ref_v = 0.0f, ref_x = 0.0f, ref_y = 0.0f, ref_z = 0.0f;
		if (fscanf(fptr, "%f, %f, %f, %f\n", &ref_v, &ref_x, &ref_y, &ref_z) != 4) {
			fprintf(stderr, "ERROR: Failed to parse reference %s at element %ld\n", path, i);
			fclose(fptr);
			return 1;
		}
		if (!almost_equal(fv_cpu[i].v, ref_v) ||
			!almost_equal(fv_cpu[i].x, ref_x) ||
			!almost_equal(fv_cpu[i].y, ref_y) ||
			!almost_equal(fv_cpu[i].z, ref_z)) {
			fprintf(stderr,
				"ERROR: Reference mismatch at element %ld\n"
				"got=(%.8f, %.8f, %.8f, %.8f) ref=(%.8f, %.8f, %.8f, %.8f)\n",
				i, fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z,
				ref_v, ref_x, ref_y, ref_z);
			fclose(fptr);
			return 1;
		}
	}
	fclose(fptr);
	printf("Reference check passed: %s\n", path);
	return 0;
}

int 
main(	int argc, 
		char *argv [])
{

	//======================================================================================================================================================150
	//	CPU/MCPU VARIABLES
	//======================================================================================================================================================150

	// counters
	int i, j, k, l, m, n;

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR* rv_cpu;
	fp* qv_cpu;
	FOUR_VECTOR* fv_cpu;
	int nh;
	run_options options;


	printf("WG size of kernel = %d \n", NUMBER_THREADS);

	//======================================================================================================================================================150
	//	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================150

	// assing default values
	dim_cpu.arch_arg = 0;
	dim_cpu.cores_arg = 1;
	dim_cpu.boxes1d_arg = 1;
	options.boxes1d = 1;
	options.seed = 7;
	options.output_file = NULL;
	options.ref_file = NULL;
	options.save_ref_file = NULL;

	// go through arguments
	if (argc >= 3) {
		for(dim_cpu.cur_arg=1; dim_cpu.cur_arg<argc; dim_cpu.cur_arg++){
			// check if -boxes1d
			if(strcmp(argv[dim_cpu.cur_arg], "-boxes1d")==0){
				// check if value provided
				if(argc>=dim_cpu.cur_arg+1){
					// check if value is a number
					if(isInteger(argv[dim_cpu.cur_arg+1])==1){
						dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg+1]);
						options.boxes1d = dim_cpu.boxes1d_arg;
						if(dim_cpu.boxes1d_arg<0){
							printf("ERROR: Wrong value to -boxes1d argument, cannot be <=0\n");
							return 0;
						}
						dim_cpu.cur_arg = dim_cpu.cur_arg+1;
					}
					// value is not a number
					else{
						printf("ERROR: Value to -boxes1d argument in not a number\n");
						return 0;
					}
				}
				// value not provided
				else{
					printf("ERROR: Missing value to -boxes1d argument\n");
					return 0;
				}
			}
            else if(strcmp(argv[dim_cpu.cur_arg], "-p")==0){
				if(argc>=dim_cpu.cur_arg+1){
				    platform_id_inuse = atoi(argv[dim_cpu.cur_arg+1]);
					dim_cpu.cur_arg = dim_cpu.cur_arg+1;
				}
			}
            else if(strcmp(argv[dim_cpu.cur_arg], "-d")==0){
				if(argc>=dim_cpu.cur_arg+1){
				    device_id_inuse = atoi(argv[dim_cpu.cur_arg+1]);
					dim_cpu.cur_arg = dim_cpu.cur_arg+1;
				}
			}
			else if(strcmp(argv[dim_cpu.cur_arg], "--seed")==0){
				if(argc>=dim_cpu.cur_arg+1 &&
						parse_uint_arg(argv[dim_cpu.cur_arg+1], &options.seed)){
					dim_cpu.cur_arg = dim_cpu.cur_arg+1;
				}
				else{
					printf("ERROR: Value to --seed argument is not a valid unsigned integer\n");
					return 1;
				}
			}
			else if(strcmp(argv[dim_cpu.cur_arg], "--output")==0){
				if(argc>=dim_cpu.cur_arg+1){
					options.output_file = argv[dim_cpu.cur_arg+1];
					dim_cpu.cur_arg = dim_cpu.cur_arg+1;
				}
				else{
					printf("ERROR: Missing value to --output argument\n");
					return 1;
				}
			}
			else if(strcmp(argv[dim_cpu.cur_arg], "--ref")==0){
				if(argc>=dim_cpu.cur_arg+1){
					options.ref_file = argv[dim_cpu.cur_arg+1];
					dim_cpu.cur_arg = dim_cpu.cur_arg+1;
				}
				else{
					printf("ERROR: Missing value to --ref argument\n");
					return 1;
				}
			}
			else if(strcmp(argv[dim_cpu.cur_arg], "--save-ref")==0){
				if(argc>=dim_cpu.cur_arg+1){
					options.save_ref_file = argv[dim_cpu.cur_arg+1];
					dim_cpu.cur_arg = dim_cpu.cur_arg+1;
				}
				else{
					printf("ERROR: Missing value to --save-ref argument\n");
					return 1;
				}
			}
			// unknown
			else{
				printf("ERROR: Unknown argument\n");
				return 1;
			}
		}
		// Print configuration
		printf("Configuration used: arch = %d, cores = %d, boxes1d = %d\n", dim_cpu.arch_arg, dim_cpu.cores_arg, dim_cpu.boxes1d_arg);
	}
	else{
		printf("Provide boxes1d argument, example: -boxes1d 16 [-p platform_id] [-d device_id]");
		return 0;
	}

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150

	par_cpu.alpha = 0.5;

	//======================================================================================================================================================150
	//	DIMENSIONS
	//======================================================================================================================================================150

	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg; // 8*8*8=512

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;							//512*100=51,200
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	//======================================================================================================================================================150
	//	SYSTEM MEMORY
	//======================================================================================================================================================150

	//====================================================================================================100
	//	BOX
	//====================================================================================================100

	// allocate boxes
	box_cpu = (box_str*)malloc(dim_cpu.box_mem);

	// initialize number of home boxes
	nh = 0;

	// home boxes in z direction
	for(i=0; i<dim_cpu.boxes1d_arg; i++){
		// home boxes in y direction
		for(j=0; j<dim_cpu.boxes1d_arg; j++){
			// home boxes in x direction
			for(k=0; k<dim_cpu.boxes1d_arg; k++){

				// current home box
				box_cpu[nh].x = k;
				box_cpu[nh].y = j;
				box_cpu[nh].z = i;
				box_cpu[nh].number = nh;
				box_cpu[nh].offset = nh * NUMBER_PAR_PER_BOX;

				// initialize number of neighbor boxes
				box_cpu[nh].nn = 0;

				// neighbor boxes in z direction
				for(l=-1; l<2; l++){
					// neighbor boxes in y direction
					for(m=-1; m<2; m++){
						// neighbor boxes in x direction
						for(n=-1; n<2; n++){

							// check if (this neighbor exists) and (it is not the same as home box)
							if(		(((i+l)>=0 && (j+m)>=0 && (k+n)>=0)==true && ((i+l)<dim_cpu.boxes1d_arg && (j+m)<dim_cpu.boxes1d_arg && (k+n)<dim_cpu.boxes1d_arg)==true)	&&
									(l==0 && m==0 && n==0)==false	){

								// current neighbor box
								box_cpu[nh].nei[box_cpu[nh].nn].x = (k+n);
								box_cpu[nh].nei[box_cpu[nh].nn].y = (j+m);
								box_cpu[nh].nei[box_cpu[nh].nn].z = (i+l);
								box_cpu[nh].nei[box_cpu[nh].nn].number =	(box_cpu[nh].nei[box_cpu[nh].nn].z * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg) + 
																			(box_cpu[nh].nei[box_cpu[nh].nn].y * dim_cpu.boxes1d_arg) + 
																			 box_cpu[nh].nei[box_cpu[nh].nn].x;
								box_cpu[nh].nei[box_cpu[nh].nn].offset = box_cpu[nh].nei[box_cpu[nh].nn].number * NUMBER_PAR_PER_BOX;

								// increment neighbor box
								box_cpu[nh].nn = box_cpu[nh].nn + 1;

							}

						} // neighbor boxes in x direction
					} // neighbor boxes in y direction
				} // neighbor boxes in z direction

				// increment home box
				nh = nh + 1;

			} // home boxes in x direction
		} // home boxes in y direction
	} // home boxes in z direction

	//====================================================================================================100
	//	PARAMETERS, DISTANCE, CHARGE AND FORCE
	//====================================================================================================100

	// random generator seed set to random value - time in this case
	srand(options.seed);

	// input (distances)
	rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		rv_cpu[i].v = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		// rv_cpu[i].v = 0.1;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].x = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		// rv_cpu[i].x = 0.2;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].y = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		// rv_cpu[i].y = 0.3;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].z = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		// rv_cpu[i].z = 0.4;			// get a number in the range 0.1 - 1.0
	}

	// input (charge)
	qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		qv_cpu[i] = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		// qv_cpu[i] = 0.5;			// get a number in the range 0.1 - 1.0
	}

	// output (forces)
	fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		fv_cpu[i].v = 0;								// set to 0, because kernels keeps adding to initial value
		fv_cpu[i].x = 0;								// set to 0, because kernels keeps adding to initial value
		fv_cpu[i].y = 0;								// set to 0, because kernels keeps adding to initial value
		fv_cpu[i].z = 0;								// set to 0, because kernels keeps adding to initial value
	}

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU_OPENCL
	//====================================================================================================100

	kernel_gpu_opencl_wrapper(	par_cpu,
								dim_cpu,
								box_cpu,
								rv_cpu,
								qv_cpu,
								fv_cpu);

	//======================================================================================================================================================150
	//	SYSTEM MEMORY DEALLOCATION
	//======================================================================================================================================================150

	if (options.output_file != NULL &&
			write_results_file(options.output_file, dim_cpu.boxes1d_arg, options.seed,
				dim_cpu.space_elem, fv_cpu) != 0) {
		return 1;
	}
	if (options.save_ref_file != NULL &&
			write_results_file(options.save_ref_file, dim_cpu.boxes1d_arg, options.seed,
				dim_cpu.space_elem, fv_cpu) != 0) {
		return 1;
	}
	if (options.save_ref_file != NULL) {
		printf("Reference saved to %s\n", options.save_ref_file);
	}
	if (options.ref_file != NULL &&
			verify_reference(options.ref_file, dim_cpu.boxes1d_arg, options.seed,
				dim_cpu.space_elem, fv_cpu) != 0) {
		return 1;
	}


	free(rv_cpu);
	free(qv_cpu);
	free(fv_cpu);
	free(box_cpu);

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	// printf("Time spent in different stages of the application:\n");

	//======================================================================================================================================================150
	//	RETURN
	//======================================================================================================================================================150

	return 0.0;																					// always returns 0.0

}

//========================================================================================================================================================================================================200
//	END
//========================================================================================================================================================================================================200

#ifdef __cplusplus
}
#endif
