//========================================================================================================================================================================================================200
//======================================================================================================================================================150
//====================================================================================================100
//==================================================50

//========================================================================================================================================================================================================200
//	UPDATE
//========================================================================================================================================================================================================200

//	14 APR 2011 Lukasz G. Szafaryn

//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	LIBRARIES
//======================================================================================================================================================150

#include <stdio.h>					// (in path known to compiler)			needed by printf
#include <stdlib.h>					// (in path known to compiler)			needed by malloc
#include <stdbool.h>				// (in path known to compiler)			needed by true/false
#include <math.h>
#include <string.h>

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

#include "./kernel/kernel_gpu_cuda_wrapper.h"	// (in library path specified here)
#include "../../common/rodinia_verify.h"

enum {
	EXIT_STATUS_SUCCESS = 0,
	EXIT_STATUS_FAILURE = 1
};

static const fp LAVAMD_REFERENCE_ABS_TOLERANCE = 1.0e-8;
static const fp LAVAMD_REFERENCE_REL_TOLERANCE = 1.0e-8;

static int fail_argument(const char* message)
{
	printf("ERROR: %s\n", message);
	return EXIT_STATUS_FAILURE;
}

static void compute_lavamd_reference(
		par_str par_cpu,
		dim_str dim_cpu,
		const box_str* box_cpu,
		const FOUR_VECTOR* rv_cpu,
		const fp* qv_cpu,
		FOUR_VECTOR* fv_reference)
{
	fp a2 = 2.0 * par_cpu.alpha * par_cpu.alpha;

	for(long index=0; index<dim_cpu.space_elem; index=index+1){
		fv_reference[index].v = 0;
		fv_reference[index].x = 0;
		fv_reference[index].y = 0;
		fv_reference[index].z = 0;
	}

	for(int bx=0; bx<dim_cpu.number_boxes; bx=bx+1){
		int first_i = box_cpu[bx].offset;
		for(int neighbor=0; neighbor<(1+box_cpu[bx].nn); neighbor=neighbor+1){
			int pointer = (neighbor==0) ? bx : box_cpu[bx].nei[neighbor-1].number;
			int first_j = box_cpu[pointer].offset;
			for(int i=0; i<NUMBER_PAR_PER_BOX; i=i+1){
				const FOUR_VECTOR rA = rv_cpu[first_i+i];
				FOUR_VECTOR* fA = &fv_reference[first_i+i];
				for(int j=0; j<NUMBER_PAR_PER_BOX; j=j+1){
					const FOUR_VECTOR rB = rv_cpu[first_j+j];
					fp r2 = rA.v + rB.v - DOT(rA, rB);
					fp u2 = a2 * r2;
					fp vij = exp(-u2);
					fp fs = 2 * vij;
					fp dx = rA.x - rB.x;
					fp fxij = fs * dx;
					fp dy = rA.y - rB.y;
					fp fyij = fs * dy;
					fp dz = rA.z - rB.z;
					fp fzij = fs * dz;

					fA->v += qv_cpu[first_j+j] * vij;
					fA->x += qv_cpu[first_j+j] * fxij;
					fA->y += qv_cpu[first_j+j] * fyij;
					fA->z += qv_cpu[first_j+j] * fzij;
				}
			}
		}
	}
}

static int compare_force_component(const char* component, long index, fp actual, fp expected)
{
	fp diff = fabs(actual - expected);
	fp tolerance = LAVAMD_REFERENCE_ABS_TOLERANCE + LAVAMD_REFERENCE_REL_TOLERANCE * fabs(expected);

	if(isfinite(actual) && isfinite(expected) && diff <= tolerance){
		return 0;
	}

	fprintf(
			stderr,
			"LavaMD CPU reference mismatch at particle %ld component %s: actual=%0.12e expected=%0.12e diff=%0.12e tolerance=%0.12e\n",
			index,
			component,
			actual,
			expected,
			diff,
			tolerance);
	return 1;
}

static int verify_lavamd_reference(const FOUR_VECTOR* actual, const FOUR_VECTOR* expected, long count)
{
	long mismatches = 0;

	for(long index=0; index<count; index=index+1){
		mismatches += compare_force_component("v", index, actual[index].v, expected[index].v);
		mismatches += compare_force_component("x", index, actual[index].x, expected[index].x);
		mismatches += compare_force_component("y", index, actual[index].y, expected[index].y);
		mismatches += compare_force_component("z", index, actual[index].z, expected[index].z);
	}

	if(mismatches != 0){
		fprintf(stderr, "LavaMD CPU reference verification failed with %ld mismatched component(s)\n", mismatches);
		rodinia_print_fail("LavaMD CPU reference verification");
		return EXIT_STATUS_FAILURE;
	}

	rodinia_print_pass("LavaMD CPU reference verification");
	return EXIT_STATUS_SUCCESS;
}

//========================================================================================================================================================================================================200
//	MAIN FUNCTION
//========================================================================================================================================================================================================200

int 
main(	int argc, 
		char *argv [])
{

	printf("thread block size of kernel = %d \n", NUMBER_THREADS);
	//======================================================================================================================================================150
	//	CPU/MCPU VARIABLES
	//======================================================================================================================================================150

	// timer
	long long time0;

	time0 = get_time();

	// timer
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;
	long long time7;

	// counters
	int i, j, k, l, m, n;

	// system memory
	par_str par_cpu;
	dim_str dim_cpu;
	box_str* box_cpu;
	FOUR_VECTOR* rv_cpu;
	fp* qv_cpu;
	FOUR_VECTOR* fv_cpu;
	FOUR_VECTOR* fv_reference_cpu;
	int nh;
	int verify_cpu = 0;

	time1 = get_time();

	//======================================================================================================================================================150
	//	CHECK INPUT ARGUMENTS
	//======================================================================================================================================================150

	// assing default values
	dim_cpu.boxes1d_arg = 1;

	// go through arguments
	for(dim_cpu.cur_arg=1; dim_cpu.cur_arg<argc; dim_cpu.cur_arg++){
		// check if -boxes1d
		if(strcmp(argv[dim_cpu.cur_arg], "-boxes1d")==0){
			// check if value provided
			if(argc>dim_cpu.cur_arg+1){
				// check if value is a number
				if(isInteger(argv[dim_cpu.cur_arg+1])==1){
					dim_cpu.boxes1d_arg = atoi(argv[dim_cpu.cur_arg+1]);
					if(dim_cpu.boxes1d_arg<=0){
						return fail_argument("Wrong value to -boxes1d parameter, cannot be <=0");
					}
					dim_cpu.cur_arg = dim_cpu.cur_arg+1;
				}
				// value is not a number
				else{
					return fail_argument("Value to -boxes1d parameter in not a number");
				}
			}
			// value not provided
			else{
				return fail_argument("Missing value to -boxes1d parameter");
			}
		}
		else if(strcmp(argv[dim_cpu.cur_arg], "--verify-cpu")==0){
			verify_cpu = 1;
		}
		// unknown
		else{
			return fail_argument("Unknown parameter");
		}
	}

	// Print configuration
	printf("Configuration used: boxes1d = %d\n", dim_cpu.boxes1d_arg);

	time2 = get_time();

	//======================================================================================================================================================150
	//	INPUTS
	//======================================================================================================================================================150

	par_cpu.alpha = 0.5;

	time3 = get_time();

	//======================================================================================================================================================150
	//	DIMENSIONS
	//======================================================================================================================================================150

	// total number of boxes
	dim_cpu.number_boxes = dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg * dim_cpu.boxes1d_arg;

	// how many particles space has in each direction
	dim_cpu.space_elem = dim_cpu.number_boxes * NUMBER_PAR_PER_BOX;
	dim_cpu.space_mem = dim_cpu.space_elem * sizeof(FOUR_VECTOR);
	dim_cpu.space_mem2 = dim_cpu.space_elem * sizeof(fp);

	// box array
	dim_cpu.box_mem = dim_cpu.number_boxes * sizeof(box_str);

	time4 = get_time();

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
	srand(7);

	// input (distances)
	rv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		rv_cpu[i].v = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].x = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].y = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
		rv_cpu[i].z = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
	}

	// input (charge)
	qv_cpu = (fp*)malloc(dim_cpu.space_mem2);
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		qv_cpu[i] = (rand()%10 + 1) / 10.0;			// get a number in the range 0.1 - 1.0
	}

	// output (forces)
	fv_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
	for(i=0; i<dim_cpu.space_elem; i=i+1){
		fv_cpu[i].v = 0;								// set to 0, because kernels keeps adding to initial value
		fv_cpu[i].x = 0;								// set to 0, because kernels keeps adding to initial value
		fv_cpu[i].y = 0;								// set to 0, because kernels keeps adding to initial value
		fv_cpu[i].z = 0;								// set to 0, because kernels keeps adding to initial value
	}

	fv_reference_cpu = NULL;
	if(verify_cpu){
		fv_reference_cpu = (FOUR_VECTOR*)malloc(dim_cpu.space_mem);
		if(fv_reference_cpu == NULL){
			fprintf(stderr, "ERROR: Could not allocate LavaMD CPU reference buffer\n");
			free(rv_cpu);
			free(qv_cpu);
			free(fv_cpu);
			free(box_cpu);
			return EXIT_STATUS_FAILURE;
		}
	}

	time5 = get_time();

	//======================================================================================================================================================150
	//	KERNEL
	//======================================================================================================================================================150

	//====================================================================================================100
	//	GPU_CUDA
	//====================================================================================================100

	kernel_gpu_cuda_wrapper(par_cpu,
							dim_cpu,
							box_cpu,
							rv_cpu,
							qv_cpu,
							fv_cpu);

	time6 = get_time();

	int status = EXIT_STATUS_SUCCESS;
	if(verify_cpu){
		compute_lavamd_reference(par_cpu, dim_cpu, box_cpu, rv_cpu, qv_cpu, fv_reference_cpu);
		status = verify_lavamd_reference(fv_cpu, fv_reference_cpu, dim_cpu.space_elem);
	}

	//======================================================================================================================================================150
	//	SYSTEM MEMORY DEALLOCATION
	//======================================================================================================================================================150

	// dump results
#ifdef OUTPUT
        FILE *fptr;
	fptr = fopen("result.txt", "w");	
	if(fptr == NULL){
		fprintf(stderr, "ERROR: Could not open result.txt for writing\n");
		status = EXIT_STATUS_FAILURE;
	}
	else{
		for(i=0; i<dim_cpu.space_elem; i=i+1){
			fprintf(fptr, "%f, %f, %f, %f\n", fv_cpu[i].v, fv_cpu[i].x, fv_cpu[i].y, fv_cpu[i].z);
		}
		fclose(fptr);
	}
#endif       	



	free(rv_cpu);
	free(qv_cpu);
	free(fv_cpu);
	free(fv_reference_cpu);
	free(box_cpu);

	time7 = get_time();

	//======================================================================================================================================================150
	//	DISPLAY TIMING
	//======================================================================================================================================================150

	// printf("Time spent in different stages of the application:\n");

	// printf("%15.12f s, %15.12f % : VARIABLES\n",						(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time7-time0) * 100);
	// printf("%15.12f s, %15.12f % : INPUT ARGUMENTS\n", 					(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time7-time0) * 100);
	// printf("%15.12f s, %15.12f % : INPUTS\n",							(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time7-time0) * 100);
	// printf("%15.12f s, %15.12f % : dim_cpu\n", 							(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time7-time0) * 100);
	// printf("%15.12f s, %15.12f % : SYS MEM: ALO\n",						(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time7-time0) * 100);

	// printf("%15.12f s, %15.12f % : KERNEL: COMPUTE\n",					(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time7-time0) * 100);

	// printf("%15.12f s, %15.12f % : SYS MEM: FRE\n", 					(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time7-time0) * 100);

	// printf("Total time:\n");
	// printf("%.12f s\n", 												(float) (time7-time0) / 1000000);

	//======================================================================================================================================================150
	//	RETURN
	//======================================================================================================================================================150

	return status;

}
