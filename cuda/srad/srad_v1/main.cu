//====================================================================================================100
//		UPDATE
//====================================================================================================100

//    2006.03   Rob Janiczek
//        --creation of prototype version
//    2006.03   Drew Gilliam
//        --rewriting of prototype version into current version
//        --got rid of multiple function calls, all code in a  
//         single function (for speed)
//        --code cleanup & commenting
//        --code optimization efforts   
//    2006.04   Drew Gilliam
//        --added diffusion coefficent saturation on [0,1]
//		2009.12 Lukasz G. Szafaryn
//		-- reading from image, command line inputs
//		2010.01 Lukasz G. Szafaryn
//		--comments

//====================================================================================================100
//	DEFINE / INCLUDE
//====================================================================================================100

#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>

#include "define.c"
#include "extract_kernel.cu"
#include "prepare_kernel.cu"
#include "reduce_kernel.cu"
#include "srad_kernel.cu"
#include "srad2_kernel.cu"
#include "compress_kernel.cu"
#include "graphics.c"
#include "resize.c"
#include "timer.c"
#include "../../../common/rodinia_verify.h"

#include "device.c"				// (in library path specified to compiler)	needed by for device functions

#define SRAD_ABS_TOLERANCE 0.05f
#define SRAD_REL_TOLERANCE 0.001f

static void usage(const char *program)
{
	fprintf(stderr, "Usage: %s <iterations> <lambda> <rows> <cols> [--verify-cpu]\n", program);
}

static void cpu_extract(long elements, fp *image)
{
	for (long i = 0; i < elements; i++) {
		image[i] = expf(image[i] / 255.0f);
	}
}

static void cpu_compress(long elements, fp *image)
{
	for (long i = 0; i < elements; i++) {
		image[i] = logf(image[i]) * 255.0f;
	}
}

static void cpu_prepare(long elements, fp *image, fp *sums, fp *sums2)
{
	for (long i = 0; i < elements; i++) {
		sums[i] = image[i];
		sums2[i] = image[i] * image[i];
	}
}

static void cpu_reduce_block(
	int block,
	int block_count,
	int element_count,
	int multiplier,
	fp *sums,
	fp *sums2)
{
	fp psum[NUMBER_THREADS];
	fp psum2[NUMBER_THREADS];
	int active = NUMBER_THREADS - (block_count * NUMBER_THREADS - element_count);
	if (active == NUMBER_THREADS || block != block_count - 1) {
		active = NUMBER_THREADS;
	}

	for (int thread = 0; thread < active; thread++) {
		int input_index = (block * NUMBER_THREADS + thread) * multiplier;
		psum[thread] = sums[input_index];
		psum2[thread] = sums2[input_index];
	}

	if (active == 1) {
		sums[block * multiplier * NUMBER_THREADS] = psum[0];
		sums2[block * multiplier * NUMBER_THREADS] = psum2[0];
		return;
	}

	int reduced = 0;
	for (int step = 2; step <= active && step <= NUMBER_THREADS; step *= 2) {
		reduced = step;
		for (int thread = step - 1; thread < active; thread += step) {
			psum[thread] = psum[thread] + psum[thread - step / 2];
			psum2[thread] = psum2[thread] + psum2[thread - step / 2];
		}
	}

	if (active == NUMBER_THREADS) {
		sums[block * multiplier * NUMBER_THREADS] = psum[NUMBER_THREADS - 1];
		sums2[block * multiplier * NUMBER_THREADS] = psum2[NUMBER_THREADS - 1];
		return;
	}

	for (int i = block * NUMBER_THREADS + reduced; i < block * NUMBER_THREADS + active; i++) {
		psum[reduced - 1] = psum[reduced - 1] + sums[i];
		psum2[reduced - 1] = psum2[reduced - 1] + sums2[i];
	}
	sums[block * multiplier * NUMBER_THREADS] = psum[reduced - 1];
	sums2[block * multiplier * NUMBER_THREADS] = psum2[reduced - 1];
}

static void cpu_reduce_all(long elements, int block_count, fp *sums, fp *sums2)
{
	int blocks = block_count;
	int no = (int)elements;
	int multiplier = 1;
	while (blocks != 0) {
		for (int block = 0; block < blocks; block++) {
			cpu_reduce_block(block, blocks, no, multiplier, sums, sums2);
		}
		no = blocks;
		if (blocks == 1) {
			blocks = 0;
		}
		else {
			multiplier = multiplier * NUMBER_THREADS;
			blocks = blocks / NUMBER_THREADS + (blocks % NUMBER_THREADS != 0);
		}
	}
}

static void cpu_srad(
	fp lambda,
	int rows,
	int cols,
	long elements,
	int *iN,
	int *iS,
	int *jE,
	int *jW,
	fp *dN,
	fp *dS,
	fp *dW,
	fp *dE,
	fp q0sqr,
	fp *c,
	fp *image)
{
	for (long ei = 0; ei < elements; ei++) {
		int row = (ei + 1) % rows - 1;
		int col = (ei + 1) / rows + 1 - 1;
		if ((ei + 1) % rows == 0) {
			row = rows - 1;
			col = col - 1;
		}

		fp Jc = image[ei];
		fp dN_loc = image[iN[row] + rows * col] - Jc;
		fp dS_loc = image[iS[row] + rows * col] - Jc;
		fp dW_loc = image[row + rows * jW[col]] - Jc;
		fp dE_loc = image[row + rows * jE[col]] - Jc;
		fp G2 = (dN_loc * dN_loc + dS_loc * dS_loc + dW_loc * dW_loc + dE_loc * dE_loc) / (Jc * Jc);
		fp L = (dN_loc + dS_loc + dW_loc + dE_loc) / Jc;
		fp num = (0.5 * G2) - ((1.0 / 16.0) * (L * L));
		fp den = 1 + (0.25 * L);
		fp qsqr = num / (den * den);

		den = (qsqr - q0sqr) / (q0sqr * (1 + q0sqr));
		fp c_loc = 1.0 / (1.0 + den);
		if (c_loc < 0) {
			c_loc = 0;
		}
		else if (c_loc > 1) {
			c_loc = 1;
		}

		dN[ei] = dN_loc;
		dS[ei] = dS_loc;
		dW[ei] = dW_loc;
		dE[ei] = dE_loc;
		c[ei] = c_loc;
	}
}

static void cpu_srad2(
	fp lambda,
	int rows,
	long elements,
	int *iS,
	int *jE,
	fp *dN,
	fp *dS,
	fp *dW,
	fp *dE,
	fp *c,
	fp *image)
{
	for (long ei = 0; ei < elements; ei++) {
		int row = (ei + 1) % rows - 1;
		int col = (ei + 1) / rows + 1 - 1;
		if ((ei + 1) % rows == 0) {
			row = rows - 1;
			col = col - 1;
		}

		fp cN = c[ei];
		fp cS = c[iS[row] + rows * col];
		fp cW = c[ei];
		fp cE = c[row + rows * jE[col]];
		fp D = cN * dN[ei] + cS * dS[ei] + cW * dW[ei] + cE * dE[ei];
		image[ei] = image[ei] + 0.25 * lambda * D;
	}
}

static int verify_cpu_reference(
	fp *actual,
	fp *image_ori,
	int image_ori_rows,
	int image_ori_cols,
	int rows,
	int cols,
	int niter,
	fp lambda,
	int *iN,
	int *iS,
	int *jE,
	int *jW,
	int block_count)
{
	long elements = rows * cols;
	fp *expected = (fp *)malloc(sizeof(fp) * elements);
	fp *sums = (fp *)malloc(sizeof(fp) * elements);
	fp *sums2 = (fp *)malloc(sizeof(fp) * elements);
	fp *dN = (fp *)malloc(sizeof(fp) * elements);
	fp *dS = (fp *)malloc(sizeof(fp) * elements);
	fp *dW = (fp *)malloc(sizeof(fp) * elements);
	fp *dE = (fp *)malloc(sizeof(fp) * elements);
	fp *c = (fp *)malloc(sizeof(fp) * elements);
	if (expected == NULL || sums == NULL || sums2 == NULL ||
		dN == NULL || dS == NULL || dW == NULL || dE == NULL || c == NULL) {
		fprintf(stderr, "Cannot allocate SRAD CPU reference buffers\n");
		free(expected);
		free(sums);
		free(sums2);
		free(dN);
		free(dS);
		free(dW);
		free(dE);
		free(c);
		return -1;
	}

	resize(image_ori, image_ori_rows, image_ori_cols, expected, rows, cols, 1);
	cpu_extract(elements, expected);

	for (int iter = 0; iter < niter; iter++) {
		cpu_prepare(elements, expected, sums, sums2);
		cpu_reduce_all(elements, block_count, sums, sums2);
		fp meanROI = sums[0] / fp(elements);
		fp meanROI2 = meanROI * meanROI;
		fp varROI = (sums2[0] / fp(elements)) - meanROI2;
		fp q0sqr = varROI / meanROI2;
		cpu_srad(lambda, rows, cols, elements, iN, iS, jE, jW, dN, dS, dW, dE, q0sqr, c, expected);
		cpu_srad2(lambda, rows, elements, iS, jE, dN, dS, dW, dE, c, expected);
	}

	cpu_compress(elements, expected);

	for (long i = 0; i < elements; i++) {
		fp diff = fabsf(actual[i] - expected[i]);
		fp tolerance = SRAD_ABS_TOLERANCE + SRAD_REL_TOLERANCE * fabsf(expected[i]);
		if (!isfinite(actual[i]) || !isfinite(expected[i]) || diff > tolerance) {
			fprintf(stderr,
				"SRAD CPU reference mismatch at index %ld: actual=%g expected=%g diff=%g tolerance=%g\n",
				i,
				actual[i],
				expected[i],
				diff,
				tolerance);
			free(expected);
			free(sums);
			free(sums2);
			free(dN);
			free(dS);
			free(dW);
			free(dE);
			free(c);
			return -1;
		}
	}

	free(expected);
	free(sums);
	free(sums2);
	free(dN);
	free(dS);
	free(dW);
	free(dE);
	free(c);
	return rodinia_print_pass("SRAD CPU reference verification");
}

//====================================================================================================100
//	MAIN FUNCTION
//====================================================================================================100

int main(int argc, char *argv []){

	//================================================================================80
	// 	VARIABLES
	//================================================================================80

	// time
	long long time0;
	long long time1;
	long long time2;
	long long time3;
	long long time4;
	long long time5;
	long long time6;
	long long time7;
	long long time8;
	long long time9;
	long long time10;
	long long time11;
	long long time12;

	time0 = get_time();

    // inputs image, input paramenters
    fp* image_ori;																// originalinput image
	int image_ori_rows;
	int image_ori_cols;
	long image_ori_elem;

    // inputs image, input paramenters
    fp* image;															// input image
    int Nr,Nc;													// IMAGE nbr of rows/cols/elements
	long Ne;

	// algorithm parameters
    int niter;																// nbr of iterations
    fp lambda;															// update step size
	int verify_cpu;
	int verify_status;

    // size of IMAGE
	int r1,r2,c1,c2;												// row/col coordinates of uniform ROI
	long NeROI;														// ROI nbr of elements

    // surrounding pixel indicies
    int *iN,*iS,*jE,*jW;    

    // counters
    int iter;   // primary loop
    long i,j;    // image row/col

	// memory sizes
	int mem_size_i;
	int mem_size_j;
	int mem_size_single;

	//================================================================================80
	// 	GPU VARIABLES
	//================================================================================80

	// CUDA kernel execution parameters
	dim3 threads;
	int blocks_x;
	dim3 blocks;
	dim3 blocks2;
	dim3 blocks3;

	// memory sizes
	int mem_size;															// matrix memory size

	// HOST
	int no;
	int mul;
	fp total;
	fp total2;
	fp meanROI;
	fp meanROI2;
	fp varROI;
	fp q0sqr;

	// DEVICE
	fp* d_sums;															// partial sum
	fp* d_sums2;
	int* d_iN;
	int* d_iS;
	int* d_jE;
	int* d_jW;
	fp* d_dN; 
	fp* d_dS; 
	fp* d_dW; 
	fp* d_dE;
	fp* d_I;																// input IMAGE on DEVICE
	fp* d_c;

	time1 = get_time();

	//================================================================================80
	// 	GET INPUT PARAMETERS
	//================================================================================80

	if(argc != 5 && argc != 6){
		fprintf(stderr, "ERROR: wrong number of arguments\n");
		usage(argv[0]);
		return EXIT_FAILURE;
	}
	else{
		niter = atoi(argv[1]);
		lambda = atof(argv[2]);
		Nr = atoi(argv[3]);						// it is 502 in the original image
		Nc = atoi(argv[4]);						// it is 458 in the original image
		verify_cpu = 0;
		verify_status = 0;
		if (argc == 6) {
			if (strcmp(argv[5], "--verify-cpu") != 0) {
				fprintf(stderr, "ERROR: unknown option: %s\n", argv[5]);
				usage(argv[0]);
				return EXIT_FAILURE;
			}
			verify_cpu = 1;
		}
	}

	time2 = get_time();

	//================================================================================80
	// 	READ IMAGE (SIZE OF IMAGE HAS TO BE KNOWN)
	//================================================================================80

    // read image
	image_ori_rows = 502;
	image_ori_cols = 458;
	image_ori_elem = image_ori_rows * image_ori_cols;

	image_ori = (fp*)malloc(sizeof(fp) * image_ori_elem);

	if (read_graphics(	"../../../data/srad/image.pgm",
								image_ori,
								image_ori_rows,
								image_ori_cols,
								1) != 0) {
		free(image_ori);
		return EXIT_FAILURE;
	}

	time3 = get_time();

	//================================================================================80
	// 	RESIZE IMAGE (ASSUMING COLUMN MAJOR STORAGE OF image_orig)
	//================================================================================80

	Ne = Nr*Nc;

	image = (fp*)malloc(sizeof(fp) * Ne);

	resize(	image_ori,
				image_ori_rows,
				image_ori_cols,
				image,
				Nr,
				Nc,
				1);

	time4 = get_time();

	//================================================================================80
	// 	SETUP
	//================================================================================80

    r1     = 0;											// top row index of ROI
    r2     = Nr - 1;									// bottom row index of ROI
    c1     = 0;											// left column index of ROI
    c2     = Nc - 1;									// right column index of ROI

	// ROI image size
	NeROI = (r2-r1+1)*(c2-c1+1);											// number of elements in ROI, ROI size

	// allocate variables for surrounding pixels
	mem_size_i = sizeof(int) * Nr;											//
	iN = (int *)malloc(mem_size_i) ;										// north surrounding element
	iS = (int *)malloc(mem_size_i) ;										// south surrounding element
	mem_size_j = sizeof(int) * Nc;											//
	jW = (int *)malloc(mem_size_j) ;										// west surrounding element
	jE = (int *)malloc(mem_size_j) ;										// east surrounding element

	// N/S/W/E indices of surrounding pixels (every element of IMAGE)
	for (i=0; i<Nr; i++) {
		iN[i] = i-1;														// holds index of IMAGE row above
		iS[i] = i+1;														// holds index of IMAGE row below
	}
	for (j=0; j<Nc; j++) {
		jW[j] = j-1;														// holds index of IMAGE column on the left
		jE[j] = j+1;														// holds index of IMAGE column on the right
	}

	// N/S/W/E boundary conditions, fix surrounding indices outside boundary of image
	iN[0]    = 0;															// changes IMAGE top row index from -1 to 0
	iS[Nr-1] = Nr-1;														// changes IMAGE bottom row index from Nr to Nr-1 
	jW[0]    = 0;															// changes IMAGE leftmost column index from -1 to 0
	jE[Nc-1] = Nc-1;														// changes IMAGE rightmost column index from Nc to Nc-1

	//================================================================================80
	// 	GPU SETUP
	//================================================================================80

	// allocate memory for entire IMAGE on DEVICE
	mem_size = sizeof(fp) * Ne;																		// get the size of float representation of input IMAGE
	cudaMalloc((void **)&d_I, mem_size);														//

	// allocate memory for coordinates on DEVICE
	cudaMalloc((void **)&d_iN, mem_size_i);													//
	cudaMemcpy(d_iN, iN, mem_size_i, cudaMemcpyHostToDevice);				//
	cudaMalloc((void **)&d_iS, mem_size_i);													// 
	cudaMemcpy(d_iS, iS, mem_size_i, cudaMemcpyHostToDevice);				//
	cudaMalloc((void **)&d_jE, mem_size_j);													//
	cudaMemcpy(d_jE, jE, mem_size_j, cudaMemcpyHostToDevice);				//
	cudaMalloc((void **)&d_jW, mem_size_j);													// 
	cudaMemcpy(d_jW, jW, mem_size_j, cudaMemcpyHostToDevice);			//

	// allocate memory for partial sums on DEVICE
	cudaMalloc((void **)&d_sums, mem_size);													//
	cudaMalloc((void **)&d_sums2, mem_size);												//

	// allocate memory for derivatives
	cudaMalloc((void **)&d_dN, mem_size);														// 
	cudaMalloc((void **)&d_dS, mem_size);														// 
	cudaMalloc((void **)&d_dW, mem_size);													// 
	cudaMalloc((void **)&d_dE, mem_size);														// 

	// allocate memory for coefficient on DEVICE
	cudaMalloc((void **)&d_c, mem_size);														// 

	checkCUDAError("setup");

	//================================================================================80
	// 	KERNEL EXECUTION PARAMETERS
	//================================================================================80

	// all kernels operating on entire matrix
	threads.x = NUMBER_THREADS;												// define the number of threads in the block
	threads.y = 1;
	blocks_x = Ne/threads.x;
	if (Ne % threads.x != 0){												// compensate for division remainder above by adding one grid
		blocks_x = blocks_x + 1;																	
	}
	blocks.x = blocks_x;													// define the number of blocks in the grid
	blocks.y = 1;

	time5 = get_time();

	//================================================================================80
	// 	COPY INPUT TO CPU
	//================================================================================80

	cudaMemcpy(d_I, image, mem_size, cudaMemcpyHostToDevice);

	time6 = get_time();

	//================================================================================80
	// 	SCALE IMAGE DOWN FROM 0-255 TO 0-1 AND EXTRACT
	//================================================================================80

	extract<<<blocks, threads>>>(	Ne,
									d_I);

	checkCUDAError("extract");

	time7 = get_time();

	//================================================================================80
	// 	COMPUTATION
	//================================================================================80

	// printf("iterations: ");

	// execute main loop
	for (iter=0; iter<niter; iter++){										// do for the number of iterations input parameter

	// printf("%d ", iter);
	// fflush(NULL);

		// execute square kernel
		prepare<<<blocks, threads>>>(	Ne,
										d_I,
										d_sums,
										d_sums2);

		checkCUDAError("prepare");

		// performs subsequent reductions of sums
		blocks2.x = blocks.x;												// original number of blocks
		blocks2.y = blocks.y;												
		no = Ne;														// original number of sum elements
		mul = 1;														// original multiplier

		while(blocks2.x != 0){

			checkCUDAError("before reduce");

			// run kernel
			reduce<<<blocks2, threads>>>(	Ne,
											no,
											mul,
											d_sums, 
											d_sums2);

			checkCUDAError("reduce");

			// update execution parameters
			no = blocks2.x;												// get current number of elements
			if(blocks2.x == 1){
				blocks2.x = 0;
			}
			else{
				mul = mul * NUMBER_THREADS;									// update the increment
				blocks_x = blocks2.x/threads.x;								// number of blocks
				if (blocks2.x % threads.x != 0){							// compensate for division remainder above by adding one grid
					blocks_x = blocks_x + 1;
				}
				blocks2.x = blocks_x;
				blocks2.y = 1;
			}

			checkCUDAError("after reduce");

		}

		checkCUDAError("before copy sum");

		// copy total sums to device
		mem_size_single = sizeof(fp) * 1;
		cudaMemcpy(&total, d_sums, mem_size_single, cudaMemcpyDeviceToHost);
		cudaMemcpy(&total2, d_sums2, mem_size_single, cudaMemcpyDeviceToHost);

		checkCUDAError("copy sum");

		// calculate statistics
		meanROI	= total / fp(NeROI);										// gets mean (average) value of element in ROI
		meanROI2 = meanROI * meanROI;										//
		varROI = (total2 / fp(NeROI)) - meanROI2;						// gets variance of ROI								
		q0sqr = varROI / meanROI2;											// gets standard deviation of ROI

		// execute srad kernel
		srad<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
									Nr,										// # of rows in input image
									Nc,										// # of columns in input image
									Ne,										// # of elements in input image
									d_iN,									// indices of North surrounding pixels
									d_iS,									// indices of South surrounding pixels
									d_jE,									// indices of East surrounding pixels
									d_jW,									// indices of West surrounding pixels
									d_dN,									// North derivative
									d_dS,									// South derivative
									d_dW,									// West derivative
									d_dE,									// East derivative
									q0sqr,									// standard deviation of ROI 
									d_c,									// diffusion coefficient
									d_I);									// output image

		checkCUDAError("srad");

		// execute srad2 kernel
		srad2<<<blocks, threads>>>(	lambda,									// SRAD coefficient 
									Nr,										// # of rows in input image
									Nc,										// # of columns in input image
									Ne,										// # of elements in input image
									d_iN,									// indices of North surrounding pixels
									d_iS,									// indices of South surrounding pixels
									d_jE,									// indices of East surrounding pixels
									d_jW,									// indices of West surrounding pixels
									d_dN,									// North derivative
									d_dS,									// South derivative
									d_dW,									// West derivative
									d_dE,									// East derivative
									d_c,									// diffusion coefficient
									d_I);									// output image

		checkCUDAError("srad2");

	}

	// printf("\n");

	time8 = get_time();

	//================================================================================80
	// 	SCALE IMAGE UP FROM 0-1 TO 0-255 AND COMPRESS
	//================================================================================80

	compress<<<blocks, threads>>>(	Ne,
									d_I);

	checkCUDAError("compress");

	time9 = get_time();

	//================================================================================80
	// 	COPY RESULTS BACK TO CPU
	//================================================================================80

	cudaMemcpy(image, d_I, mem_size, cudaMemcpyDeviceToHost);

	checkCUDAError("copy back");

	if (verify_cpu) {
		verify_status = verify_cpu_reference(
			image,
			image_ori,
			image_ori_rows,
			image_ori_cols,
			Nr,
			Nc,
			niter,
			lambda,
			iN,
			iS,
			jE,
			jW,
			blocks.x);
		if (verify_status != 0) {
			rodinia_print_fail("SRAD CPU reference verification");
		}
	}

	time10 = get_time();

	//================================================================================80
	// 	WRITE IMAGE AFTER PROCESSING
	//================================================================================80

	write_graphics(	"image_out.pgm",
					image,
					Nr,
					Nc,
					1,
					255);

	time11 = get_time();

	//================================================================================80
	//	DEALLOCATE
	//================================================================================80

	free(image_ori);
	free(image);
	free(iN); 
	free(iS); 
	free(jW); 
	free(jE);

	cudaFree(d_I);
	cudaFree(d_c);
	cudaFree(d_iN);
	cudaFree(d_iS);
	cudaFree(d_jE);
	cudaFree(d_jW);
	cudaFree(d_dN);
	cudaFree(d_dS);
	cudaFree(d_dE);
	cudaFree(d_dW);
	cudaFree(d_sums);
	cudaFree(d_sums2);

	time12 = get_time();

	//================================================================================80
	//	DISPLAY TIMING
	//================================================================================80

	printf("Time spent in different stages of the application:\n");
	printf("%15.12f s, %15.12f % : SETUP VARIABLES\n", 														(float) (time1-time0) / 1000000, (float) (time1-time0) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : READ COMMAND LINE PARAMETERS\n", 										(float) (time2-time1) / 1000000, (float) (time2-time1) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : READ IMAGE FROM FILE\n", 												(float) (time3-time2) / 1000000, (float) (time3-time2) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : RESIZE IMAGE\n", 														(float) (time4-time3) / 1000000, (float) (time4-time3) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : GPU DRIVER INIT, CPU/GPU SETUP, MEMORY ALLOCATION\n", 					(float) (time5-time4) / 1000000, (float) (time5-time4) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : COPY DATA TO CPU->GPU\n", 												(float) (time6-time5) / 1000000, (float) (time6-time5) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : EXTRACT IMAGE\n", 														(float) (time7-time6) / 1000000, (float) (time7-time6) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : COMPUTE\n", 																(float) (time8-time7) / 1000000, (float) (time8-time7) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : COMPRESS IMAGE\n", 														(float) (time9-time8) / 1000000, (float) (time9-time8) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : COPY DATA TO GPU->CPU\n", 												(float) (time10-time9) / 1000000, (float) (time10-time9) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : SAVE IMAGE INTO FILE\n", 												(float) (time11-time10) / 1000000, (float) (time11-time10) / (float) (time12-time0) * 100);
	printf("%15.12f s, %15.12f % : FREE MEMORY\n", 															(float) (time12-time11) / 1000000, (float) (time12-time11) / (float) (time12-time0) * 100);
	printf("Total time:\n");
	printf("%.12f s\n", 																					(float) (time12-time0) / 1000000);

	return verify_status == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}

//====================================================================================================100
//	END OF FILE
//====================================================================================================100
