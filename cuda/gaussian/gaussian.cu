/*-----------------------------------------------------------
 ** gaussian.cu -- The program is to solve a linear system Ax = b
 **   by using Gaussian Elimination. The algorithm on page 101
 **   ("Foundations of Parallel Programming") is used.  
 **   The sequential version is gaussian.c.  This parallel 
 **   implementation converts three independent for() loops 
 **   into three Fans.  Use the data file ge_3.dat to verify 
 **   the correction of the output. 
 **
 ** Written by Andreas Kura, 02/15/95
 ** Modified by Chong-wei Xu, 04/20/95
 ** Modified by Chris Gregg for CUDA, 07/20/2009
 **-----------------------------------------------------------
 */
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "cuda.h"
#include <string.h>
#include <math.h>
#include "../../common/rodinia_verify.h"

#ifdef TIMING
#include "timing.h"
#endif

#ifdef RD_WG_SIZE_0_0
        #define MAXBLOCKSIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
        #define MAXBLOCKSIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
        #define MAXBLOCKSIZE RD_WG_SIZE
#else
        #define MAXBLOCKSIZE 512
#endif

//2D defines. Go from specific to general                                                
#ifdef RD_WG_SIZE_1_0
        #define BLOCK_SIZE_XY RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
        #define BLOCK_SIZE_XY RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
        #define BLOCK_SIZE_XY RD_WG_SIZE
#else
        #define BLOCK_SIZE_XY 4
#endif

#ifdef TIMING
struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

int Size;
float *a, *b, *finalVec;
float *m;

FILE *fp;

#define GAUSSIAN_ABS_TOLERANCE 0.001f
#define GAUSSIAN_REL_TOLERANCE 0.001f
#define GAUSSIAN_PIVOT_TOLERANCE 1.0e-20f

typedef struct {
    int verbose;
    int verify_cpu;
} Options;

int InitProblemOnce(const char *filename);
void InitPerRun();
void ForwardSub();
void BackSub();
__global__ void Fan1(float *m, float *a, int Size, int t);
__global__ void Fan2(float *m, float *a, float *b,int Size, int j1, int t);
int InitMat(float *ary, int nrow, int ncol);
int InitAry(float *ary, int ary_size);
void PrintMat(float *ary, int nrow, int ncolumn);
void PrintAry(float *ary, int ary_size);
void PrintDeviceProperties();
void checkCUDAError(const char *msg);
static void usage(const char *program);
static int initialize_from_args(int argc, char **argv, Options *options);
static int initialize_generated_problem(int size);
static int parse_positive_size(const char *text, int *value);
static float *duplicate_array(const float *source, int count, const char *label);
static void release_problem_buffers();
static int run_gaussian(int verbose, int verify_cpu, const float *initial_a, const float *initial_b);
static int verify_cpu_reference(const float *actual, const float *initial_a, const float *initial_b);
static int solve_cpu_reference(float *matrix, float *rhs, float *expected);
static int compare_solution(const float *actual, const float *expected);

unsigned int totalKernelTime = 0;

// create both matrix and right hand side, Ke Wang 2013/08/12 11:51:06
void
create_matrix(float *m, int size){
  int i,j;
  float lamda = -0.01;
  float coe[2*size-1];
  float coe_i =0.0;

  for (i=0; i < size; i++)
    {
      coe_i = 10*exp(lamda*i); 
      j=size-1+i;     
      coe[j]=coe_i;
      j=size-1-i;     
      coe[j]=coe_i;
    }


  for (i=0; i < size; i++) {
      for (j=0; j < size; j++) {
	m[i*size+j]=coe[size-1-i+j];
      }
  }


}


int main(int argc, char *argv[])
{
  printf("WG size of kernel 1 = %d, WG size of kernel 2= %d X %d\n", MAXBLOCKSIZE, BLOCK_SIZE_XY, BLOCK_SIZE_XY);
    if (argc < 2) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    PrintDeviceProperties();

    Options options;
    if (initialize_from_args(argc, argv, &options) != 0) {
        usage(argv[0]);
        release_problem_buffers();
        return EXIT_FAILURE;
    }

    float *initial_a = NULL;
    float *initial_b = NULL;
    if (options.verify_cpu) {
        initial_a = duplicate_array(a, Size * Size, "initial matrix");
        initial_b = duplicate_array(b, Size, "initial rhs");
        if (initial_a == NULL || initial_b == NULL) {
            free(initial_a);
            free(initial_b);
            release_problem_buffers();
            return EXIT_FAILURE;
        }
    }

    int status = run_gaussian(options.verbose, options.verify_cpu, initial_a, initial_b);

    free(initial_a);
    free(initial_b);
    free(finalVec);
    release_problem_buffers();

#ifdef  TIMING
	printf("Exec: %f\n", kernel_time);
#endif

    return status;
}

static void usage(const char *program)
{
    fprintf(stderr, "Usage: %s -f filename / -s size [-q] [--verify-cpu]\n\n", program);
    fprintf(stderr, "-q (quiet) suppresses printing the matrix and result values.\n");
    fprintf(stderr, "-f (filename) path of input file\n");
    fprintf(stderr, "-s (size) size of matrix. Create matrix and rhs in this program\n");
    fprintf(stderr, "--verify-cpu compares the CUDA solution with a CPU Gaussian reference.\n");
    fprintf(stderr, "The first line of the file contains the dimension of the matrix, n.\n");
    fprintf(stderr, "The second line of the file is a newline.\n");
    fprintf(stderr, "The next n lines contain n tab separated values for the matrix.\n");
    fprintf(stderr, "The next line of the file is a newline.\n");
    fprintf(stderr, "The next line of the file is a 1xn vector with tab separated values.\n");
    fprintf(stderr, "The next line of the file is a newline. (optional)\n");
    fprintf(stderr, "The final line of the file is the pre-computed solution. (optional)\n");
}

static int initialize_from_args(int argc, char **argv, Options *options)
{
    int input_loaded = 0;
    options->verbose = 0;
    options->verify_cpu = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--verify-cpu") == 0) {
            options->verify_cpu = 1;
            continue;
        }
        if (strcmp(argv[i], "-q") == 0) {
            options->verbose = 0;
            continue;
        }
        if (strcmp(argv[i], "-s") == 0) {
            if (input_loaded) {
                fprintf(stderr, "Only one Gaussian input may be specified\n");
                return -1;
            }
            int parsed_size;
            if (++i >= argc || parse_positive_size(argv[i], &parsed_size) != 0) {
                fprintf(stderr, "Invalid Gaussian matrix size\n");
                return -1;
            }
            if (initialize_generated_problem(parsed_size) != 0) {
                return -1;
            }
            input_loaded = 1;
            continue;
        }
        if (strcmp(argv[i], "-f") == 0) {
            if (input_loaded) {
                fprintf(stderr, "Only one Gaussian input may be specified\n");
                return -1;
            }
            if (++i >= argc) {
                fprintf(stderr, "Missing Gaussian input filename\n");
                return -1;
            }
            printf("Read file from %s \n", argv[i]);
            if (InitProblemOnce(argv[i]) != 0) {
                return -1;
            }
            input_loaded = 1;
            continue;
        }
        fprintf(stderr, "Unknown option: %s\n", argv[i]);
        return -1;
    }

    if (!input_loaded) {
        fprintf(stderr, "Gaussian input must be specified with -f or -s\n");
        return -1;
    }
    return 0;
}

static int initialize_generated_problem(int size)
{
    Size = size;
    printf("Create matrix internally in parse, size = %d \n", Size);

    a = (float *)malloc(Size * Size * sizeof(float));
    b = (float *)malloc(Size * sizeof(float));
    m = (float *)malloc(Size * Size * sizeof(float));
    if (a == NULL || b == NULL || m == NULL) {
        fprintf(stderr, "Cannot allocate Gaussian input buffers\n");
        release_problem_buffers();
        return -1;
    }

    create_matrix(a, Size);
    for (int j = 0; j < Size; j++) {
        b[j] = 1.0f;
    }
    return 0;
}

static int parse_positive_size(const char *text, int *value)
{
    char *end = NULL;
    long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed < 2 || parsed > 2147483647L) {
        return -1;
    }
    *value = (int)parsed;
    return 0;
}

static float *duplicate_array(const float *source, int count, const char *label)
{
    float *copy = (float *)malloc(count * sizeof(float));
    if (copy == NULL) {
        fprintf(stderr, "Cannot allocate copy of %s\n", label);
        return NULL;
    }
    memcpy(copy, source, count * sizeof(float));
    return copy;
}

static void release_problem_buffers()
{
    free(m);
    free(a);
    free(b);
    m = NULL;
    a = NULL;
    b = NULL;
}

static int run_gaussian(int verbose, int verify_cpu, const float *initial_a, const float *initial_b)
{
    InitPerRun();
    struct timeval time_start;
    gettimeofday(&time_start, NULL);

    ForwardSub();

    struct timeval time_end;
    gettimeofday(&time_end, NULL);
    unsigned int time_total = (time_end.tv_sec * 1000000 + time_end.tv_usec) -
        (time_start.tv_sec * 1000000 + time_start.tv_usec);

    if (verbose) {
        printf("Matrix m is: \n");
        PrintMat(m, Size, Size);

        printf("Matrix a is: \n");
        PrintMat(a, Size, Size);

        printf("Array b is: \n");
        PrintAry(b, Size);
    }
    BackSub();
    if (verbose) {
        printf("The final solution is: \n");
        PrintAry(finalVec, Size);
    }
    printf("\nTime total (including memory transfers)\t%f sec\n", time_total * 1e-6);
    printf("Time for CUDA kernels:\t%f sec\n", totalKernelTime * 1e-6);

    if (verify_cpu) {
        if (verify_cpu_reference(finalVec, initial_a, initial_b) != 0) {
            rodinia_print_fail("Gaussian CPU reference verification");
            return EXIT_FAILURE;
        }
        return EXIT_SUCCESS;
    }
    return EXIT_SUCCESS;
}

static int verify_cpu_reference(const float *actual, const float *initial_a, const float *initial_b)
{
    float *matrix = duplicate_array(initial_a, Size * Size, "CPU reference matrix");
    float *rhs = duplicate_array(initial_b, Size, "CPU reference rhs");
    float *expected = (float *)malloc(Size * sizeof(float));
    if (matrix == NULL || rhs == NULL || expected == NULL) {
        fprintf(stderr, "Cannot allocate Gaussian CPU reference buffers\n");
        free(matrix);
        free(rhs);
        free(expected);
        return -1;
    }

    int status = solve_cpu_reference(matrix, rhs, expected);
    if (status == 0) {
        status = compare_solution(actual, expected);
    }

    free(matrix);
    free(rhs);
    free(expected);
    if (status != 0) {
        return -1;
    }
    return rodinia_print_pass("Gaussian CPU reference verification");
}

static int solve_cpu_reference(float *matrix, float *rhs, float *expected)
{
    for (int t = 0; t < Size - 1; t++) {
        float pivot = matrix[Size * t + t];
        if (fabsf(pivot) <= GAUSSIAN_PIVOT_TOLERANCE) {
            fprintf(stderr, "Gaussian CPU reference saw near-zero pivot at row %d: %g\n", t, pivot);
            return -1;
        }
        for (int row = t + 1; row < Size; row++) {
            float multiplier = matrix[Size * row + t] / pivot;
            matrix[Size * row + t] = 0.0f;
            for (int col = t + 1; col < Size; col++) {
                matrix[Size * row + col] -= multiplier * matrix[Size * t + col];
            }
            rhs[row] -= multiplier * rhs[t];
        }
    }

    for (int offset = 0; offset < Size; offset++) {
        int row = Size - offset - 1;
        float sum = rhs[row];
        for (int col = row + 1; col < Size; col++) {
            sum -= matrix[Size * row + col] * expected[col];
        }
        float pivot = matrix[Size * row + row];
        if (fabsf(pivot) <= GAUSSIAN_PIVOT_TOLERANCE) {
            fprintf(stderr, "Gaussian CPU reference saw near-zero pivot at row %d: %g\n", row, pivot);
            return -1;
        }
        expected[row] = sum / pivot;
    }
    return 0;
}

static int compare_solution(const float *actual, const float *expected)
{
    for (int index = 0; index < Size; index++) {
        float diff = fabsf(actual[index] - expected[index]);
        float tolerance = GAUSSIAN_ABS_TOLERANCE + GAUSSIAN_REL_TOLERANCE * fabsf(expected[index]);
        if (!isfinite(actual[index]) || diff > tolerance) {
            fprintf(stderr,
                "Gaussian CPU reference mismatch at index %d: actual=%g expected=%g diff=%g tolerance=%g\n",
                index,
                actual[index],
                expected[index],
                diff,
                tolerance);
            return -1;
        }
    }
    return 0;
}
/*------------------------------------------------------
 ** PrintDeviceProperties
 **-----------------------------------------------------
 */
void PrintDeviceProperties(){
	cudaDeviceProp deviceProp;  
	int nDevCount = 0;  
	
	cudaGetDeviceCount( &nDevCount );  
	printf( "Total Device found: %d", nDevCount );  
	for (int nDeviceIdx = 0; nDeviceIdx < nDevCount; ++nDeviceIdx )  
	{  
	    memset( &deviceProp, 0, sizeof(deviceProp));  
	    if( cudaSuccess == cudaGetDeviceProperties(&deviceProp, nDeviceIdx))  
	        {
				printf( "\nDevice Name \t\t - %s ", deviceProp.name );  
			    printf( "\n**************************************");  
			    printf( "\nTotal Global Memory\t\t\t - %lu KB", deviceProp.totalGlobalMem/1024 );  
			    printf( "\nShared memory available per block \t - %lu KB", deviceProp.sharedMemPerBlock/1024 );  
			    printf( "\nNumber of registers per thread block \t - %d", deviceProp.regsPerBlock );  
			    printf( "\nWarp size in threads \t\t\t - %d", deviceProp.warpSize );  
			    printf( "\nMemory Pitch \t\t\t\t - %zu bytes", deviceProp.memPitch );  
			    printf( "\nMaximum threads per block \t\t - %d", deviceProp.maxThreadsPerBlock );  
			    printf( "\nMaximum Thread Dimension (block) \t - %d %d %d", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2] );  
			    printf( "\nMaximum Thread Dimension (grid) \t - %d %d %d", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2] );  
			    printf( "\nTotal constant memory \t\t\t - %zu bytes", deviceProp.totalConstMem );  
			    printf( "\nCUDA ver \t\t\t\t - %d.%d", deviceProp.major, deviceProp.minor );  
			    printf( "\nTexture Alignment \t\t\t - %zu bytes", deviceProp.textureAlignment );  
			    printf( "\nNumber of Multi processors \t\t - %d\n\n", deviceProp.multiProcessorCount );  
			}  
	    else  
	        printf( "\n%s", cudaGetErrorString(cudaGetLastError()));  
	}  
}
 
 
/*------------------------------------------------------
 ** InitProblemOnce -- Initialize all of matrices and
 ** vectors by opening a data file specified by the user.
 **
 ** We used dynamic array *a, *b, and *m to allocate
 ** the memory storages.
 **------------------------------------------------------
 */
int InitProblemOnce(const char *filename)
{
	//char *filename = argv[1];
	
	//printf("Enter the data file name: ");
	//scanf("%s", filename);
	//printf("The file name is: %s\n", filename);
	
	fp = fopen(filename, "r");
	if (fp == NULL) {
		fprintf(stderr, "Cannot open Gaussian input file: %s\n", filename);
		return -1;
	}
	
	if (fscanf(fp, "%d", &Size) != 1 || Size < 2) {
		fprintf(stderr, "Invalid Gaussian matrix size in input file: %s\n", filename);
		fclose(fp);
		fp = NULL;
		return -1;
	}
	 
	a = (float *) malloc(Size * Size * sizeof(float));
	b = (float *) malloc(Size * sizeof(float));
	m = (float *) malloc(Size * Size * sizeof(float));
	if (a == NULL || b == NULL || m == NULL) {
		fprintf(stderr, "Cannot allocate Gaussian input buffers\n");
		fclose(fp);
		fp = NULL;
		release_problem_buffers();
		return -1;
	}

	if (InitMat(a, Size, Size) != 0) {
		fclose(fp);
		fp = NULL;
		release_problem_buffers();
		return -1;
	}
	//printf("The input matrix a is:\n");
	//PrintMat(a, Size, Size);
	if (InitAry(b, Size) != 0) {
		fclose(fp);
		fp = NULL;
		release_problem_buffers();
		return -1;
	}
	//printf("The input array b is:\n");
	//PrintAry(b, Size);

	fclose(fp);
	fp = NULL;
	return 0;
}

/*------------------------------------------------------
 ** InitPerRun() -- Initialize the contents of the
 ** multipier matrix **m
 **------------------------------------------------------
 */
void InitPerRun() 
{
	int i;
	for (i=0; i<Size*Size; i++)
			*(m+i) = 0.0;
}

/*-------------------------------------------------------
 ** Fan1() -- Calculate multiplier matrix
 ** Pay attention to the index.  Index i give the range
 ** which starts from 0 to range-1.  The real values of
 ** the index should be adjust and related with the value
 ** of t which is defined on the ForwardSub().
 **-------------------------------------------------------
 */
__global__ void Fan1(float *m_cuda, float *a_cuda, int Size, int t)
{   
	//if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) printf(".");
	//printf("blockIDx.x:%d,threadIdx.x:%d,Size:%d,t:%d,Size-1-t:%d\n",blockIdx.x,threadIdx.x,Size,t,Size-1-t);

	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	*(m_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) = *(a_cuda+Size*(blockDim.x*blockIdx.x+threadIdx.x+t+1)+t) / *(a_cuda+Size*t+t);
}

/*-------------------------------------------------------
 ** Fan2() -- Modify the matrix A into LUD
 **-------------------------------------------------------
 */ 

__global__ void Fan2(float *m_cuda, float *a_cuda, float *b_cuda,int Size, int j1, int t)
{
	if(threadIdx.x + blockIdx.x * blockDim.x >= Size-1-t) return;
	if(threadIdx.y + blockIdx.y * blockDim.y >= Size-t) return;
	
	int xidx = blockIdx.x * blockDim.x + threadIdx.x;
	int yidx = blockIdx.y * blockDim.y + threadIdx.y;
	//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
	
	a_cuda[Size*(xidx+1+t)+(yidx+t)] -= m_cuda[Size*(xidx+1+t)+t] * a_cuda[Size*t+(yidx+t)];
	//a_cuda[xidx+1+t][yidx+t] -= m_cuda[xidx+1+t][t] * a_cuda[t][yidx+t];
	if(yidx == 0){
		//printf("blockIdx.x:%d,threadIdx.x:%d,blockIdx.y:%d,threadIdx.y:%d,blockDim.x:%d,blockDim.y:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,blockDim.x,blockDim.y);
		//printf("xidx:%d,yidx:%d\n",xidx,yidx);
		b_cuda[xidx+1+t] -= m_cuda[Size*(xidx+1+t)+(yidx+t)] * b_cuda[t];
	}
}

/*------------------------------------------------------
 ** ForwardSub() -- Forward substitution of Gaussian
 ** elimination.
 **------------------------------------------------------
 */
void ForwardSub()
{
	int t;
    float *m_cuda,*a_cuda,*b_cuda;
	
	// allocate memory on GPU
	cudaMalloc((void **) &m_cuda, Size * Size * sizeof(float));
	 
	cudaMalloc((void **) &a_cuda, Size * Size * sizeof(float));
	
	cudaMalloc((void **) &b_cuda, Size * sizeof(float));	

	// copy memory to GPU
	cudaMemcpy(m_cuda, m, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(a_cuda, a, Size * Size * sizeof(float),cudaMemcpyHostToDevice );
	cudaMemcpy(b_cuda, b, Size * sizeof(float),cudaMemcpyHostToDevice );
	
	int block_size,grid_size;
	
	block_size = MAXBLOCKSIZE;
	grid_size = (Size/block_size) + (!(Size%block_size)? 0:1);
	//printf("1d grid size: %d\n",grid_size);


	dim3 dimBlock(block_size);
	dim3 dimGrid(grid_size);
	//dim3 dimGrid( (N/dimBlock.x) + (!(N%dimBlock.x)?0:1) );
	
	int blockSize2d, gridSize2d;
	blockSize2d = BLOCK_SIZE_XY;
	gridSize2d = (Size/blockSize2d) + (!(Size%blockSize2d?0:1)); 
	
	dim3 dimBlockXY(blockSize2d,blockSize2d);
	dim3 dimGridXY(gridSize2d,gridSize2d);

#ifdef  TIMING
	gettimeofday(&tv_kernel_start, NULL);
#endif

    // begin timing kernels
    struct timeval time_start;
    gettimeofday(&time_start, NULL);
	for (t=0; t<(Size-1); t++) {
		Fan1<<<dimGrid,dimBlock>>>(m_cuda,a_cuda,Size,t);
		cudaDeviceSynchronize();
		Fan2<<<dimGridXY,dimBlockXY>>>(m_cuda,a_cuda,b_cuda,Size,Size-t,t);
		cudaDeviceSynchronize();
		checkCUDAError("Fan2");
	}
	// end timing kernels
	struct timeval time_end;
    gettimeofday(&time_end, NULL);
    totalKernelTime = (time_end.tv_sec * 1000000 + time_end.tv_usec) - (time_start.tv_sec * 1000000 + time_start.tv_usec);
	
#ifdef  TIMING
	tvsub(&time_end, &tv_kernel_start, &tv);
	kernel_time += tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	// copy memory back to CPU
	cudaMemcpy(m, m_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(a, a_cuda, Size * Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaMemcpy(b, b_cuda, Size * sizeof(float),cudaMemcpyDeviceToHost );
	cudaFree(m_cuda);
	cudaFree(a_cuda);
	cudaFree(b_cuda);
}

/*------------------------------------------------------
 ** BackSub() -- Backward substitution
 **------------------------------------------------------
 */

void BackSub()
{
	// create a new vector to hold the final answer
	finalVec = (float *) malloc(Size * sizeof(float));
	// solve "bottom up"
	int i,j;
	for(i=0;i<Size;i++){
		finalVec[Size-i-1]=b[Size-i-1];
		for(j=0;j<i;j++)
		{
			finalVec[Size-i-1]-=*(a+Size*(Size-i-1)+(Size-j-1)) * finalVec[Size-j-1];
		}
		finalVec[Size-i-1]=finalVec[Size-i-1]/ *(a+Size*(Size-i-1)+(Size-i-1));
	}
}

int InitMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			if (fscanf(fp, "%f", ary+Size*i+j) != 1) {
				fprintf(stderr, "Invalid Gaussian matrix value at row %d column %d\n", i, j);
				return -1;
			}
		}
	}  
	return 0;
}

/*------------------------------------------------------
 ** PrintMat() -- Print the contents of the matrix
 **------------------------------------------------------
 */
void PrintMat(float *ary, int nrow, int ncol)
{
	int i, j;
	
	for (i=0; i<nrow; i++) {
		for (j=0; j<ncol; j++) {
			printf("%8.2f ", *(ary+Size*i+j));
		}
		printf("\n");
	}
	printf("\n");
}

/*------------------------------------------------------
 ** InitAry() -- Initialize the array (vector) by reading
 ** data from the data file
 **------------------------------------------------------
 */
int InitAry(float *ary, int ary_size)
{
	int i;
	
	for (i=0; i<ary_size; i++) {
		if (fscanf(fp, "%f", &ary[i]) != 1) {
			fprintf(stderr, "Invalid Gaussian vector value at index %d\n", i);
			return -1;
		}
	}
	return 0;
}  

/*------------------------------------------------------
 ** PrintAry() -- Print the contents of the array (vector)
 **------------------------------------------------------
 */
void PrintAry(float *ary, int ary_size)
{
	int i;
	for (i=0; i<ary_size; i++) {
		printf("%.2f ", ary[i]);
	}
	printf("\n\n");
}
void checkCUDAError(const char *msg)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) 
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, 
                                  cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }                         
}
