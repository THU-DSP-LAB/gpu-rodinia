//====================================================================================================100
//		UPDATE
//====================================================================================================100

// Lukasz G. Szafaryn 24 JAN 09

//====================================================================================================100
//		DESCRIPTION
//====================================================================================================100

// Myocyte application models cardiac myocyte (heart muscle cell) and simulates its behavior according to the work by Saucerman and Bers [8]. The model integrates 
// cardiac myocyte electrical activity with the calcineurin pathway, which is a key aspect of the development of heart failure. The model spans large number of temporal 
// scales to reflect how changes in heart rate as observed during exercise or stress contribute to calcineurin pathway activation, which ultimately leads to the expression 
// of numerous genes that remodel the heart’s structure. It can be used to identify potential therapeutic targets that may be useful for the treatment of heart failure. 
// Biochemical reactions, ion transport and electrical activity in the cell are modeled with 91 ordinary differential equations (ODEs) that are determined by more than 200 
// experimentally validated parameters. The model is simulated by solving this group of ODEs for a specified time interval. The process of ODE solving is based on the 
// causal relationship between values of ODEs at different time steps, thus it is mostly sequential. At every dynamically determined time step, the solver evaluates the 
// model consisting of a set of 91 ODEs and 480 supporting equations to determine behavior of the system at that particular time instance. If evaluation results are not 
// within the expected tolerance at a given time step (usually as a result of incorrect determination of the time step), another calculation attempt is made at a modified 
// (usually reduced) time step. Since the ODEs are stiff (exhibit fast rate of change within short time intervals), they need to be simulated at small time scales with an 
// adaptive step size solver. 

//	1) The original version of the current solver code was obtained from: Mathematics Source Library (http://mymathlib.webtrellis.net/index.html). The solver has been 
//      somewhat modified to tailor it to our needs. However, it can be reverted back to original form or modified to suit other simulations.
// 2) This solver and particular solving algorithm used with it (embedded_fehlberg_7_8) were adapted to work with a set of equations, not just one like in original version.
//	3) In order for solver to provide deterministic number of steps (needed for particular amount of memore previousely allocated for results), every next step is 
//      incremented by 1 time unit (h_init).
//	4) Function assumes that simulation starts at some point of time (whatever time the initial values are provided for) and runs for the number of miliseconds (xmax) 
//      specified by the uses as a parameter on command line.
// 5) The appropriate amount of memory is previousely allocated for that range (y).
//	6) This setup in 3) - 5) allows solver to adjust the step ony from current time instance to current time instance + 0.9. The next time instance is current time instance + 1;
//	7) The original solver cannot handle cases when equations return NAN and INF values due to discontinuities and /0. That is why equations provided by user need to 
//      make sure that no NAN and INF are returned.
// 8) Application reads initial data and parameters from text files: y.txt and params.txt respectively that need to be located in the same folder as source files. 
//     For simplicity and testing purposes only, when multiple number of simulation instances is specified, application still reads initial data from the same input files. That 
//     can be modified in this source code.

//====================================================================================================100
//		IMPLEMENTATION-SPECIFIC DESCRIPTION (CUDA)
//====================================================================================================100

// This is the CUDA version of Myocyte code.

// The original single-threaded code was written in MATLAB and used MATLAB ode45 ODE solver. In the process of accelerating this code, we arrived with the 
// intermediate versions that used single-threaded Sundials CVODE solver which evaluated model parallelized with CUDA at each time step. In order to convert entire 
// solver to CUDA code (to remove some of the operational overheads such as kernel launches and data transfer in CUDA) we used a simpler solver, from Mathematics 
// Source Library, and tailored it to our needs. The parallelism in the cardiac myocyte model is on a very fine-grained level, close to that of ILP, therefore it is very hard 
// to exploit as DLP or TLB in CUDA code. We were able to divide the model into 4 individual groups that run in parallel. However, even that is not enough work to 
// compensate for some of the CUDA thread launch and data transfer overheads which resulted in performance worse than that of single-threaded C code. Speedup in 
// this code could be achieved only if a customizable accelerator such as FPGA was used for evaluation of the model itself. We also approached the application from 
// another angle and allowed it to run several concurrent simulations, thus turning it into an embarrassingly parallel problem. This version of the code is also useful for 
// scientists who want to run the same simulation with different sets of input parameters. Speedup achieved with CUDA code is variable on the other hand. It depends on 
// the number of concurrent simulations and it saturates around 300 simulations with the speedup of about 10x.

// Speedup numbers reported in the description of this application were obtained on the machine with: Intel Quad Core CPU, 4GB of RAM, Nvidia GTX280 GPU.  

// 1) When running with parallelization inside each simulation instance (value of 3rd command line parameter equal to 0), performance is bad because:
// a) underutilization of GPU (only 1 out of 32 threads in each block)
// b) significant CPU-GPU memory copy overhead
// c) kernel launch overhead (kernel needs to be finished every time model is evaluated as it is the only way to synchronize threads in different blocks)
// 2) When running with parallelization across simulation instances, code gets continues speedup with the increasing number of simulation insances which saturates
//     around 240 instances on GTX280 (roughly corresponding to the number of multiprocessorsXprocessors in GTX280), with the speedup of around 10x compared
//     to serial C version of code. Limited performance is explained mainly by:
// a) significant CPU-GPU memory copy overhead
// b) increasingly uncoalesced memory accesses with the increasing number of workloads
// c) lack of cache compared to CPU, or no use of shared memory to compensate
// d) frequency of GPU shader being lower than that of CPU core
// 3) GPU version has an issue with memory allocation that has not been resolved yet. For certain simulation ranges memory allocation fails, or pointers incorrectly overlap 
//     causeing value trashing.

// The following are the command parameters to the application:
// 1) Simulation time interval which is the number of miliseconds to simulate. Needs to be integer > 0
// 2) Number of instances of simulation to run. Needs to be integer > 0.
// 3) Method of parallelization. Need to be 0 for parallelization inside each simulation instance, or 1 for parallelization across instances.
// Example:
// a.out 100 100 1

//====================================================================================================100
//		DEFINE / INCLUDE
//====================================================================================================100

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>

#include "define.c"

#include "file.c"
#include "timer.c"

#include "work.cu"
#include "work_2.cu"
#include "myocyte_reference.c"

enum {
	DECIMAL_ARGUMENT_BASE = 10,
	MYOCYTE_MODE_INTRA_INSTANCE = 0,
	MYOCYTE_MODE_INTER_INSTANCE = 1
};

typedef struct {
	const char *text;
	const char *name;
	int min_value;
	int max_value;
} IntegerArgument;

static int isDecimalArgument(const char *text)
{
	if (text[0] == '\0') {
		return 0;
	}
	for (const char *cursor = text; *cursor != '\0'; ++cursor) {
		if (*cursor < '0' || *cursor > '9') {
			return 0;
		}
	}
	return 1;
}

static int parseIntegerArgument(const IntegerArgument *argument, int *value)
{
	char *end = NULL;
	if (!isDecimalArgument(argument->text)) {
		fprintf(stderr, "ERROR: invalid %s argument '%s'\n", argument->name, argument->text);
		return -1;
	}
	errno = 0;
	long parsed = strtol(argument->text, &end, DECIMAL_ARGUMENT_BASE);
	if (*end != '\0' || errno == ERANGE ||
		parsed < argument->min_value || parsed > argument->max_value) {
		fprintf(stderr, "ERROR: invalid %s argument '%s'\n", argument->name, argument->text);
		return -1;
	}
	*value = (int)parsed;
	return 0;
}

//====================================================================================================100
//		MAIN FUNCTION
//====================================================================================================100

int main(int argc, char *argv []){

	//================================================================================80
	//		VARIABLES
	//================================================================================80

	//============================================================60
	//		COMMAND LINE PARAMETERS
	//============================================================60

	int xmax;
	int workload;
	int mode;

	//================================================================================80
	// 	GET INPUT PARAMETERS
	//================================================================================80

	//============================================================60
	//		CHECK NUMBER OF ARGUMENTS
	//============================================================60

	if(argc != 4 && argc != 6){
		fprintf(stderr, "ERROR: %d is the incorrect number of arguments\n", argc - 1);
		printUsage(argv[0]);
		return 1;
	}

	//============================================================60
	//		GET AND CHECK PARTICULAR ARGUMENTS
	//============================================================60

	else{

		//========================================40
		//		SPAN
		//========================================40

		IntegerArgument xmax_argument = { argv[1], "xmax", 1, INT_MAX };
		if(parseIntegerArgument(&xmax_argument, &xmax) != 0){
			return 1;
		}

		//========================================40
		//		WORKLOAD
		//========================================40

		IntegerArgument workload_argument = { argv[2], "workload", 1, INT_MAX };
		if(parseIntegerArgument(&workload_argument, &workload) != 0){
			return 1;
		}

		//========================================40
		//		MODE
		//========================================40

		IntegerArgument mode_argument = { argv[3], "mode", MYOCYTE_MODE_INTRA_INSTANCE, MYOCYTE_MODE_INTER_INSTANCE };
		if(parseIntegerArgument(&mode_argument, &mode) != 0){
			return 1;
		}

	}

	const char *save_reference_path = NULL;
	const char *verify_reference_path = NULL;
	if (argc == 6) {
		if (strcmp(argv[4], "--save-reference") == 0) {
			save_reference_path = argv[5];
		} else if (strcmp(argv[4], "--verify-reference") == 0) {
			verify_reference_path = argv[5];
		} else {
			fprintf(stderr, "ERROR: unknown reference option: %s\n", argv[4]);
			printUsage(argv[0]);
			return 1;
		}
	}
	if ((save_reference_path != NULL || verify_reference_path != NULL) && mode != 0) {
		fprintf(stderr, "ERROR: myocyte reference verification requires mode 0 because mode 1 does not write output.txt\n");
		return 1;
	}

	//================================================================================80
	//		EXECUTION IF THERE IS 1 WORKLOAD, PARALLELIZE INSIDE 1 WORKLOAD
	//================================================================================80

	int status = 0;
	if(mode == 0){

		status = work(	xmax,
					workload);

	}

	//================================================================================80
	//		EXECUTION IF THERE ARE MANY WORKLOADS, PARALLELIZE ACROSS WORKLOADS
	//================================================================================80

	else{

		status = work_2(	xmax,
						workload);

	}

	if (status != 0) {
		return status;
	}

	if (save_reference_path != NULL || verify_reference_path != NULL) {
		MyocyteReference reference;
		if (populateReference(xmax, workload, mode, &reference) != 0) {
			return 1;
		}
		if (save_reference_path != NULL) {
			if (writeReference(save_reference_path, &reference) != 0) {
				return 1;
			}
			printf("Myocyte reference saved to '%s'\n", save_reference_path);
		}
		if (verify_reference_path != NULL && verifyReference(verify_reference_path, &reference) != 0) {
			return 1;
		}
	}

//====================================================================================================100
//		END OF FILE
//====================================================================================================100

	return 0;

}
