// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "backprop.h"

#ifdef NV //NVIDIA
	#include <oclUtils.h>
#else
	#include <CL/cl.h>
#endif

#ifdef TIMING
    #include "timing.h"
#endif

////////////////////////////////////////////////////////////////////////////////

// local variables
static cl_context	    context;
static cl_command_queue cmd_queue;
static cl_device_id   * device_list;
static cl_uint          num_devices;

// OCL config
int platform_id_inuse = 0;            // platform id in use (default: 0)
int device_id_inuse = 0;              //device id in use (default : 0)
cl_device_type device_type = CL_DEVICE_TYPE_GPU;
unsigned long long backprop_last_input_hidden_hash = 0;

static unsigned long long hash_bytes(const unsigned char *data, size_t size)
{
	const unsigned long long fnv_offset = 1469598103934665603ULL;
	const unsigned long long fnv_prime = 1099511628211ULL;
	unsigned long long hash = fnv_offset;
	size_t idx;

	for (idx = 0; idx < size; ++idx) {
		hash ^= (unsigned long long)data[idx];
		hash *= fnv_prime;
	}

	return hash;
}

//Primitives for timing
#ifdef TIMING
struct timeval tv;
struct timeval tv_total_start, tv_total_end;
struct timeval tv_init_end;
struct timeval tv_h2d_start, tv_h2d_end;
struct timeval tv_d2h_start, tv_d2h_end;
struct timeval tv_kernel_start, tv_kernel_end;
struct timeval tv_mem_alloc_start, tv_mem_alloc_end;
struct timeval tv_close_start, tv_close_end;
float init_time = 0, mem_alloc_time = 0, h2d_time = 0, kernel_time = 0,
      d2h_time = 0, close_time = 0, total_time = 0;
#endif

static int initialize(void)
{
	cl_int result;
    cl_uint num_platforms;
    cl_device_id selected_device;

    // get OpenCL platforms
	if (clGetPlatformIDs(0, NULL, &num_platforms) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(0,0,*) failed\n"); return -1; }
	cl_platform_id all_platform_id[num_platforms];
	if (clGetPlatformIDs(num_platforms, all_platform_id, NULL) != CL_SUCCESS) { printf("ERROR: clGetPlatformIDs(*,*,0) failed\n"); return -1; }
    cl_platform_id platform_id = all_platform_id[platform_id_inuse];

    // get device
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices) != CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };
	printf("num_devices = %d\n", num_devices);
    if(device_id_inuse >= (int)num_devices) {
        printf("Invalid Device Number\n");
        return -1;
    }
	device_list = new cl_device_id[num_devices];
	//device_list = (cl_device_id *)malloc(sizeof(cl_device_id)*num_devices);
	if( !device_list ) { printf("ERROR: new cl_device_id[] failed\n"); return -1; }
    if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, num_devices, device_list, NULL) != CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };

    // get device type
    if (clGetDeviceInfo(device_list[device_id_inuse], CL_DEVICE_TYPE, sizeof(device_type), (void *)&device_type, NULL)!= CL_SUCCESS) { printf("ERROR: clGetDeviceIDs failed\n"); return -1; };

	selected_device = device_list[device_id_inuse];

	// create OpenCL context
	cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
	context = clCreateContext(ctxprop, 1, &selected_device, NULL, NULL, &result);
	if( !context || result != CL_SUCCESS ) { printf("ERROR: clCreateContext() failed => %d\n", result); return -1; }

	// create command queue for the specific device
#ifdef TIMING
	cmd_queue = clCreateCommandQueue( context, selected_device, CL_QUEUE_PROFILING_ENABLE, NULL );
#else
	cmd_queue = clCreateCommandQueue( context, selected_device, 0, NULL );
#endif
	if( !cmd_queue ) { printf("ERROR: clCreateCommandQueue() failed\n"); return -1; }
	return 0;
}

static int shutdown()
{
	// release resources
	if( cmd_queue ) clReleaseCommandQueue( cmd_queue );
	if( context ) clReleaseContext( context );
	if( device_list ) delete[] device_list;

	// reset all variables
	cmd_queue = 0;
	context = 0;
	device_list = 0;
	num_devices = 0;
	device_type = CL_DEVICE_TYPE_GPU;

	return 0;
}

double gettime() {
  struct timeval t;
  gettimeofday(&t,NULL);
  return t.tv_sec+t.tv_usec*1e-6;
}

unsigned int num_threads = 0;
unsigned int num_blocks = 0;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main( int argc, char** argv)
{
	return setup(argc, argv);
}


int bpnn_train_kernel(BPNN *net, float *eo, float *eh)
{
	int in, hid, out;
	float out_err, hid_err;
	int status = -1;
	int event_idx;
	cl_int err = CL_SUCCESS;
	cl_program prog = 0;
	cl_kernel kernel1 = 0;
	cl_kernel kernel2 = 0;
	cl_mem input_hidden_ocl = 0;
	cl_mem input_ocl = 0;
	cl_mem output_hidden_ocl = 0;
	cl_mem hidden_partial_sum = 0;
	cl_mem hidden_delta_ocl = 0;
	cl_mem input_prev_weights_ocl = 0;
	cl_event event = 0;
	cl_event write_event[3] = {0, 0, 0};
	float *input_weights_one_dim = 0;
    float *input_weights_prev_one_dim = 0;
	float *partial_sum = 0;
	char *source = 0;
	FILE *fp = 0;
	const char *kernel_bp1 = "bpnn_layerforward_ocl";
	const char *kernel_bp2 = "bpnn_adjust_weights_ocl";
	const char *tempchar = "./backprop_kernel.cl";
	const char *slist[2] = {0, 0};
	float sum = 0.0f;
	float num_blocks = 0.0f;
	size_t global_work[3] = {0, 0, 0};
	size_t local_work[3] = {0, 0, 0};
	const int max_launch_groups_y = 65535;
	int num_blocks_int = 0;
	int block_offset = 0;
	int launch_blocks = 0;
	int m = 0;

	in = net->input_n;
	hid = net->hidden_n;
	out = net->output_n;

	int sourcesize = 1024*1024;
	source = (char *)calloc(sourcesize, sizeof(char));
	if(!source) { printf("ERROR: calloc(%d) failed\n", sourcesize); return -1; }

	// read the kernel core source
	fp = fopen(tempchar, "rb");
	if(!fp) { printf("ERROR: unable to open '%s'\n", tempchar); goto cleanup; }
	fread(source + strlen(source), sourcesize, 1, fp);
	fclose(fp);
	fp = 0;

#ifdef  TIMING
    gettimeofday(&tv_total_start, NULL);
#endif
	if(initialize()) goto cleanup;

	// compile kernel
	slist[0] = source;
	prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateProgramWithSource() => %d\n", err); goto cleanup; }
	err = clBuildProgram(prog, 1, &device_list[device_id_inuse], NULL, NULL, NULL);
	{ // show warnings/errors
		//static char log[65536]; memset(log, 0, sizeof(log));
		//cl_device_id device_id = 0;
		//err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
		//clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
		//if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
	}
	if(err != CL_SUCCESS) { printf("ERROR: clBuildProgram() => %d\n", err); goto cleanup; }

	kernel1 = clCreateKernel(prog, kernel_bp1, &err);
	kernel2 = clCreateKernel(prog, kernel_bp2, &err);
	if(err != CL_SUCCESS) { printf("ERROR: clCreateKernel() 0 => %d\n", err); goto cleanup; }
	clReleaseProgram(prog);
	prog = 0;

#ifdef  TIMING
	gettimeofday(&tv_init_end, NULL);
	tvsub(&tv_init_end, &tv_total_start, &tv);
	init_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif


	num_blocks = in / BLOCK_SIZE;
	num_blocks_int = in / BLOCK_SIZE;

	input_weights_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
	input_weights_prev_one_dim = (float *) malloc((in + 1)* (hid + 1) * sizeof(float));
	partial_sum = (float *) malloc(num_blocks_int * WIDTH * sizeof(float));
	if (!input_weights_one_dim || !input_weights_prev_one_dim || !partial_sum) {
		printf("ERROR: host allocation failed\n");
		goto cleanup;
	}

	// set global and local workitems
	global_work[0] = BLOCK_SIZE;
	global_work[1] = (size_t)(BLOCK_SIZE * num_blocks);
	global_work[2] = 1;
	local_work[0] = BLOCK_SIZE;
	local_work[1] = BLOCK_SIZE;
	local_work[2] = 1;

	// this preprocessing stage is temporarily added to correct the bug of wrong memcopy using two-dimensional net->inputweights
	// todo: fix mem allocation
	m = 0;
	for (int k = 0; k <= in; k++) {
		for (int j = 0; j <= hid; j++) {
		input_weights_one_dim[m] = net->input_weights[k][j];
		input_weights_prev_one_dim[m] = net-> input_prev_weights[k][j];
	    m++;
			}
	}

#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_start, NULL);
#endif
	input_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (in + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_ocl\n"); goto cleanup; }
	input_hidden_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (in + 1) * (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_hidden_ocl\n"); goto cleanup; }
	output_hidden_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer output_hidden_ocl\n"); goto cleanup; }
	hidden_partial_sum = clCreateBuffer(context, CL_MEM_READ_WRITE, num_blocks_int * WIDTH * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer hidden_partial_sum\n"); goto cleanup; }
	hidden_delta_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer hidden_delta_ocl\n"); goto cleanup; }
	input_prev_weights_ocl = clCreateBuffer(context, CL_MEM_READ_WRITE, (in + 1) * (hid + 1) * sizeof(float), NULL, &err );
	if(err != CL_SUCCESS) { printf("ERROR: clCreateBuffer input_prev_weights_ocl\n"); goto cleanup; }
#ifdef  TIMING
    gettimeofday(&tv_mem_alloc_end, NULL);
    tvsub(&tv_mem_alloc_end, &tv_mem_alloc_start, &tv);
    mem_alloc_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
#endif

	printf("Performing %s computation\n", device_type == CL_DEVICE_TYPE_GPU ? "GPU" : "CPU");

	//write buffers
	err = clEnqueueWriteBuffer(cmd_queue, input_ocl, 1, 0, (in + 1) * sizeof(float), net->input_units, 0, 0, &write_event[0]);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_ocl\n"); goto cleanup; }

	err = clEnqueueWriteBuffer(cmd_queue, input_hidden_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, &write_event[1]);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_hidden_ocl\n"); goto cleanup; }
#ifdef TIMING
    h2d_time += probe_event_time(write_event[0],cmd_queue);
    h2d_time += probe_event_time(write_event[1],cmd_queue);
#endif
    clReleaseEvent(write_event[0]);
    write_event[0] = 0;
    clReleaseEvent(write_event[1]);
    write_event[1] = 0;

	clSetKernelArg(kernel1, 0, sizeof(void *), (void*) &input_ocl);
	clSetKernelArg(kernel1, 1, sizeof(void *), (void*) &output_hidden_ocl);
	clSetKernelArg(kernel1, 2, sizeof(void *), (void*) &input_hidden_ocl);
	clSetKernelArg(kernel1, 3, sizeof(void *), (void*) &hidden_partial_sum );
	clSetKernelArg(kernel1, 4, sizeof(float) *  HEIGHT, (void*)NULL );
	clSetKernelArg(kernel1, 5, sizeof(float ) *  HEIGHT * WIDTH, (void*)NULL );
	clSetKernelArg(kernel1, 6, sizeof(cl_int), (void*) &in);
	clSetKernelArg(kernel1, 7, sizeof(cl_int), (void*) &hid);
	for (block_offset = 0; block_offset < num_blocks_int; block_offset += max_launch_groups_y) {
		launch_blocks = num_blocks_int - block_offset;
		if (launch_blocks > max_launch_groups_y) {
			launch_blocks = max_launch_groups_y;
		}
		global_work[1] = (size_t)(BLOCK_SIZE * launch_blocks);
		clSetKernelArg(kernel1, 8, sizeof(cl_int), (void*) &block_offset);
		err = clEnqueueNDRangeKernel(cmd_queue, kernel1, 2, NULL, global_work, local_work, 0, 0, &event);
		if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); goto cleanup; }
#ifdef TIMING
		kernel_time += probe_event_time(event,cmd_queue);
#endif
		clReleaseEvent(event);
		event = 0;
	}

	err = clEnqueueReadBuffer(cmd_queue, hidden_partial_sum, 1, 0, num_blocks_int * WIDTH * sizeof(float), partial_sum, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueReadBuffer: partial sum\n"); goto cleanup; }
#ifdef TIMING
    d2h_time += probe_event_time(event,cmd_queue);
#endif
    clReleaseEvent(event);
    event = 0;

	for (int j = 1; j <= hid; j++) {
		sum = 0.0;
		for (int k = 0; k < num_blocks_int; k++) {
			sum += partial_sum[k * hid + j-1] ;
	    }
		sum += net->input_weights[0][j];
		net-> hidden_units[j] = float(1.0 / (1.0 + exp(-sum)));
	}


	bpnn_layerforward(net->hidden_units, net->output_units, net->hidden_weights, hid, out);
	bpnn_output_error(net->output_delta, net->target, net->output_units, out, &out_err);
	bpnn_hidden_error(net->hidden_delta, hid, net->output_delta, out, net->hidden_weights, net->hidden_units, &hid_err);
	bpnn_adjust_weights(net->output_delta, out, net->hidden_units, hid, net->hidden_weights, net->hidden_prev_weights);

	err = clEnqueueWriteBuffer(cmd_queue, hidden_delta_ocl,       1, 0, (hid + 1) * sizeof(float), net->hidden_delta, 0, 0, &write_event[0]);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer hidden_delta_ocl\n"); goto cleanup; }

	err = clEnqueueWriteBuffer(cmd_queue, input_prev_weights_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_prev_one_dim, 0, 0, &write_event[1]);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_prev_weights_ocl\n"); goto cleanup; }

	err = clEnqueueWriteBuffer(cmd_queue, input_hidden_ocl,       1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, &write_event[2]);
	if(err != CL_SUCCESS) { printf("ERROR: clEnqueueWriteBuffer input_hidden_ocl\n"); goto cleanup; }
#ifdef TIMING
    h2d_time += probe_event_time(write_event[0],cmd_queue);
    h2d_time += probe_event_time(write_event[1],cmd_queue);
    h2d_time += probe_event_time(write_event[2],cmd_queue);
#endif
    clReleaseEvent(write_event[0]);
    write_event[0] = 0;
    clReleaseEvent(write_event[1]);
    write_event[1] = 0;
    clReleaseEvent(write_event[2]);
    write_event[2] = 0;

	clSetKernelArg(kernel2, 0, sizeof(void *), (void*) &hidden_delta_ocl);
	clSetKernelArg(kernel2, 1, sizeof(cl_int), (void*) &hid);
	clSetKernelArg(kernel2, 2, sizeof(void *), (void*) &input_ocl);
	clSetKernelArg(kernel2, 3, sizeof(cl_int), (void*) &in);
	clSetKernelArg(kernel2, 4, sizeof(void *), (void*) &input_hidden_ocl);
	clSetKernelArg(kernel2, 5, sizeof(void *), (void*) &input_prev_weights_ocl );
	for (block_offset = 0; block_offset < num_blocks_int; block_offset += max_launch_groups_y) {
		launch_blocks = num_blocks_int - block_offset;
		if (launch_blocks > max_launch_groups_y) {
			launch_blocks = max_launch_groups_y;
		}
		global_work[1] = (size_t)(BLOCK_SIZE * launch_blocks);
		clSetKernelArg(kernel2, 6, sizeof(cl_int), (void*) &block_offset);
		err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 2, NULL, global_work, local_work, 0, 0, &event);
		if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueNDRangeKernel()=>%d failed\n", err); goto cleanup; }
#ifdef TIMING
		kernel_time += probe_event_time(event,cmd_queue);
#endif
		clReleaseEvent(event);
		event = 0;
	}

	err = clEnqueueReadBuffer(cmd_queue, input_hidden_ocl, 1, 0, (in + 1) * (hid + 1) * sizeof(float), input_weights_one_dim, 0, 0, &event);
	if(err != CL_SUCCESS) { printf("ERROR: 1  clEnqueueReadBuffer: input_hidden_ocl\n"); goto cleanup; }
#ifdef TIMING
    d2h_time += probe_event_time(event,cmd_queue);
#endif
    clReleaseEvent(event);
    event = 0;

	backprop_last_input_hidden_hash = hash_bytes(
		(const unsigned char *)input_weights_one_dim,
		(size_t)(in + 1) * (hid + 1) * sizeof(float));
	*eo = out_err;
	*eh = hid_err;
	status = 0;

cleanup:
#ifdef  TIMING
	gettimeofday(&tv_close_start, NULL);
#endif

	if (event) clReleaseEvent(event);
	for (event_idx = 0; event_idx < 3; ++event_idx) {
		if (write_event[event_idx]) clReleaseEvent(write_event[event_idx]);
	}
	if (fp) fclose(fp);
	if (input_prev_weights_ocl) clReleaseMemObject(input_prev_weights_ocl);
	if (hidden_delta_ocl) clReleaseMemObject(hidden_delta_ocl);
	if (hidden_partial_sum) clReleaseMemObject(hidden_partial_sum);
	if (input_hidden_ocl) clReleaseMemObject(input_hidden_ocl);
	if (output_hidden_ocl) clReleaseMemObject(output_hidden_ocl);
	if (input_ocl) clReleaseMemObject(input_ocl);
	if (kernel2) clReleaseKernel(kernel2);
	if (kernel1) clReleaseKernel(kernel1);
	if (prog) clReleaseProgram(prog);

	free(input_weights_prev_one_dim);
	free(partial_sum);
	free(input_weights_one_dim);
	free(source);

    shutdown();

#ifdef  TIMING
	gettimeofday(&tv_close_end, NULL);
	tvsub(&tv_close_end, &tv_close_start, &tv);
	close_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;
	tvsub(&tv_close_end, &tv_total_start, &tv);
	total_time = tv.tv_sec * 1000.0 + (float) tv.tv_usec / 1000.0;

	printf("Init: %f\n", init_time);
	printf("MemAlloc: %f\n", mem_alloc_time);
	printf("HtoD: %f\n", h2d_time);
	printf("Exec: %f\n", kernel_time);
	printf("DtoH: %f\n", d2h_time);
	printf("Close: %f\n", close_time);
	printf("Total: %f\n", total_time);
#endif
	return status;
}
