//========================================================================================================================================================================================================200
//	DEFINE/INCLUDE
//========================================================================================================================================================================================================200

//======================================================================================================================================================150
//	DEFINE
//======================================================================================================================================================150

// double precision support (switch between as needed for NVIDIA/AMD)
#ifdef AMDAPP
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

// clBuildProgram compiler cannot link this file for some reason, so had to redefine constants and structures below
// #include ../common.h						// (in directory specified to compiler)			main function header

//======================================================================================================================================================150
//	DEFINE (had to bring from ../common.h here because feature of including headers in clBuildProgram does not work for some reason)
//======================================================================================================================================================150

// change to double if double precision needed
#define fp float

//#define DEFAULT_ORDER_2 256

//======================================================================================================================================================150
//	STRUCTURES (had to bring from ../common.h here because feature of including headers in clBuildProgram does not work for some reason)
//======================================================================================================================================================150

// ???
typedef struct knode {
	int location;
	int indices [DEFAULT_ORDER_2 + 1];
	int  keys [DEFAULT_ORDER_2 + 1];
	bool is_leaf;
	int num_keys;
} knode; 

//========================================================================================================================================================================================================200
//	findRangeK function
//========================================================================================================================================================================================================200

__kernel void 
findRangeK(	long height,
			__global knode *knodesD,
			long knodes_elem,

			__global long *currKnodeD,
			__global long *offsetD,
			__global long *lastKnodeD,
			__global long *offset_2D,
			__global int *startD,
			__global int *endD,
			__global int *RecstartD, 
			__global int *ReclenD)
{

	// private thread IDs
	int thid = get_local_id(0);
	int bid = get_group_id(0);
	__local long currKnodeL;
	__local long offsetL;
	__local long lastKnodeL;
	__local long offset2L;

	if (thid == 0) {
		currKnodeL = currKnodeD[bid];
		offsetL = offsetD[bid];
		lastKnodeL = lastKnodeD[bid];
		offset2L = offset_2D[bid];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// ???
	int i;
	for(i = 0; i < height; i++){
		long curr = currKnodeL;
		long last = lastKnodeL;

		if((knodesD[curr].keys[thid] <= startD[bid]) && (knodesD[curr].keys[thid+1] > startD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[curr].indices[thid] < knodes_elem){
				offsetL = knodesD[curr].indices[thid];
			}
		}
		if((knodesD[last].keys[thid] <= endD[bid]) && (knodesD[last].keys[thid+1] > endD[bid])){
			// this conditional statement is inserted to avoid crush due to but in original code
			// "offset_2[bid]" calculated below that later addresses part of knodes goes outside of its bounds cause segmentation fault
			// more specifically, values saved into knodes->indices in the main function are out of bounds of knodes that they address
			if(knodesD[last].indices[thid] < knodes_elem){
				offset2L = knodesD[last].indices[thid];
			}
		}
		barrier(CLK_LOCAL_MEM_FENCE);
		// set for next tree level
		if(thid==0){
			currKnodeL = offsetL;
			lastKnodeL = offset2L;
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (thid == 0) {
		currKnodeD[bid] = currKnodeL;
		offsetD[bid] = offsetL;
		lastKnodeD[bid] = lastKnodeL;
		offset_2D[bid] = offset2L;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Find the index of the starting record
	if(knodesD[currKnodeL].keys[thid] == startD[bid]){
		RecstartD[bid] = knodesD[currKnodeL].indices[thid];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Find the index of the ending record
	if(knodesD[lastKnodeL].keys[thid] == endD[bid]){
		ReclenD[bid] = knodesD[lastKnodeL].indices[thid] - RecstartD[bid]+1;
	}

}

//========================================================================================================================================================================================================200
//	End
//========================================================================================================================================================================================================200
