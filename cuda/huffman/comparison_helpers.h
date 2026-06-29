#ifndef _COMPARISON_HELPERS_H_
#define _COMPARISON_HELPERS_H_

#include "../../common/rodinia_verify.h"

template <typename T>
__inline int compare_vectors(T* data1, T* data2, unsigned int size) {
	printf("Comparing vectors: \n");
	bool match = true;
	for(unsigned int i = 0; i < size; i++)  
		if (data1[i]!= data2[i]) {
			match = false;
			printf("Diff: data1[%d]=%d,  data1[%d]=%d.\n",i,data1[i],i,data2[i]);
		}

	if (match) { return rodinia_print_pass("Huffman CPU reference verification");	}
	else { rodinia_print_fail("Huffman CPU reference verification"); return -1; }
}

#endif
