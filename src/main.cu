// -----------------------------------------------------------------------------
// * Name:       main.cu
// * Purpose:    Main CUDA program for SIFT algorithm implementation on GPU
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#include <cmath>
#include <iostream>
#include <string>
#include <stdlib.h>
#include "SIFT_CUDA.h"


int main(int argc, char **argv) {

    SIFT_CUDA sift;

    sift.display();

    return 0;
}
