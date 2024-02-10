// -----------------------------------------------------------------------------
// * Name:       main.cu
// * Purpose:    Main CUDA program for SIFT algorithm implementation on GPU
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#include <iostream>
#include <string>
#include <stdlib.h>
#include <cuda.h>
#include "SIFT_CUDA.hxx"


//const char* filename = "../img/chartreuse.jpg";
const char* filename = "../img/landscape512.jpg";

int main(int argc, char **argv) {

    //Init CUDA device
    cudaError_t error;
    cudaDeviceProp deviceProp;
    int devID = 0;
    error = cudaGetDevice(&devID);
    error = cudaGetDeviceProperties(&deviceProp, devID);

    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties returned error code " << error << "line " << __LINE__  << std::endl;
        exit(0);
    } else {
        std::cout << "GPU Device " << devID << deviceProp.name << " compute capability " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "totalGlobalMem: " << deviceProp.totalGlobalMem << " bytes" << " sharedMemPerBlock: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "maxGridSize x:" << deviceProp.maxGridSize[0] << " y:" <<  deviceProp.maxGridSize[1] << " maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock;
    }

    //Build Gaussian pyramid from base image
    SIFT_CUDA sift;
    Image img = Image(filename);
    sift.BuildGaussianPyramid(img);
    sift.BuildDoGPyramid(sift.gPyramid);
    //sift.gPyramid.WriteAllImagesToFile();

    sift.FreePyramidMemory(); 
    exit(0);
}
