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


#define NUM_THREADS_1D 16 //Number of threads in 1 dimension of thread block
const char* filename = "../img/landscape512.jpg";


__device__ unsigned char getPixelColour(int x, int y, int width, int height, int numChannels, pxChannel colour, unsigned char *data)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || numChannels!=3 || colour < 0 || colour > 3){
    return 0;
  }

  else{
    return *(data + ((x + y*width) * numChannels) + (int) colour);
  }
}

__device__ void setPixelColour(int x, int y, int width, int height, int numChannels, pxChannel colour, unsigned char *data, unsigned char value)
{
  if (x < 0 || x >= width || y < 0 ||  y >= height || numChannels!=3 || colour < 0 || colour > 3 ){
  }

  else{
    *(data + ((x + y*width) * numChannels) + (int) colour) = value;
  }
}

  __global__ void FINDMAX_CUDA(int inputWidth, int inputHeight, int inputChannels, unsigned char *cudaDeviceInputData, unsigned char *cudaDeviceResult)
  {
    int colIdx = blockDim.x * blockIdx.x + threadIdx.x;
    int rowIdx = blockDim.y * blockIdx.y + threadIdx.y;
    int numKeypoints = 0;

    //TODO load part of image for each block into shared memory from global memory

    unsigned char px00=0, px01=0, px02=0, px10=0, px12=0, px20=0, px21=0, px22=0; //neighbour pixels
    int isMax = 1;

    //Assign centre pixel and check neighbours for local maximum. Read 1 channel only
    unsigned char curPx =  getPixelColour(colIdx, rowIdx, inputWidth, inputHeight, inputChannels, RED, cudaDeviceInputData);

    //pxXY
    px00 = getPixelColour(colIdx-1, rowIdx-1, inputWidth, inputHeight, inputChannels, RED, cudaDeviceInputData);
    px01 = getPixelColour(colIdx-1, rowIdx, inputWidth, inputHeight, inputChannels, RED, cudaDeviceInputData);
    px02 = getPixelColour(colIdx-1, rowIdx+1, inputWidth, inputHeight, inputChannels, RED, cudaDeviceInputData);

    px10 = getPixelColour(colIdx, rowIdx-1, inputWidth, inputHeight, inputChannels, RED, cudaDeviceInputData);
    //px11 = curPx
    px12 = getPixelColour(colIdx, rowIdx+1, inputWidth, inputHeight, inputChannels, RED, cudaDeviceInputData);

    px20 = getPixelColour(colIdx+1, rowIdx-1, inputWidth, inputHeight, inputChannels, RED, cudaDeviceInputData);
    px21 = getPixelColour(colIdx+1, rowIdx, inputWidth, inputHeight, inputChannels, RED, cudaDeviceInputData);
    px22 = getPixelColour(colIdx+1, rowIdx+1, inputWidth, inputHeight, inputChannels, RED, cudaDeviceInputData);
    
    if((px00>curPx) || (px01>curPx) || (px02>curPx) || (px10>curPx) || (px12>curPx) || (px20>curPx) || (px21>curPx) || (px22>curPx)){
      isMax = 0;
    }

    else{
        //Update keypoint count in global memory
        cudaDeviceResult[(rowIdx * blockDim.x) + colIdx] =  1;
      __syncthreads();
    }
  }

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
        std::cout << "maxGridSize x:" << deviceProp.maxGridSize[0] << " y:" <<  deviceProp.maxGridSize[1] << " maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << std::endl << std::endl;
    }
    
    //Build Gaussian pyramid from base image
    SIFT_CUDA sift;
    Image img = Image(filename);
    sift.BuildGaussianPyramid(img);
    sift.BuildDoGPyramid(sift.gPyramid);

    //Take base image from first octave
    //Currently we only check a single image,
    //next step is to loop through for each scale
    Image dogImg = sift.dogPyramid.octaves[0][0];

    //Allocate device memory for input image
    unsigned char *cudaDeviceInputData;
    cudaMalloc((void **)&cudaDeviceInputData, dogImg.size());

    //Allocate and init device memory for result
    unsigned char *cudaDeviceResult;
    cudaMalloc((void **)&cudaDeviceResult, dogImg.size() );
    cudaMemset(cudaDeviceResult, 0, dogImg.size());
    
    //Allocate host memory for result (num keypoints)
    unsigned char *hostResultData = new unsigned char[dogImg.size()];

    //Timer
    cudaEvent_t start;
    cudaEvent_t stop;
    float msecTotal;

    cudaEventCreate(&start);
    cudaEventRecord(start, NULL);

    //Copy image from host to device
    cudaMemcpy(cudaDeviceInputData, dogImg.data(), dogImg.size(), cudaMemcpyHostToDevice);

    //Set up kernel
    dim3 threadsPerBlock(NUM_THREADS_1D, NUM_THREADS_1D);
    dim3 numBlocks(ceil(dogImg.width() / threadsPerBlock.x), ceil(dogImg.height() / threadsPerBlock.y));
    printf("numBlocks.x / y : %d total threadsPerBlock: %d\n", numBlocks.x, threadsPerBlock.x * threadsPerBlock.x);

    //Run kernel
    std::cout << "Run kernel" << std::endl;
    FINDMAX_CUDA<<<numBlocks, threadsPerBlock>>>(dogImg.width(), dogImg.height(), dogImg.numChannels(), cudaDeviceInputData, cudaDeviceResult);

    //Copy back to host 
    std::cout << "Done. Copy result to host" << std::endl << std::endl;
    cudaMemcpy(hostResultData, cudaDeviceResult, dogImg.size(), cudaMemcpyDeviceToHost);

    //Stop timer
    cudaEventCreate(&stop);
    cudaEventRecord(stop, NULL);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&msecTotal, start, stop);

    //Check num keypoints in results
    int total = 0;
    for (int i = 0; i < dogImg.size(); i++){
      total += hostResultData[i];
    }

    printf("Total kps: %d\n", total);
    std::cout << "Processing time: " << msecTotal << " (ms)" << std::endl;

    sift.FreePyramidMemory();
    exit(0);
}
