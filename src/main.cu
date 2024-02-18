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

__device__ bool CheckForLocalMaxInNeighbourScales(unsigned char *imgScale1, unsigned char *imgScale2, unsigned char *imgScale3, unsigned char curPxVal, int inputWidth, int inputHeight, int x, int y)
{
  int inputChannels = 1;
  bool is_min = true, is_max = true;
  unsigned char neighbor = 0;

  for (int dx : {-1,0,1}) {
      for (int dy : {-1,0,1}) {
          unsigned char neighbor = getPixelColour(x, y, inputWidth, inputHeight, inputChannels, RED, imgScale1);

          if (neighbor > curPxVal) is_max = false;
          if (neighbor < curPxVal) is_min = false;

          //neighbor = img2.getPixelValue(x+dx, y+dy, RED);
          //if (neighbor > curPxVal) is_max = false;
         // if (neighbor < curPxVal) is_min = false;

          //neighbor = img.get_pixel(x+dx, y+dy, 0);
          //neighbor = img3.getPixelValue(x+dx, y+dy, RED);
          // std::cout << "curPxVal: " <<  curPxVal<< " neighbor: " << neighbor << std::endl; 
          //if (neighbor > curPxVal) is_max = false;
         // if (neighbor < curPxVal) is_min = false;

          if (!is_min && !is_max) return false;
      }
  }
  return true;
}

__global__ void FINDMAX_CUDA(int inputWidth, int inputHeight, int inputChannels, unsigned char *cudaDeviceInputData, unsigned char *cudaDeviceResult)
{
  float contrast_threshold = 0.012;
  int max = 0, min = 0;

  unsigned char *imgScale1 = cudaDeviceInputData;

  for (int x = 0; x < inputWidth; x++) {
      for (int y = 0; y < inputHeight; y++) {

          unsigned char curPxVal = getPixelColour(x, y, inputWidth, inputHeight, inputChannels, RED, imgScale1);
          //printf("curPxVal: %d\n", curPxVal);

      if (curPxVal < (255*0.7)) {
      
          if (CheckForLocalMaxInNeighbourScales(imgScale1, imgScale1, imgScale1, curPxVal, inputWidth, inputHeight, x,y)) {
              max++;
          }
          else{
              min++;
          }
      }//if thresh
      }
  }
  printf("max: %d min: %d\n", max, min);
}


/* Submitted version
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
  } */

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
    
    Image currImg = sift.gPyramid.octaves[0][0]; //note g for now, not DoG
    currImg.ConvertToGrayscale();

    Image prevImg = sift.gPyramid.octaves[0][1]; 
    prevImg.ConvertToGrayscale();

    Image nextImg = sift.gPyramid.octaves[0][2]; //TODO use actual prev/next. No prev for first run
    nextImg.ConvertToGrayscale();

    sift.FindLocalMaxima(currImg, prevImg, nextImg);


/* disable gpu for now, measure perf on host 

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
    //FINDMAX_CUDA<<<numBlocks, threadsPerBlock>>>(dogImg.width(), dogImg.height(), dogImg.numChannels(), cudaDeviceInputData, cudaDeviceResult);

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
    */

    sift.FreePyramidMemory();
    exit(0);
}
