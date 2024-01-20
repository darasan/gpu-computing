// -----------------------------------------------------------------------------
// * Name:       SIFT_CUDA.h
// * Purpose:    Header file for SIFT_CUDA class 
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#ifndef SIFT_CUDA_H
#define SIFT_CUDA_H

#include <iostream>

using namespace std;

 enum pxColour{
        RED = 0,
        GREEN,
        BLUE
    };
 
class SIFT_CUDA {
    private:
        static const int kernelSize = 9;
        float kernel[kernelSize] = {0.0f};
        
        unsigned char getPixelColour(int x, int y, int width, int height, int numChannels, pxColour colour, unsigned char *data);
        void setPixelColour(int x, int y, int width, int height, int numChannels, pxColour colour, unsigned char *data, unsigned char value);
    
    public:
        SIFT_CUDA(){};
        void CreateGaussianKernel(float sigma);
        void ApplyGaussianBlur(unsigned char *inputData, int inputHeight, int inputWidth, int inputChannels);
};

#endif
