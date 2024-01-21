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

class Image {
    private:
        int _width;
        int _height;
        int _numChannels;
        int _size;
        unsigned char *_data;

    public:
        Image(int width, int height, int numChannels, unsigned char *data){
            this->_width = width;
            this->_height = height;
            this->_numChannels = numChannels;
            this->_size = (width*height*numChannels);
            this->_data = data;
        };

        int width() const {return _width;} //read-only
        int height() const {return _height;}
        int numChannels() const {return _numChannels;}
        int size() const {return _size;}
        unsigned char* data() {return _data;} //read/write
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
        void ApplyGaussianBlur(unsigned char *inputData, int inputWidth, int inputHeight, int inputChannels);
};

#endif
