// -----------------------------------------------------------------------------
// * Name:       SIFT_CUDA.h
// * Purpose:    Header file for SIFT_CUDA class 
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#ifndef SIFT_CUDA_H
#define SIFT_CUDA_H

#include <iostream>

using namespace std;

enum pxChannel{
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
        Image(int width, int height, int numChannels){
            std::cout << "const blank\n" << std::endl;
            this->_width = width;
            this->_height = height;
            this->_numChannels = numChannels;
            this->_size = (width*height*numChannels);
            this->_data =  new unsigned char[_size];
        };

        Image(int width, int height, int numChannels, unsigned char *data){
            this->_width = width;
            this->_height = height;
            this->_numChannels = numChannels;
            this->_size = (width*height*numChannels);
            this->_data = data;
        };

        ~Image(){std::cout << "destructor\n" << std::endl;};

        void FreeImageData(){delete[] this->_data;}
        int width() const {return _width;} //read-only
        int height() const {return _height;}
        int numChannels() const {return _numChannels;}
        int size() const {return _size;}
        unsigned char* data() {return _data;} //read/write

        unsigned char getPixelValue(int x, int y, pxChannel channel);
        void setPixelValue(int x, int y, pxChannel channel, unsigned char value);
};

class SIFT_CUDA {
    private:
        static const int kernelSize = 9;
        float kernel[kernelSize] = {0.0f};
    
    public:
        SIFT_CUDA(){};
        void CreateGaussianKernel(float sigma);
        void ApplyGaussianBlur(Image img);
};

#endif
