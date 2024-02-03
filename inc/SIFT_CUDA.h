// -----------------------------------------------------------------------------
// * Name:       SIFT_CUDA.h
// * Purpose:    Header file for SIFT_CUDA class 
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#ifndef SIFT_CUDA_H
#define SIFT_CUDA_H

#include <iostream>
#include <cmath>

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

        //Map coordinate from 0-current_max range to 0-new_max range
        float MapCoordinate(float new_max, float current_max, float coord)
        {
            float a = new_max / current_max;
            float b = -0.5 + a*0.5;
            return a*coord + b;
        }

        Image Resize(int new_w, int new_h)
        {
            Image resized(new_w, new_h, this->_numChannels);
            float value = 0;
            for (int x = 0; x < new_w; x++) {
                for (int y = 0; y < new_h; y++) {
                    for (int c = 0; c < resized.numChannels(); c++) {
                        float old_x = MapCoordinate(this->_width, new_w, x);
                        float old_y = MapCoordinate(this->_height, new_h, y);
                        value = InterpolateBilinear(*this, old_x, old_y, (pxChannel) c);
                        resized.setPixelValue(x, y, (pxChannel) c, value); 
                    }
                }
            }
            return resized;
        }

        float InterpolateBilinear(Image& img, float x, float y, pxChannel c)
        {
            float p1, p2, p3, p4, q1, q2;
            float floor_x = std::floor(x), floor_y = std::floor(y);
            float x_ceil = floor_x + 1, y_ceil = floor_y + 1;

            p1 = img.getPixelValue(floor_x, floor_y, c);
            p2 = img.getPixelValue(x_ceil, floor_y, c);
            p3 = img.getPixelValue(floor_x, y_ceil, c);
            p4 = img.getPixelValue(x_ceil, y_ceil, c);
            q1 = (y_ceil-y)*p1 + (y-floor_y)*p3;
            q2 = (y_ceil-y)*p2 + (y-floor_y)*p4;

            return (x_ceil-x)*q1 + (x-floor_x)*q2;
        }
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
