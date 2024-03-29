// -----------------------------------------------------------------------------
// * Name:       SIFT_CUDA.hxx
// * Purpose:    Header file for SIFT_CUDA class and Image class
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#ifndef SIFT_CUDA_H
#define SIFT_CUDA_H

#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

using namespace std;

enum pxChannel{
        RED = 0,
        GREEN,
        BLUE
    };

enum InterpType{
        BILINEAR = 0, 
        NEAREST
    };

class Image {
    private:
        int _width;
        int _height;
        int _numChannels;
        int _size;
        unsigned char *_data;

    public:
        //Create image from input data. Copies data to own memory
        Image(int width, int height, int numChannels, unsigned char *data){

            this->_width = width;
            this->_height = height;
            this->_numChannels = numChannels;
            this->_size = (width*height*numChannels);
            this->_data =  new unsigned char[_size];
            std::memcpy(this->_data,data,_size);
        };

        //Create blank image
        Image(int width, int height, int numChannels){
            this->_width = width;
            this->_height = height;
            this->_numChannels = numChannels;
            this->_size = (width*height*numChannels);
            this->_data =  new unsigned char[_size];
            std::memset(this->_data,0,this->_size);
        };

        Image(){};
        ~Image(){};

        Image(std::string filename);

        void FreeImageData()
        {if(this->_data != NULL) {delete[] this->_data;}}

        int width() const {return _width;} //read-only
        int height() const {return _height;}
        int numChannels() const {return _numChannels;}
        int size() const {return _size;}
        unsigned char* data() {return _data;} //read/write

        unsigned char getPixelValue(int x, int y, pxChannel channel);
        void setPixelValue(int x, int y, pxChannel channel, unsigned char value);

        void setPixelValueOnData(int x, int y, pxChannel channel, int width, int height, int numChannels, unsigned char value, unsigned char* data)
        {
            if (x < 0 || x >= width || y < 0 ||  y >= height || channel > numChannels){
                printf("Error setPixelValueOnData: out of bounds x:%d y:%d chans:%d colour%d\n", x,y, numChannels, (int) channel);
                return;
            }

            else{
                *(data + ((x + y*(width)) * (numChannels)) + (int) channel) = value;
            }
        }

        void ConvertToGrayscale(void)
        {
            if(this->_numChannels != 3)
            {
                std::cout << "Grayscale convert error: numChans less than 3\n" << std::endl;
            }

            for (int x = 0; x < this->_width; x++) {
                for (int y = 0; y < this->_height; y++) {
                    float pxRed, pxGreen, pxBlue;
                    pxRed = this->getPixelValue(x, y, RED);
                    pxGreen = this->getPixelValue(x, y, GREEN);
                    pxBlue = this->getPixelValue(x, y, BLUE);
                    this->_numChannels = 1;
                    this->setPixelValue(x, y, RED, 0.299*pxRed + 0.587*pxGreen + 0.114*pxBlue);
                }
            }
        }

        //Map coordinate from 0-current_max range to 0-new_max range
        float MapCoordinate(float new_max, float current_max, float coord)
        {
            float a = new_max / current_max;
            float b = -0.5 + a*0.5;
            return a*coord + b;
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

        float InterpolateNearestNeighbour(Image& img, float x, float y, pxChannel c)
        {
            return img.getPixelValue(std::round(x), std::round(y), c);
        }

        void Resize(int new_w, int new_h, InterpType interp)
        {
            unsigned char* resized_data = new unsigned char[new_w*new_h*(this->_numChannels)];
            if(resized_data == NULL ) {
                std::cout << "Unable to create memory for resized image\n" << std::endl;
            }

            float value = 0;
            for (int x = 0; x < new_w; x++) {
                for (int y = 0; y < new_h; y++) {
                    for (int c = 0; c < this->_numChannels; c++) {
                        float old_x = MapCoordinate(this->_width, new_w, x);
                        float old_y = MapCoordinate(this->_height, new_h, y);
                        if(interp==InterpType::BILINEAR){
                            value = InterpolateBilinear(*this, old_x, old_y, (pxChannel) c);
                        }
                        else if(interp==InterpType::NEAREST) {
                            value = InterpolateNearestNeighbour(*this, old_x, old_y, (pxChannel) c);
                        }
                        setPixelValueOnData(x, y, (pxChannel) c, new_w, new_h, this->_numChannels, value, resized_data); 
                    }
                }
            }

            //Update sizes
            this->_width = new_w;
            this->_height = new_h;
            this->_size = (new_w*new_h*(this->_numChannels));

            //Reassign data
            delete[] this->_data;
            this->_data = resized_data;
        }
};

class Keypoint {
    private:
        int _posX; //location in image
        int _posY;
        int _octaveIdx; //octave/scale where found
        int _scaleIdx;

    public:
        Keypoint(){};
        Keypoint(int posX, int posY, int octIdx, int scaleIdx){
            this->_posX = posX;
            this->_posY = posY;
            this->_octaveIdx = octIdx;
            this->_scaleIdx = scaleIdx;
        };

        int posX() const {return _posX;}
        int posY() const {return _posY;}
        int octaveIdx() const {return _octaveIdx;}
        int scaleIdx() const {return _scaleIdx;}
};

class GaussianPyramid {

    public:
        int _numOctaves = 8;
        int _numScalesPerOctave = 6;
        std::vector<std::vector<Image>> octaves;
        
        GaussianPyramid(){};
        void WriteAllImagesToFile(void);
        int numOctaves() const {return _numOctaves;}
        int numScalesPerOctave() const {return _numScalesPerOctave;}
};

class SIFT_CUDA {
    private:
        static const int kernelSize = 9;
        float kernel[kernelSize] = {0.0f};
    
    public:
        SIFT_CUDA(){};
        GaussianPyramid gPyramid;
        GaussianPyramid dogPyramid;
        std::vector<Keypoint> keypoints;

        void CreateGaussianKernel(float sigma);
        void ApplyGaussianBlur(Image img);
        void BuildGaussianPyramid(Image baseImage);
        Image ComputeDoG(Image img1, Image img2);
        void BuildDoGPyramid(GaussianPyramid gPyramid);
        bool CheckForLocalMaxInNeighbourScales(Image img1, Image img2, Image img3, unsigned char curPxVal, int x, int y);
        int FindLocalMaxima(Image img1, Image img2, Image img3);
        void FindKeypoints(void);
        int FindKeypointsInImage(int imgOctave, int imgScale);
        void WriteImageToFile(Image img);
        void FreePyramidMemory(void);
};

#endif
