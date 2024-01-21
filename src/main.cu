// -----------------------------------------------------------------------------
// * Name:       main.cu
// * Purpose:    Main CUDA program for SIFT algorithm implementation on GPU
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#include <iostream>
#include <string>
#include <stdlib.h>
#include "SIFT_CUDA.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h" 

const char* filename = "../img/chartreuse.jpg";

int main(int argc, char **argv) {

    //Load image
    int inputWidth, inputHeight, inputChannels;
    unsigned char *inputData = stbi_load(filename, &inputWidth, &inputHeight, &inputChannels, 0);

    if(inputData == NULL){
        std::cout << "Error loading image\n" << std::endl;
        exit(1);
    }

    //Create Image object
    Image img = Image(inputWidth, inputHeight, inputChannels, inputData);

    SIFT_CUDA sift;
    sift.CreateGaussianKernel(4.0f);
    sift.ApplyGaussianBlur(img.data(), img.width(), img.height(), img.numChannels());

    //Set params for output file
    int outputWidth = img.width();
    int outputHeight = img.height();
    int outputChannels = img.numChannels();
    int outputImageSize = img.size();

    unsigned char *outputData = new unsigned char[outputImageSize];
    if(outputData == NULL) {
        std::cout << "Unable to allocate memory for output image\n" << std::endl;
        exit(1);
    }

    //Write result to file
    std::cout <<"Write to file\n" << std::endl;
    int success = stbi_write_jpg("output.jpg", outputWidth, outputHeight, outputChannels, inputData, 100);
    if(success){
        std::cout <<"Wrote file OK! w:" << outputWidth << " h: " << outputHeight << " chans: " << outputChannels << std::endl;
    }

    else{
        std::cout <<"Error writing output file\n" << std::endl;
    }

    stbi_image_free(inputData);
    delete(outputData);

    exit(0);
}
