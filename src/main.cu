// -----------------------------------------------------------------------------
// * Name:       main.cu
// * Purpose:    Main CUDA program for SIFT algorithm implementation on GPU
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#include <iostream>
#include <string>
#include <stdlib.h>
#include "SIFT_CUDA.hxx"


const char* filename = "../img/chartreuse.jpg";

int main(int argc, char **argv) {

    Image img = Image(filename);

    //Buil Gaussian pyramid from base image
    SIFT_CUDA sift;
    sift.BuildGaussianPyramid(img);

    std::cout <<"pyramid numOctaves: " << sift.gPyramid.numOctaves() << " numImagesPerOctave: " <<  sift.gPyramid.numImagesPerOctave() << " num elements: " << sift.gPyramid.octaves.size() << std::endl;
    sift.gPyramid.WriteAllImagesToFile();

    //TODO free pyramid memory
    exit(0);
}
