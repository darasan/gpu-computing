// -----------------------------------------------------------------------------
// * Name:       SIFT_CUDA.h
// * Purpose:    Header file for SIFT_CUDA class 
// * History:    Daire O'Neill, December 2023
// -----------------------------------------------------------------------------

#ifndef SIFT_CUDA_H
#define SIFT_CUDA_H

#include <iostream>
using namespace std;
 
class SIFT_CUDA {
    private:
        int private_variable;
    
    protected:
        int protected_variable;
    
    public:
        SIFT_CUDA();
        void display();
};

#endif
