//C++
#include <time.h>
#include <iostream>
using namespace std;

//openCV
#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

//CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

//timer
#include "timer.hpp"

//texture
texture<unsigned char>  tex_in_image;

__global__ void convolution_kernel(short* out_image,int *H,int width,int height)
{
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;

    if(x<0 || x>width || y<0 || y>height)
    {
        return;
    }

    int pos = y*width+x;

    if(x==0||y==0||(x==width-1)||(y==height-1))
    {
        out_image[pos] = tex1Dfetch(tex_in_image,pos);
        return;
    }

    int left = pos - 1;
    int right = pos + 1;
    int up = pos - width;
    int down = pos + width;
    int up_left = up - 1;
    int up_right = up + 1;
    int down_left = down - 1;
    int down_right = down + 1;

    out_image[pos] = H[0]*tex1Dfetch(tex_in_image,  up_left) + H[1]*tex1Dfetch(tex_in_image,up     ) + H[2]*tex1Dfetch(tex_in_image,up_right  )
                    +H[3]*tex1Dfetch(tex_in_image,     left) + H[4]*tex1Dfetch(tex_in_image,pos    ) + H[5]*tex1Dfetch(tex_in_image,right     )
                    +H[6]*tex1Dfetch(tex_in_image,down_left) + H[7]*tex1Dfetch(tex_in_image,down   ) + H[8]*tex1Dfetch(tex_in_image,down_right);

}

int main()
{
    //in data
    Mat in_image = imread("test.jpg",CV_LOAD_IMAGE_GRAYSCALE);
    Mat out_image = Mat(in_image.size(),CV_16S); 
    
    //convolution kernel
    int H[9];
    H[0]=-1;H[1]=-1;H[2]=-1;
    H[3]=-1;H[4]= 8;H[5]=-1;
    H[6]=-1;H[7]=-1;H[8]=-1;

    //calc
    Timer start_time;

    //init CUDA
    //error status
    cudaError_t cuda_status;

    //only chose one GPU
    //init 
    cuda_status = cudaSetDevice(0);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaSetDevice failed! Do you have a CUDA-Capable GPU installed?");
        return -1;
    }
    
    //in image and out image
    unsigned char * dev_in_image;
    short * dev_out_image;
    int *dev_H;

    //size of image
    int image_size = in_image.cols*in_image.rows;

    //allocate memory on the GPU
    cuda_status = cudaMalloc((void**)&dev_in_image,sizeof(unsigned char)*image_size);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaMalloc Failed");
        exit( EXIT_FAILURE );
    }
    cuda_status = cudaMalloc((void**)&dev_out_image,sizeof(short)*image_size);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaMalloc Failed");
        exit( EXIT_FAILURE );
    }
    cuda_status = cudaMalloc((void**)&dev_H,sizeof(int)*9);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaMalloc Failed");
        exit( EXIT_FAILURE );
    }


    //copy 
    cuda_status = cudaMemcpy(dev_in_image,in_image.data,sizeof(unsigned char)*image_size,cudaMemcpyHostToDevice);  
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaMemcpy Failed");
        exit( EXIT_FAILURE );
    }
    cudaMemset(dev_out_image,0,sizeof(short)*image_size);
    cuda_status = cudaMemcpy(dev_H,H,sizeof(int)*9,cudaMemcpyHostToDevice);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaMemcpy Failed");
        exit( EXIT_FAILURE );
    }

    //texture
    cuda_status = cudaBindTexture(NULL,tex_in_image,dev_in_image,image_size);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaBindTexture Failed");
        exit( EXIT_FAILURE );
    }



    dim3 threads(16,16);
    dim3 grid(max((in_image.cols+threads.x-1)/threads.x,1),max((in_image.rows+threads.y-1)/threads.y,1));
    convolution_kernel<<<grid,threads>>>(dev_out_image,dev_H,in_image.cols,in_image.rows);

    //copy out
    cuda_status = cudaMemcpy((short*)out_image.data,dev_out_image,sizeof(short)*image_size,cudaMemcpyDeviceToHost);
    if(cuda_status != cudaSuccess)
    {
        fprintf(stderr,"cudaMemcpy Failed");
        exit( EXIT_FAILURE );
    }
    
    cudaUnbindTexture( tex_in_image );
    cudaFree(dev_in_image);
    cudaFree(dev_out_image);
    cudaFree(dev_H);

    cout<<start_time.elapsedMs()<<endl;  

    //output
    Mat abs_dst;
    convertScaleAbs( out_image, abs_dst );  
    imwrite("cuda_texture.jpg",abs_dst);


    
    return 0;
}