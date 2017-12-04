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
    
    unsigned char * in_image_ptr = in_image.data;
    short * out_image_ptr = (short*)out_image.data;
    for(int x = 0;x<in_image.cols;x++)
    {
        for(int y=0;y<in_image.rows;y++)
        {
            int pos = y*in_image.cols+x;
            if(x==0||y==0||(x==in_image.cols-1)||(y==in_image.rows-1))
            {
                out_image_ptr[pos] = in_image_ptr[pos];
                continue;
            }
            int left = pos - 1;
            int right = pos + 1;
            int up = pos - in_image.cols;
            int down = pos + in_image.cols;
            int up_left = up - 1;
            int up_right = up + 1;
            int down_left = down - 1;
            int down_right = down + 1;

            out_image_ptr[pos] = H[0]*in_image_ptr[up_left] + H[1]*in_image_ptr[up] + H[2]*in_image_ptr[up_right]
                                +H[3]*in_image_ptr[left] + H[4]*in_image_ptr[pos] + H[5]*in_image_ptr[right]
                                +H[6]*in_image_ptr[down_left] + H[7]*in_image_ptr[down] + H[8]*in_image_ptr[down_right];
        }
    }

    cout<<start_time.elapsedMs()<<endl;  

    //output
    Mat abs_dst;
    convertScaleAbs( out_image, abs_dst );  
    imwrite("cpu.jpg",abs_dst);
    
    return 0;
}