Compare the time used for convolution

TASK
----
Input a image named "test.jpg" and do convolution with a 3×3 kernel like this:
| -1 -1 -1 |
| -1  8 -1 |
| -1 -1 -1 |

COMPARE
----
cpu
cpu_openMP
cuda
cuda_constant
cuda_texture
cuda_texture_2d
cuda_texture_constant

RESULT
---
1. CUDA is FASTER than CPU
2. cuda_texture_constant is the FASTEST
3. image size dosen't slow down the speed of the CUDA
4. Ubuntu is faster than Windows(I have a PC installed Windows and Ubuntu at the same time)
5. openMP is useful when you have no CUDA.

For more information,please refer to the data_and_figure.xlsx

Some Result Figure
---


DATA
----
![IMAGE A](https://wallpapers.wallhaven.cc/wallpapers/full/wallhaven-94976.jpg)
![IMAGE B](https://wallpapers.wallhaven.cc/wallpapers/full/wallhaven-537962.jpg)
