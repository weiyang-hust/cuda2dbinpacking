# Cuda 2D Binpacking Based on Binary Tree Algorithm
Author: Wei Yang

This project is a cuda implementation of 2D binpacking algorithm.

The original idea is from here:
http://blackpawn.com/texts/lightmaps/default.html

There have been a lot of implementations of this algorithm, but no GPU version. Hence I implement it in Cuda.
According to my experiment, the cuda version normally is 9x faster then the CPU version.

# Content
binpacking.cu: cuda souce file of 2d bin packing algorithm.
plotbins.m: a Matlab script to plot the packing result.

# A examplary result of packing 2500 bins
![Alt text](cudabinpacking_result.png?raw=true "Title")

# Contact
If you have any ideas, suggestions or bug reports, reach me at: wyangcs@udel.edu.
