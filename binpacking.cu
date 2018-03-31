#include "comm.h"
#include <iostream>
#include <fstream>

#define MAX_BINS         5000
#define MAX_CHILD_NUMBER (20 * MAX_BINS + 2)

__device__  Bin2D	 g_dev_bins[MAX_BINS];

__device__	int		 g_child_count;
__device__  TreeNode g_child_list[MAX_CHILD_NUMBER];

__global__ 
void InitDevBinsKernel(int canvas_width, int canvas_height)
{
	unsigned int Index = threadIdx.x + blockIdx.x * blockDim.x;

	if (Index >= MAX_CHILD_NUMBER)
		return;

	if (Index == 0)
	{
		g_child_count = 1;

		g_child_list[Index].Wr = canvas_width;
		g_child_list[Index].Hr = canvas_height;
	}
	else
	{
		g_child_list[Index].Wr = -1;
		g_child_list[Index].Hr = -1;
	}
	
	g_child_list[Index].BinId = -1;

	g_child_list[Index].x = 0;
	g_child_list[Index].y = 0;

	g_child_list[Index].left_child  = -1;
	g_child_list[Index].right_child = -1;
}

__device__
void SplitCanvas(int cur, int bid, int lid, int rid)
{

	int diff_w = g_child_list[cur].Wr - g_dev_bins[bid].W;
	int diff_h = g_child_list[cur].Hr - g_dev_bins[bid].H;

	if (g_child_list[lid].left_child  >= 0 ||
		g_child_list[lid].right_child >= 0 ||
		g_child_list[rid].left_child  >= 0 ||
		g_child_list[rid].right_child >= 0)
	{
		// shouldn't hit here, something is wrong
	}

	g_child_list[lid] = g_child_list[cur];
	g_child_list[rid] = g_child_list[cur];

	g_child_list[cur].left_child  = lid;
	g_child_list[cur].right_child = rid;

	g_child_list[lid].BinId = -1;
	g_child_list[rid].BinId = -1;

	if (diff_w > diff_h)
	{
		g_child_list[lid].Wr  = g_dev_bins[bid].W;
		g_child_list[rid].Wr -= g_dev_bins[bid].W;

		g_child_list[rid].x  += g_dev_bins[bid].W;
	}
	else
	{
		g_child_list[lid].Hr  = g_dev_bins[bid].H;
		g_child_list[rid].Hr -= g_dev_bins[bid].H;

		g_child_list[rid].y  += g_dev_bins[bid].H;
	}

}

__device__ 
int Insert(int current, int bi)
{
	if (g_child_list[current].left_child >= 0 || g_child_list[current].right_child >= 0)
	{
		int res = Insert(g_child_list[current].left_child, bi);
		
		if (res == -1)
		{
			return Insert(g_child_list[current].right_child, bi);
		}
		return res;
	}
	else
	{
		if (g_child_list[current].BinId >= 0 || g_dev_bins[bi].W > g_child_list[current].Wr || g_dev_bins[bi].H > g_child_list[current].Hr)
		{
			return -1;
		}

		if (g_dev_bins[bi].W == g_child_list[current].Wr && g_dev_bins[bi].H == g_child_list[current].Hr)
		{
			if (atomicCAS(&g_child_list[current].BinId, -1, bi) == -1) // mutex
			{
				g_dev_bins[bi].x = g_child_list[current].x;
				g_dev_bins[bi].y = g_child_list[current].y;

				return 0;
			}
			else
			{
				return -2;
			}
		}

		if (atomicCAS(&g_child_list[current].BinId, -1, -2) == -1) // mutex
		{
			int lch = atomicAdd(&g_child_count, 1);
			int rch = atomicAdd(&g_child_count, 1);

			if (g_child_count > MAX_CHILD_NUMBER)
				return -1;

			SplitCanvas(current, bi, lch, rch);

			return Insert(lch, bi);
		}
		else {
			return -2;
		}
	}
}

__global__
void PackingRecursiveKernel(int bin_count)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
	unsigned int stride = blockDim.x * gridDim.x;
	unsigned int offset = 0;

	int iter = 0;

	// insert block by block, reduce race rate
	while ((index + offset) < bin_count && iter < 1000) {

		int res = Insert(0, index + offset);

		if (res == 0) {
			// insert finish
			iter = 0;
			offset += stride;
		}
		else if (res == -2) {
			++iter;
		}
		else if (res == -1) {
			// fail to insert, break
			break;
		}
		else {
			// not possible
			break;
		}
		__syncthreads(); // not strictly needed 
	}

}

void TreeBasedBinPacking(int bin_num, Bin2D* bins, int canvas_width, int canvas_heigth)
{
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(g_dev_bins, bins, bin_num * sizeof(Bin2D), 0, cudaMemcpyHostToDevice));

	dim3 init_block(256, 1);
	dim3 init_grid((MAX_CHILD_NUMBER + init_block.x - 1) / init_block.x, 1);

	InitDevBinsKernel << <init_grid, init_block >> > (canvas_width, canvas_heigth);

	dim3 pack_block(256, 1);
	dim3 pack_grid((bin_num + pack_block.x - 1) / pack_block.x, 1);

	PackingRecursiveKernel << <pack_grid, pack_block >> > (bin_num);

	CUDA_SAFE_CALL(cudaMemcpyFromSymbol(bins, g_dev_bins, bin_num * sizeof(Bin2D), 0, cudaMemcpyDeviceToHost));

	CUDA_CHECK_ERROR();
}
