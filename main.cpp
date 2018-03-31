#include <sstream>
#include <iostream>
#include <fstream>

#include "comm.h"

#define N 2500
#define LOOSE_FACTOR 1.05

void TreeBasedBinPacking(int bin_num, Bin2D* bins, int canvas_width, int canvas_heigth);

int main()
{
	
	Bin2D bins[N];

	// random N bins

	srand(time(NULL));

	float w_len_total = 0.0, h_len_total = 0.0, area_total = 0.0;

	for (int i = 0; i < N; ++i)
	{
		int w = rand() % 100 + 1;
		int h = rand() % 100 + 1;

		bins[i].x = 0;
		bins[i].y = 0;

		bins[i].W = w;
		bins[i].H = h;

		area_total += w * h;
	}

	int canvas_width  = sqrt(area_total) * LOOSE_FACTOR;
	int canvas_height = sqrt(area_total) * LOOSE_FACTOR;

	std::cout << "Canvas size: " << canvas_width << "," << canvas_height << std::endl;

	// Packing

	TreeBasedBinPacking(N, bins, canvas_width, canvas_height);

	std::stringstream debug_outputstream;
	std::cout << "Packing Results:" << std::endl;

	for (int i = 0; i < N; ++i) {
		debug_outputstream << (float)bins[i].x << ","
			<< (float)bins[i].y << ","
			<< (float)bins[i].W << ","
			<< (float)bins[i].H << ";"
			<< std::endl;
	}

	std::string debug_outputfile = "packing_result.txt";
	std::ofstream ofstr(debug_outputfile.c_str(), std::ios_base::trunc | std::ios_base::out);

	ofstr.write(debug_outputstream.str().c_str(), debug_outputstream.str().length());
	ofstr.close(); 
}