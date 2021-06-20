#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <THC/THC.h>
#include <iostream>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <cuda_runtime_api.h>

#include<cmath>
#include<iostream>
#include<cstdlib>

static const float 	TWO_DIV_PI = 0.6366197723;
static const float 	PI_HALF    = 1.570796327;
static const float 	PI    	   = 3.1415926535;
static const float  TWO_PI 	   = 6.283185307;

// FIX_PT -> Determines the number of decimal places upto which to be considered
// NMAX -> Number of iterations in the CORDIC algo

// These values can be altered to view the effect on the accuracy of the model
static const int 	FIX_PT 	   = 11;
static const int 	NMAX 	   = 11;

static const float LUT_arctan[] = {
		0.7853981633974483,
		0.4636476090008061,
		0.24497866312686414,
		0.12435499454676144,
		0.06241880999595735,
		0.031239833430268277,
		0.015623728620476831,
		0.007812341060101111,
		0.0039062301319669718,
		0.0019531225164788188,
		0.0009765621895593195,
		0.0004882812111948983,
		0.00024414062014936177,
		0.00012207031189367021,
		6.103515617420877e-05,
		3.0517578115526096e-05,
		1.5258789061315762e-05,
		7.62939453110197e-06,
		3.814697265606496e-06,
		1.907348632810187e-06,
		9.536743164059608e-07
		};

void compute(float &x, float &y, float phi, int fix_pt){
	
	int nMax = NMAX;
	int z_r, z_i;
	int z_r_tmp, z_i_tmp;

	// Shifting the input values based on the fix_pt value -> informs about the number of significant digits after the decimal place
	int p_tmp = (int)(phi*pow(2,fix_pt));
	int x_tmp = (int)(x*pow(2,fix_pt));
	int y_tmp = (int)(y*pow(2,fix_pt));	

	uint16_t arctan[nMax];

	for(int i=0; i<nMax; i++){
		float tmp = LUT_arctan[i];
		// Shifting the float values to a shifted version based on the fix_pt value which determines the number of significant digits after the decimal place
		arctan[i] = (uint16_t)(tmp*pow(2,fix_pt));
	}

	for(int i=0; i<nMax; i++){
		int a = arctan[i];
		z_r_tmp = x_tmp >> i;
		z_i_tmp = y_tmp >> i;

		if(p_tmp<0){
			p_tmp += a;
			z_r = x_tmp + z_i_tmp;
			z_i = y_tmp - z_r_tmp;

		}else{
			p_tmp -= a;
			z_r = x_tmp - z_i_tmp;
			z_i = y_tmp + z_r_tmp;
		}

		x_tmp = z_r;
		y_tmp = z_i;
	}

	// Dividing by the shifted value
	x = (float)x_tmp/((float)(1<<fix_pt)-1);
	y = (float)y_tmp/((float)(1<<fix_pt)-1);

}

void cordic(float angle, float & appr_sin){
	float abs_ang;
	if(angle<0)
		abs_ang = -angle;
	else
		abs_ang = angle;

	//  fmod function has both division and multiplication - very costly operator
	//  This function is used to bring the input angles in the range of [0,2*pi]
	abs_ang = fmod(abs_ang, TWO_PI);

	float multiple = abs_ang*TWO_DIV_PI;
	int intK = multiple;
	float dif = abs_ang - intK * PI_HALF;

	float alpha;

	if(intK==1 || intK==3)
		alpha = PI_HALF - dif;
	else
		alpha = dif;

	float z_r = 0.607252935;
	float z_i = 0;
	int fix_pt = FIX_PT;

	compute(z_r, z_i, alpha, fix_pt);

	if (intK==0) // First quadrant
		appr_sin = z_i;
	else if(intK==1) // Second quadrant
		appr_sin = z_i;
	else if(intK==2) // Third quadrant
		appr_sin = -z_i;
	else // Fourth quadrant
		appr_sin = -z_i;

	if(angle<0)
		appr_sin = -appr_sin;
}

torch::Tensor sine(torch::Tensor input){
	
	int64_t dim = input.size(0);

	torch::Tensor output = torch::zeros(torch::IntArrayRef({dim}));
	auto access = input.accessor<float,1>();

	for (int i=0; i < access.size(0); i++){
		float appr_sin;
		cordic(access[i], appr_sin);
		output[i] = appr_sin;
	}

	return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sine", &sine,"Sine function (CUDA)");
}