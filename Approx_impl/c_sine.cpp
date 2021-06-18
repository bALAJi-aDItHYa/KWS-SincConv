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

// Something's off with the arctan values... need to check on them
// Check out http://www.realitypixels.com/turk/computergraphics/FixedPointTrigonometry.pdf


// Arctan values need to be changed
static const uint16_t arctan[] = {
	51471,
	30385,
	16054,
	8149,
	4090,
	2047,
	1023,
	511,
	255,
	127,
	63,
	31,
	15,
	7,
	3,
	1};

void compute(float &x, float &y, float phi){
	int nMax = 16;
	int z_r, z_i;
	int z_r_tmp, z_i_tmp;
	int p_tmp = (int)(phi * 65536);
	int x_tmp = (int)(x * 65536);
	int y_tmp = (int)(y * 65536);

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
	x = (float)x_tmp/65535.f;
	y = (float)y_tmp/65535.f;
}

void cordic(float angle, float & appr_sin){
	float abs_ang;
	if(angle<0)
		abs_ang = -angle;
	else
		abs_ang = angle;

	abs_ang = abs_ang - floor(abs_ang/TWO_PI);
	std::cout<<"abs_ang = "<<abs_ang<<std::endl;

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

	compute(z_r, z_i, alpha);

	if (intK==0)
		appr_sin = z_i;
	else if(intK==1)
		appr_sin = z_i;
	else if(intK==2)
		appr_sin = -z_i;
	else
		appr_sin = -z_i;

	// appr_sin = -appr_sin;
	if(angle<0)
		appr_sin = -appr_sin;
}

torch::Tensor sine(torch::Tensor input){
	
	int64_t dim = input.size(0);

	torch::Tensor output = torch::zeros(torch::IntArrayRef({dim}));
	auto access = input.accessor<float,1>();

	//Look into CUDA accessors

	// std::cout<<"I'm inside the c++ extension"<<std::endl;
	// std::cout<<access.size(0)<<std::endl;

	for (int i=0; i < access.size(0); i++){
		float appr_sin, acc_sin;
		float err;
		// appr_sin = sin(access[i]);
		cordic(access[i], appr_sin);
		output[i] = appr_sin;
		acc_sin = sin(access[i]);

		err = abs((acc_sin-appr_sin)/acc_sin);
		if(err>1)
			std::cout<<"Angle= "<<access[i]<<" sin val= "<<appr_sin<<" actual val = "<<acc_sin<<std::endl;

	}

	// for (int i=0; i<dim; i++){
	// 	float appr_sin;
	// 	cordic(input[i].data_ptr<float>(), appr_sin);
	// 	output[i] = appr_sin;
	// }

	return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sine", &sine,"Sine function (CUDA)");
}