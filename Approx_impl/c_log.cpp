// --------------------------------------------------------------------------------------
// c_log.cpp - function to perform the log_compression over the input data
// -by Balaji Adithya V
// Algo used:
// f(x) = log(1+|x|) ~ |x| if |x|<0
// f(x) = log(1+|x|) = log(y), where y=1+|x|, |x|>0
//      Let y = (2**i)*(1+f) , i-> int, f-> fraction & f<0
//      Hence f(x) = i+f, when |x|>0
// --------------------------------------------------------------------------------------

#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <THC/THC.h>
#include <iostream>
#include <ATen/NativeFunctions.h>
#include <ATen/Config.h>
#include <cuda_runtime_api.h>

#include<omp.h>
#include<pybind11/pybind11.h>

#include<cmath>
#include<iostream>
#include<cstdlib>

float compute(float inp){

	int64_t temp = inp;
	int64_t lod_val = 1;
	int count=0;
	

	while(lod_val < temp){
		lod_val = lod_val<<1;
		count++;
	}	
	count--;
	int c = inp*256;
	c = c>>count;
	float f = c/256.0 - 1;


	return count+f;
}

float solve(float x){
	float ans;

	if(abs(x)<1)
		ans = abs(x);
	else{
		ans = compute(1+abs(x));
	} 

	return ans;

}

torch::Tensor Log(torch::Tensor input){

	
	int64_t b_size = input.size(0);
	int64_t filt_n = input.size(1);
	int64_t features = input.size(2);

	torch::Tensor output = torch::zeros(torch::IntArrayRef({b_size, filt_n, features}));
	auto access = input.accessor<float,3>();


	for(int i=0 ;i<b_size; i++){
		for(int j=0; j<filt_n; j++){
			for(int k=0; k<features; k++){
				output[i][j][k] = solve(access[i][j][k]);
			}
		
		}
	}
	return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("Log", &Log,"Log function");
}