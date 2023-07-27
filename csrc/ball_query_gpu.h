#ifndef _BALL_QUERY_GPU_H
#define _BALL_QUERY_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

void ball_query_wrapper_fast(int64_t b, int64_t n, int64_t m, double radius, int64_t nsample,
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void ball_query_kernel_launcher_fast(int b, int n, int m, float radius, int nsample, 
	const float *xyz, const float *new_xyz, int *idx);

#endif
