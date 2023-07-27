#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <torch/script.h>

#include "ball_query_gpu.h"
#include "group_points_gpu.h"
#include "sampling_gpu.h"
#include "interpolate_gpu.h"


TORCH_LIBRARY(my_ops, m) {
    m.def("ball_query_wrapper", &ball_query_wrapper_fast);

    m.def("group_points_wrapper", &group_points_wrapper_fast);
    m.def("group_points_grad_wrapper", &group_points_grad_wrapper_fast);

    m.def("gather_points_wrapper", &gather_points_wrapper_fast);
    m.def("gather_points_grad_wrapper", &gather_points_grad_wrapper_fast);

    m.def("furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper);

    m.def("three_nn_wrapper", &three_nn_wrapper_fast);
    m.def("three_interpolate_wrapper", &three_interpolate_wrapper_fast);
    m.def("three_interpolate_grad_wrapper", &three_interpolate_grad_wrapper_fast);
}
