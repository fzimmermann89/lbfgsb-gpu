
#include <torch/extension.h>
#include <vector>
#include <pybind11/functional.h>
#include "culbfgsb/culbfgsb.h"
#include <cuda.h>
#include <cuda_runtime.h>

using real = double;

int run(

    torch::Tensor x0, torch::Tensor xl, torch::Tensor xu, torch::Tensor nbd, std:: function < float(torch::Tensor,LBFGSB_CUDA_SUMMARY<real>) > & callback, int maxiter) {

    LBFGSB_CUDA_OPTION < real > lbfgsb_options;

    lbfgsbcuda::lbfgsbdefaultoption < real > (lbfgsb_options);
    lbfgsb_options.mode = LCM_CUDA;
    lbfgsb_options.eps_f = static_cast < real > (1e-9);
    lbfgsb_options.eps_g = static_cast < real > (1e-9);
    lbfgsb_options.eps_x = static_cast < real > (1e-9);
    lbfgsb_options.max_iteration = maxiter;

    LBFGSB_CUDA_STATE < real > state;
    memset( & state, 0, sizeof(state));
    cublasStatus_t stat = cublasCreate( & (state.m_cublas_handle));
    auto options = x0.options().requires_grad(false);
    x0.requires_grad_(true);
    auto sizes = x0.sizes();
    float f;

    state.m_funcgrad_callback = [ & x0, & callback, &sizes, &options](
        real * x, real & f, real * g,
        const cudaStream_t & stream,
            const LBFGSB_CUDA_SUMMARY < real > & summary) {
        auto gten = torch::from_blob(g, sizes, options);
        x0.mutable_grad() = gten;
        f = callback(x0,summary);

        return 0;
    };

    LBFGSB_CUDA_SUMMARY < real > summary;
    memset( & summary, 0, sizeof(summary));

    int N_elements = x0.numel();
    auto x0p = x0.data_ptr < real >();
    auto nbdp= nbd.data_ptr < int > ();
    auto xlp =xl.data_ptr < real >();
    auto xup=xu.data_ptr < real >();
    lbfgsbcuda::lbfgsbminimize < real > (N_elements, state, lbfgsb_options, x0p, nbdp , xlp, xup , summary);

    x0.mutable_grad() = torch::Tensor();

    return summary.info;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run", & run, "run");
    py::class_<LBFGSB_CUDA_SUMMARY<real>>(m, "LBFGSB_CUDA_SUMMARY")
      .def_readonly("num_iteration", &LBFGSB_CUDA_SUMMARY<real>::num_iteration)
      .def_readonly("residual_g", &LBFGSB_CUDA_SUMMARY<real>::residual_g)
      .def_readonly("residual_f", &LBFGSB_CUDA_SUMMARY<real>::residual_f)
      .def_readonly("residual_x", &LBFGSB_CUDA_SUMMARY<real>::residual_x)
      .def_readonly("info", &LBFGSB_CUDA_SUMMARY<real>::info);



}

