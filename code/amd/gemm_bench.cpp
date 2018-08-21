#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cmath>
#include <rocblas.h>

#include "tensor.h"
#include "gemm_problems.h"

void MyF(int a) { std::cout << "VVVVVV=" << a << "\n"; };

//typedef void (*gemm_function) (int);

template <typename T1, typename T2,
          rocblas_status (*ROCBLAS_GEMM_FUNCTION)(rocblas_handle handle,
                                       rocblas_operation transa,
                                       rocblas_operation transb,
                                       rocblas_int m,
                                       rocblas_int n,
                                       rocblas_int k,
                                       const T1* alpha,
                                       const T1* A,
                                       rocblas_int lda,
                                       const T1* B,
                                       rocblas_int ldb,
                                       const T1* beta,
                                       T2* C,
                                       rocblas_int ldc) >
int time_gemm(Tensor<T1> A, Tensor<T1> B, Tensor<T2> C, bool a_t, bool b_t, rocblas_handle handle) {
    const T1 alpha = 1.f / static_cast<T1>(A.dims()[1]);
    const T1 beta  = 1.f;

    int m = C.dims()[0];
    int k = a_t ? A.dims()[0] : A.dims()[1];
    int n = C.dims()[1];

    int numRepeats = std::max(std::ceil(1e11 / (m * k * n)), 10.);

    // Warm up
    rocblas_status stat = ROCBLAS_GEMM_FUNCTION(
                		handle,
                		a_t ? rocblas_operation_transpose : rocblas_operation_none,
                		b_t ? rocblas_operation_transpose : rocblas_operation_none,
                		m, n, k,
                		&alpha,
                		A.begin(), A.dims()[0],
                		B.begin(), B.dims()[0],
                		&beta,
                    C.begin(), C.dims()[0] );

    if (stat != rocblas_status_success) {
        throw std::runtime_error("gemm failed");
    }

    hipDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
      rocblas_status stat = ROCBLAS_GEMM_FUNCTION(
                  		handle,
                		a_t ? rocblas_operation_transpose : rocblas_operation_none,
                		b_t ? rocblas_operation_transpose : rocblas_operation_none,
                  		m, n, k,
                  		&alpha,
                  		A.begin(), A.dims()[0],
                  		B.begin(), B.dims()[0],
                  		&beta,
                      C.begin(), C.dims()[0] );
        if (stat != rocblas_status_success) {
            throw std::runtime_error("gemm failed");
        }
    }
    hipDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    return static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);
}

int main(int argc, char **argv) {
    hipFree(0);

    std::string precision = "float";
    if (argc > 1) {
        precision = argv[1];
    }

    rocblas_handle handle;
    rocblas_create_handle(&handle);


    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t     b_t       precision    time(us)   gflops";
    std::cout << "\n";

    int totalTime = 0;
    double geoFlops = 1.0;
    int numProblems = 0;
    for (const auto &problem : training_set) {
        int m, n, k;
        bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;

        std::cout << std::setw(7) << m;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << (a_t ? "t" : "n");
        std::cout << std::setw(7) << (b_t ? "t" : "n");
        int time;
        double flops;
        if (precision == "half_pure") {
          auto a = rand<uint16_t>({a_t ? k : m, a_t ? m : k});
          auto b = rand<uint16_t>({b_t ? n : k, b_t ? k : n});
          auto c = zeros<uint16_t>({m, n});
          std::cout << std::setw(13) << precision;
          time=  time_gemm<uint16_t, uint16_t, rocblas_hgemm>(a, b, c, a_t, b_t, handle);

          flops = (double(m)*n*k*2/time/1000.0);
          std::cout << std::setw(13) << std::setprecision(6) << time << "       " << flops;
        //else if (precision == "half_mixed") {
        } else if (precision == "float") {
          auto a = rand<float>({a_t ? k : m, a_t ? m : k});
          auto b = rand<float>({b_t ? n : k, b_t ? k : n});
          auto c = zeros<float>({m, n});
          std::cout << std::setw(13) << precision;
          time = time_gemm<float,float, rocblas_sgemm>(a, b, c, a_t, b_t, handle);
          flops = (double(m)*n*k*2/time/1000.0);
          std::cout << std::setw(13) << std::setprecision(6) << time << "       " << flops;
        } else {
          throw std::runtime_error(std::string("unsupported precision=")+precision);
        }

        numProblems++;
        totalTime += time;
        geoFlops *= flops;
        std::cout << std::endl;
    }
    double power = 1.0/numProblems;
    geoFlops = std::pow(geoFlops, power);
    //Summary shows total-time and  geomean(gflops)
    std::cout << std::setw(7+7+7+7+7) << "summary:" << std::setw(13) << "";
    std::cout << std::setw(13) << std::setprecision(6) << totalTime << "       " << geoFlops << "\n";

    rocblas_destroy_handle(handle);
    return 0;
}

