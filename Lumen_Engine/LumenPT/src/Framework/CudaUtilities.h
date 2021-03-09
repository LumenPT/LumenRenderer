#pragma once
#include <Optix/optix_host.h>
#include <Cuda/cuda_runtime.h>
#include <string>

inline void CheckOptixRes(const OptixResult& a_res)
{
    if (a_res != OPTIX_SUCCESS)
    {
        const std::string errorName = optixGetErrorName(a_res);
        const std::string errorMessage = optixGetErrorString(a_res);

        std::fprintf(
            stderr,
            "Optix error occured: %s \n Description: %s",
            errorName.c_str(),
            errorMessage.c_str());

#if !defined(NO_ABORT)

        abort();

#endif

    }
}

inline void CheckCudaErr(const cudaError& a_err)
{

    if (a_err != cudaSuccess)
    {

        const std::string errorName = cudaGetErrorName(a_err);
        const std::string errorMessage = cudaGetErrorString(a_err);

        std::fprintf(
            stderr,
            "Optix error occured: %s \n Description: %s",
            errorName.c_str(),
            errorMessage.c_str());

#if !defined(NO_ABORT)

        abort();

#endif

}

}

inline void CheckCudaLastErr()
{

    cudaError err = cudaGetLastError();
    CheckCudaErr(err);

}

#if defined(OPTIX_NOCHECK)  || ! defined(_DEBUG)
#define CHECKOPTIXRESULT
#elif defined(OPTIX_CHECK) || defined(_DEBUG)
#define CHECKOPTIXRESULT(x)\
    CheckOptixRes(x);
#endif

#if defined(CUDA_NOCHECK)  || ! defined(_DEBUG)
#define CHECKCUDAERROR 
#elif defined(CUDA_CHECK) || defined(_DEBUG)
#define CHECKCUDAERROR(x)\
    CheckCudaErr(x);
#endif

#if defined(CUDA_NOCHECK) || ! defined(_DEBUG)
#define CHECKLASTCUDAERROR
#elif defined(CUDA_CHECK) || defined(_DEBUG)
#define CHECKLASTCUDAERROR\
    CheckCudaLastErr();
#endif