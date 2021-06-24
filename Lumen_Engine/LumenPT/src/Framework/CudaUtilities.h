#pragma once
#include <Optix/optix_host.h>
#include <Cuda/cuda_runtime.h>
#include <string>

inline void CheckOptixRes(const OptixResult& a_res, const char* a_File, int a_Line)
{
    if (a_res != OPTIX_SUCCESS)
    {
        const std::string errorName = optixGetErrorName(a_res);
        const std::string errorMessage = optixGetErrorString(a_res);

        std::fprintf(
            stderr,
            "Optix error occured: %s \n"
            "\tFile: %s \n"
            "\tLine: %i \n"
            "\tDescription: %s",
            errorName.c_str(),
            a_File,
            a_Line,
            errorMessage.c_str());

#if !defined(NO_ABORT)

        abort();

#endif

    }
}

inline void CheckCudaErr(const cudaError& a_err, const char* a_File, int a_Line)
{

    if (a_err != cudaSuccess)
    {

        const std::string errorName = cudaGetErrorName(a_err);
        const std::string errorMessage = cudaGetErrorString(a_err);

        std::fprintf(
            stderr,
            "\nCUDA error occured: %s \n"
            "\tFile: %s \n"
            "\tLine: %i \n"
            "\tDescription: %s",
            errorName.c_str(),
            a_File,
            a_Line,
            errorMessage.c_str());

#if !defined(NO_ABORT)

        abort();

#endif

}

}

inline void CheckCudaLastErr(const char* a_File, int a_Line)
{

    cudaDeviceSynchronize();

    cudaError err = cudaGetLastError();
    CheckCudaErr(err, a_File, a_Line);

}

#if defined(OPTIX_NOCHECK)  || (!defined(_DEBUG) && !defined(OPTIX_CHECK))
#define CHECKOPTIXRESULT
#elif defined(OPTIX_CHECK) || defined(_DEBUG)
#if defined(__FILE__) && defined(__LINE__)
#define CHECKOPTIXRESULT(x)\
    CheckOptixRes(x, __FILE__, __LINE__);
#else
#define CHECKOPTIXRESULT(x)\
    CheckOptixRes(x, "", 0);
#endif
#endif

#if defined(CUDA_NOCHECK)  || (!defined(_DEBUG) && !defined(CUDA_CHECK))
#define CHECKCUDAERROR 
#elif defined(CUDA_CHECK) || defined(_DEBUG)
#if defined(__FILE__) && defined(__LINE__)
#define CHECKCUDAERROR(x)\
    CheckCudaErr(x, __FILE__, __LINE__);
#else
#defined CHECKCUDAERROR(x)\
    CheckCudaErr(x, "", 0);
#endif
#endif

#if defined(CUDA_NOCHECK) || (!defined(_DEBUG) && !defined(CUDA_CHECK))
#define CHECKLASTCUDAERROR
#elif defined(CUDA_CHECK) || defined(_DEBUG)
#if defined(__FILE__) && defined(__LINE__)
#define CHECKLASTCUDAERROR\
    CheckCudaLastErr(__FILE__, __LINE__);
#else
#define CHECKLASTCUDAERROR\
    CheckCudaLastErr("", 0);
#endif
#endif