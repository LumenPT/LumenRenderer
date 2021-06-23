#pragma once
#include <comdef.h>
#include <string>

void CheckDX11Res(const HRESULT& a_Res, const char* a_File, int a_Line)
{

    if(FAILED(a_Res))
    {

        _com_error error(a_Res);
        auto message = error.ErrorMessage();

        std::fprintf(
            stderr,
            "DX11 error occured: %s \n"
            "\tFile: %s \n"
            "\tLine: %i \n",
            message,
            a_File,
            a_Line);

#if !defined(NO_ABORT)

        abort();

#endif

    }

}

#if defined(DX11_NOCHECK) || (!defined(_DEBUG) && !defined(DX11_CHECK))
#define CHECKDX11RESULT
#elif defined(DX11_CHECK) || defined(_DEBUG)
#if defined(__FILE__) && defined(__LINE__)
#define CHECKDX11RESULT(x)\
CheckDX11Res(x, __FILE__, __LINE__);
#else
#define CHECKDX11RESULT(x)\
    CheckDX11Res(x, "", 0);
#endif
#endif