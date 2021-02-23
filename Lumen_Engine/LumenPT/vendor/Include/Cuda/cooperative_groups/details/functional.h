 /* Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
  *
  * NOTICE TO LICENSEE:
  *
  * The source code and/or documentation ("Licensed Deliverables") are
  * subject to NVIDIA intellectual property rights under U.S. and
  * international Copyright laws.
  *
  * The Licensed Deliverables contained herein are PROPRIETARY and
  * CONFIDENTIAL to NVIDIA and are being provided under the terms and
  * conditions of a form of NVIDIA software license agreement by and
  * between NVIDIA and Licensee ("License Agreement") or electronically
  * accepted by Licensee.  Notwithstanding any terms or conditions to
  * the contrary in the License Agreement, reproduction or disclosure
  * of the Licensed Deliverables to any third party without the express
  * written consent of NVIDIA is prohibited.
  *
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
  * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  THEY ARE
  * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
  * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
  * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
  * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
  * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
  * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
  * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
  * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
  * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
  * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
  * OF THESE LICENSED DELIVERABLES.
  *
  * U.S. Government End Users.  These Licensed Deliverables are a
  * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
  * 1995), consisting of "commercial computer software" and "commercial
  * computer software documentation" as such terms are used in 48
  * C.F.R. 12.212 (SEPT 1995) and are provided to the U.S. Government
  * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
  * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
  * U.S. Government End Users acquire the Licensed Deliverables with
  * only those rights set forth herein.
  *
  * Any use of the Licensed Deliverables in individual and commercial
  * software must include, in the user documentation and internal
  * comments to the code, the above Disclaimer and U.S. Government End
  * Users Notice.
  */

#ifndef _CG_FUNCTIONAL_H
#define _CG_FUNCTIONAL_H

#include "info.h"

#ifdef _CG_USE_CUDA_STL
# include <cuda/std/functional>
#endif

_CG_BEGIN_NAMESPACE

namespace details {
#ifdef _CG_USE_CUDA_STL
    using cuda::std::plus;
    using cuda::std::bit_and;
    using cuda::std::bit_xor;
    using cuda::std::bit_or;
#else
    template <typename Ty> struct plus {__device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {return arg1 + arg2;}};
    template <typename Ty> struct bit_and {__device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {return arg1 & arg2;}};
    template <typename Ty> struct bit_xor {__device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {return arg1 ^ arg2;}};
    template <typename Ty> struct bit_or {__device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {return arg1 | arg2;}};
#endif // _CG_USE_PLATFORM_STL
} // details

template <typename Ty>
struct plus : public details::plus<Ty> {};

template <typename Ty>
struct less {
    __device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {
        return (arg2 < arg1) ? arg2 : arg1;
    }
};

template <typename Ty>
struct greater {
    __device__ __forceinline__ Ty operator()(Ty arg1, Ty arg2) const {
        return (arg1 < arg2) ? arg2 : arg1;
    }
};

template <typename Ty>
struct bit_and : public details::bit_and<Ty> {};

template <typename Ty>
struct bit_xor : public details::bit_xor<Ty> {};

template <typename Ty>
struct bit_or : public details::bit_or<Ty> {};

_CG_END_NAMESPACE

#endif //_CG_FUNCTIONAL_H
