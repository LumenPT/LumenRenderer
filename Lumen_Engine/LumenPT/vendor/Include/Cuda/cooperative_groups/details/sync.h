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

#ifndef _CG_GRID_H
#define _CG_GRID_H

#include "info.h"

_CG_BEGIN_NAMESPACE

namespace details
{
_CG_STATIC_QUALIFIER bool bar_has_flipped(unsigned int old_arrive, unsigned int current_arrive)
{
    return (((old_arrive ^ current_arrive) & 0x80000000) != 0);
}

_CG_STATIC_QUALIFIER void bar_flush(volatile unsigned int *addr)
{
#if __CUDA_ARCH__ < 700
    __threadfence();
#else
    unsigned int val;
    asm volatile("ld.acquire.gpu.u32 %0,[%1];" : "=r"(val) : _CG_ASM_PTR_CONSTRAINT((unsigned int*)addr) : "memory");
    // Avoids compiler warnings from unused variable val
    (void)(val = val);
#endif
}

_CG_STATIC_QUALIFIER unsigned int atomic_add(volatile unsigned int *addr, unsigned int val) {
    unsigned int old;
#if __CUDA_ARCH__ < 700
    old = atomicAdd((unsigned int*)addr, val);
#else
    asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(old) : _CG_ASM_PTR_CONSTRAINT((unsigned int*)addr), "r"(val) : "memory");
#endif
    return old;
}

_CG_STATIC_QUALIFIER void sync_grids(unsigned int expected, volatile unsigned int *arrived) {
    bool cta_master = (threadIdx.x + threadIdx.y + threadIdx.z == 0);
    bool gpu_master = (blockIdx.x + blockIdx.y + blockIdx.z == 0);

    __syncthreads();

    if (cta_master) {
        unsigned int nb = 1;
        if (gpu_master) {
            nb = 0x80000000 - (expected - 1);
        }

        __threadfence();

        unsigned int oldArrive;
        oldArrive = atomic_add(arrived, nb);

        while (!bar_has_flipped(oldArrive, *arrived));

        //flush barrier upon leaving
        bar_flush((unsigned int*)arrived);
    }

    __syncthreads();
}

_CG_STATIC_QUALIFIER void sync_warps(unsigned int expected, volatile unsigned int *arrived, unsigned int thread_rank) {
    bool group_master = (thread_rank == 0);
    bool warp_master = (thread_rank % 32 == 0);

    __syncwarp(0xFFFFFFFF);

    if (warp_master) {
        unsigned int nb = 1;

        if (group_master) {
            nb = 0x80000000 - (expected - 1);
        }

        unsigned int oldArrive;
        oldArrive = atomic_add(arrived, nb);

        while (!bar_has_flipped(oldArrive, *arrived));
    }

    __syncwarp(0xFFFFFFFF);
}

_CG_STATIC_QUALIFIER unsigned int sync_warps_arrive(unsigned int expected, volatile unsigned int *arrived, unsigned int thread_rank) {
    bool group_master = (thread_rank == 0);
    bool warp_master = (thread_rank % 32 == 0);

    __syncwarp(0xFFFFFFFF);

    if (warp_master) {
        unsigned int nb = 1;

        if (group_master) {
            nb = 0x80000000 - (expected - 1);
        }

        return atomic_add(arrived, nb);
    }

    return 0;
}

_CG_STATIC_QUALIFIER void sync_warps_wait(volatile unsigned int *arrived, unsigned int old_arrive, unsigned int thread_rank) {
    bool warp_master = (thread_rank % 32 == 0);

    if (warp_master) {
        while (!bar_has_flipped(old_arrive, *arrived));
    }

    __syncwarp(0xFFFFFFFF);
}

_CG_STATIC_QUALIFIER bool sync_warps_last_releases(unsigned int expected, volatile unsigned int *arrived, unsigned int thread_rank) {
    bool warp_master = (thread_rank % 32 == 0);

    __syncwarp(0xFFFFFFFF);

    unsigned int oldArrive = 0;
    if (warp_master) {
        oldArrive = atomic_add(arrived, 1);
    }

    // Shfl here so all threads in the last warp can return
    oldArrive = __shfl_sync(0xFFFFFFFF, oldArrive, 0);
    if (((oldArrive + 1) & ~0x80000000) == expected) {
        return true;
    }

    while (!bar_has_flipped(oldArrive, *arrived));

    return false;
}

_CG_STATIC_QUALIFIER void sync_warps_wait_for_release(volatile unsigned int *arrived, bool is_master, unsigned int warp_sync_mask = 0xFFFFFFFF) {
    __syncwarp(warp_sync_mask);

    unsigned int oldArrive = 0;
    if (is_master) {
        oldArrive = atomic_add(arrived, 1);
        while (!bar_has_flipped(oldArrive, *arrived));
    }

    __syncwarp(warp_sync_mask);
}

_CG_STATIC_QUALIFIER void sync_warps_wait_for_count(unsigned int expected_count, volatile unsigned int *arrived) {
    while ((*arrived & ~0x80000000) < expected_count);
}

_CG_STATIC_QUALIFIER void sync_warps_release(volatile unsigned int *arrived, bool is_master) {
    if (is_master) {
        *arrived = (*arrived ^ 0x80000000) & 0x80000000;
    }
}

} // details

_CG_END_NAMESPACE

#endif // _CG_GRID_H
