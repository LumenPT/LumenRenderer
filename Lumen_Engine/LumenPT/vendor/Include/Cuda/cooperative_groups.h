/*
 * Copyright 1993-2016 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
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
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
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

#ifndef _COOPERATIVE_GROUPS_H_
#define _COOPERATIVE_GROUPS_H_

#if defined(__cplusplus) && defined(__CUDACC__)

#include "cooperative_groups/details/info.h"
#include "cooperative_groups/details/driver_abi.h"
#include "cooperative_groups/details/helpers.h"

#if defined(_CG_CPP11_FEATURES) && defined(_CG_USE_CUDA_STL)
#include <cuda/atomic>
#define _CG_THREAD_SCOPE(scope) _CG_STATIC_CONST_DECL cuda::thread_scope thread_scope = scope;
#else
#define _CG_THREAD_SCOPE(scope)
#endif

_CG_BEGIN_NAMESPACE

_CG_CONST_DECL unsigned int coalesced_group_id = 1;
_CG_CONST_DECL unsigned int multi_grid_group_id = 2;
_CG_CONST_DECL unsigned int grid_group_id = 3;
_CG_CONST_DECL unsigned int thread_block_id = 4;

/**
 * class thread_group;
 *
 * Generic thread group type, into which all groups are convertible.
 * It acts as a container for all storage necessary for the derived groups,
 * and will dispatch the API calls to the correct derived group. This means
 * that all derived groups must implement the same interface as thread_group.
 */
class thread_group
{
protected:
    struct group_data {
        unsigned int _unused : 1;
        unsigned int type : 7, : 0;
    };

    struct gg_data  {
        details::grid_workspace *gridWs;
    };

#if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
    struct mg_data  {
        unsigned long long _unused : 1;
        unsigned long long type    : 7;
        unsigned long long handle  : 56;
        const details::multi_grid::multi_grid_functions *functions;
    };
#endif

    struct tg_data {
        unsigned int is_tiled : 1;
        unsigned int type : 7;
        unsigned int size : 24;
        // packed to 4b
        unsigned int metaGroupSize : 16;
        unsigned int metaGroupRank : 16;
        // packed to 8b
        unsigned int mask;
        // packed to 12b
        unsigned int _res;
    };

    friend _CG_QUALIFIER thread_group this_thread();
    friend _CG_QUALIFIER thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);
    friend class thread_block;

    union __align__(8) {
        group_data  group;
        tg_data     coalesced;
        gg_data     grid;
#if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
        mg_data     multi_grid;
#endif
    } _data;

    _CG_QUALIFIER thread_group operator=(const thread_group& src);

    _CG_QUALIFIER thread_group(unsigned int type) {
        _data.group.type = type;
        _data.group._unused = false;
    }

#ifdef _CG_CPP11_FEATURES
    static_assert(sizeof(tg_data) <= 16, "Failed size check");
    static_assert(sizeof(gg_data) <= 16, "Failed size check");
#  ifdef _CG_ABI_EXPERIMENTAL
    static_assert(sizeof(mg_data) <= 16, "Failed size check");
#  endif
#endif

public:
    _CG_THREAD_SCOPE(cuda::thread_scope::thread_scope_device)

    _CG_QUALIFIER unsigned long long size() const;
    _CG_QUALIFIER unsigned long long thread_rank() const;
    _CG_QUALIFIER void sync() const;
    _CG_QUALIFIER unsigned int get_type() const {
        return _data.group.type;
    }

};

template <unsigned int TyId>
struct thread_group_base : public thread_group {
    _CG_QUALIFIER thread_group_base() : thread_group(TyId) {}
    _CG_STATIC_CONST_DECL unsigned int id = TyId;
};

/**
 * thread_group this_thread()
 *
 * Constructs a generic thread_group containing only the calling thread
 */
_CG_QUALIFIER thread_group this_thread()
{
    thread_group g(coalesced_group_id);
    g._data.coalesced.mask = details::lanemask32_eq();
    g._data.coalesced.metaGroupRank = 0;
    g._data.coalesced.metaGroupSize = 1;
    g._data.coalesced.size = 1;
    g._data.coalesced.is_tiled = false;
    return (g);
}

#if defined(_CG_HAS_MULTI_GRID_GROUP)

/**
 * class multi_grid_group;
 *
 * Threads within this this group are guaranteed to be co-resident on the
 * same system, on multiple devices within the same launched kernels.
 * To use this group, the kernel must have been launched with
 * cuLaunchCooperativeKernelMultiDevice (or the CUDA Runtime equivalent),
 * and the device must support it (queryable device attribute).
 *
 * Constructed via this_multi_grid();
 */
# if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
class multi_grid_group;

// Multi grid group requires these functions to be templated to prevent ptxas from trying to use CG syscalls
template <typename = void>
__device__ multi_grid_group this_multi_grid();

class multi_grid_group : public thread_group_base<multi_grid_group_id>
{
private:
    template <typename = void>
    _CG_QUALIFIER multi_grid_group() {
        _data.multi_grid.functions = details::multi_grid::load_grid_intrinsics();
        _data.multi_grid.handle = _data.multi_grid.functions->get_intrinsic_handle();
    }

    friend multi_grid_group this_multi_grid<void>();

public:
    _CG_THREAD_SCOPE(cuda::thread_scope::thread_scope_system)

    _CG_QUALIFIER bool is_valid() const {
        return (_data.multi_grid.handle != 0);
    }

    _CG_QUALIFIER void sync() const {
        if (!is_valid()) {
            _CG_ABORT();
        }
        _data.multi_grid.functions->sync(_data.multi_grid.handle);
    }

    _CG_QUALIFIER unsigned long long size() const {
        _CG_ASSERT(is_valid());
        return _data.multi_grid.functions->size(_data.multi_grid.handle);
    }

    _CG_QUALIFIER unsigned long long thread_rank() const {
        _CG_ASSERT(is_valid());
        return _data.multi_grid.functions->thread_rank(_data.multi_grid.handle);
    }

    _CG_QUALIFIER unsigned int grid_rank() const {
        _CG_ASSERT(is_valid());
        return (_data.multi_grid.functions->grid_rank(_data.multi_grid.handle));
    }

    _CG_QUALIFIER unsigned int num_grids() const {
        _CG_ASSERT(is_valid());
        return (_data.multi_grid.functions->num_grids(_data.multi_grid.handle));
    }
};
# else
class multi_grid_group
{
private:
    unsigned long long _handle;
    unsigned int _size;
    unsigned int _rank;

    friend _CG_QUALIFIER multi_grid_group this_multi_grid();

    _CG_QUALIFIER multi_grid_group() {
        _handle = details::multi_grid::get_intrinsic_handle();
        _size = details::multi_grid::size(_handle);
        _rank = details::multi_grid::thread_rank(_handle);
    }

public:
    _CG_THREAD_SCOPE(cuda::thread_scope::thread_scope_system)

    _CG_QUALIFIER bool is_valid() const {
        return (_handle != 0);
    }

    _CG_QUALIFIER void sync() const {
        if (!is_valid()) {
            _CG_ABORT();
        }
        details::multi_grid::sync(_handle);
    }

    _CG_QUALIFIER unsigned long long size() const {
        _CG_ASSERT(is_valid());
        return _size;
    }

    _CG_QUALIFIER unsigned long long thread_rank() const {
        _CG_ASSERT(is_valid());
        return _rank;
    }

    _CG_QUALIFIER unsigned int grid_rank() const {
        _CG_ASSERT(is_valid());
        return (details::multi_grid::grid_rank(_handle));
    }

    _CG_QUALIFIER unsigned int num_grids() const {
        _CG_ASSERT(is_valid());
        return (details::multi_grid::num_grids(_handle));
    }
};
# endif

/**
 * multi_grid_group this_multi_grid()
 *
 * Constructs a multi_grid_group
 */
# if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
template <typename>
__device__
#else
_CG_QUALIFIER
# endif
multi_grid_group this_multi_grid()
{
    return multi_grid_group();
}
#endif

/**
 * class grid_group;
 *
 * Threads within this this group are guaranteed to be co-resident on the
 * same device within the same launched kernel. To use this group, the kernel
 * must have been launched with cuLaunchCooperativeKernel (or the CUDA Runtime equivalent),
 * and the device must support it (queryable device attribute).
 *
 * Constructed via this_grid();
 */
class grid_group : public thread_group_base<grid_group_id>
{
    friend _CG_QUALIFIER grid_group this_grid();

private:
    _CG_QUALIFIER grid_group(details::grid_workspace *gridWs) {
        _data.grid.gridWs = gridWs;
    }

 public:
    _CG_THREAD_SCOPE(cuda::thread_scope::thread_scope_device)

    _CG_QUALIFIER bool is_valid() const {
        return (_data.grid.gridWs != NULL);
    }

    _CG_QUALIFIER void sync() const {
        if (!is_valid()) {
            _CG_ABORT();
        }
        details::grid::sync(&_data.grid.gridWs->barrier);
    }

    _CG_QUALIFIER unsigned long long size() const {
        return details::grid::size();
    }

    _CG_QUALIFIER unsigned long long thread_rank() const {
        return details::grid::thread_rank();
    }

    _CG_QUALIFIER dim3 group_dim() const {
        return details::grid::grid_dim();
    }
};

_CG_QUALIFIER grid_group this_grid() {
    // Load a workspace from the driver
    grid_group gg(details::get_grid_workspace());
#ifdef _CG_DEBUG
    // *all* threads must be available to synchronize
    gg.sync();
#endif // _CG_DEBUG
    return gg;
}

#if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
namespace details {
    _CG_STATIC_QUALIFIER constexpr unsigned int scratch_size_needed(unsigned int communication_size, unsigned int max_block_size) {
        return max_block_size / 32 * (sizeof(int) + communication_size);
    }

    _CG_CONST_DECL unsigned int default_tile_communication_size = 8;
    _CG_CONST_DECL unsigned int default_max_block_size = 1024;

    struct multi_warp_scratch {
        unsigned int communication_size;
        unsigned int max_block_size;
        char memory[1];
    };
}

class thread_block;
namespace experimental {
    template <unsigned int TileCommunicationSize = details::default_tile_communication_size,
              unsigned int MaxBlockSize = details::default_max_block_size>
    struct block_tile_memory {
    private:
        char scratch[details::scratch_size_needed(TileCommunicationSize, MaxBlockSize)];

    public:
        _CG_QUALIFIER void* get_memory() {
            return static_cast<void*>(scratch);
        }

        _CG_STATIC_QUALIFIER unsigned int get_size() {
            return details::scratch_size_needed(TileCommunicationSize, MaxBlockSize);
        }
    };

    template <unsigned int TileCommunicationSize, unsigned int MaxBlockSize>
    _CG_QUALIFIER thread_block this_thread_block(experimental::block_tile_memory<TileCommunicationSize, MaxBlockSize>& scratch);
}
#endif

/**
 * class thread_block
 *
 * Every GPU kernel is executed by a grid of thread blocks, and threads within
 * each block are guaranteed to reside on the same streaming multiprocessor.
 * A thread_block represents a thread block whose dimensions are not known until runtime.
 *
 * Constructed via this_thread_block();
 */
class thread_block : public thread_group_base<thread_block_id>
{
    // Friends
    friend _CG_QUALIFIER thread_block this_thread_block();
    friend _CG_QUALIFIER thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);
    friend _CG_QUALIFIER thread_group tiled_partition(const thread_block& parent, unsigned int tilesz);

#if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
    template <unsigned int TileCommunicationSize, unsigned int MaxBlockSize>
    friend _CG_QUALIFIER thread_block experimental::this_thread_block(
            experimental::block_tile_memory<TileCommunicationSize, MaxBlockSize>& scratch);

    details::multi_warp_scratch* const tile_memory;

    template <unsigned int Size>
    friend class __static_size_multi_warp_tile_base;

    template <unsigned int TileCommunicationSize, unsigned int MaxBlockSize>
    _CG_QUALIFIER thread_block(experimental::block_tile_memory<TileCommunicationSize, MaxBlockSize>& scratch) :
        tile_memory(reinterpret_cast<details::multi_warp_scratch*>(&scratch)) {
        if (thread_rank() < MaxBlockSize / 32) {
            unsigned int* barriers = reinterpret_cast<unsigned int*>(&tile_memory->memory);
            barriers[thread_rank()] = 0;
        }
        if (thread_rank() == MaxBlockSize / 32) {
            tile_memory->communication_size = TileCommunicationSize;
            tile_memory->max_block_size = MaxBlockSize;
        }
        sync();
    }
#endif

    // Disable constructor
    _CG_QUALIFIER thread_block()
#if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
    : tile_memory(NULL)
#endif
    { }

    // Internal Use
    _CG_QUALIFIER thread_group _get_tiled_threads(unsigned int tilesz) const {
        const bool pow2_tilesz = ((tilesz & (tilesz - 1)) == 0);

        // Invalid, immediately fail
        if (tilesz == 0 || (tilesz > 32) || !pow2_tilesz) {
            details::abort();
            return (thread_block());
        }

        unsigned int mask;
        unsigned int base_offset = thread_rank() & (~(tilesz - 1));
        unsigned int masklength = min((unsigned int)size() - base_offset, tilesz);

        mask = (unsigned int)(-1) >> (32 - masklength);
        mask <<= (details::laneid() & ~(tilesz - 1));
        thread_group tile = thread_group(coalesced_group_id);
        tile._data.coalesced.mask = mask;
        tile._data.coalesced.size = __popc(mask);
        tile._data.coalesced.metaGroupSize = (details::cta::size() + tilesz - 1) / tilesz;
        tile._data.coalesced.metaGroupRank = details::cta::thread_rank() / tilesz;
        tile._data.coalesced.is_tiled = true;
        return (tile);
    }

 public:
    _CG_THREAD_SCOPE(cuda::thread_scope::thread_scope_block)

    _CG_STATIC_QUALIFIER void sync() {
        details::cta::sync();
    }

    _CG_STATIC_QUALIFIER unsigned int size() {
        return (details::cta::size());
    }

    _CG_STATIC_QUALIFIER unsigned int thread_rank() {
        return (details::cta::thread_rank());
    }

    // Additional functionality exposed by the group
    _CG_STATIC_QUALIFIER dim3 group_index() {
        return (details::cta::group_index());
    }

    _CG_STATIC_QUALIFIER dim3 thread_index() {
        return (details::cta::thread_index());
    }

    _CG_STATIC_QUALIFIER dim3 group_dim() {
        return (details::cta::block_dim());
    }

};

/**
 * thread_block this_thread_block()
 *
 * Constructs a thread_block group
 */
_CG_QUALIFIER thread_block this_thread_block()
{
    return (thread_block());
}

#if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
namespace experimental {
    template <unsigned int TileCommunicationSize, unsigned int MaxBlockSize>
    _CG_QUALIFIER thread_block this_thread_block(experimental::block_tile_memory<TileCommunicationSize, MaxBlockSize>& scratch) {
        return (thread_block(scratch));
    }
}
#endif

/**
 * class coalesced_group
 *
 * A group representing the current set of converged threads in a warp.
 * The size of the group is not guaranteed and it may return a group of
 * only one thread (itself).
 *
 * This group exposes warp-synchronous builtins.
 * Constructed via coalesced_threads();
 */
class coalesced_group : public thread_group_base<coalesced_group_id>
{
private:
    friend _CG_QUALIFIER coalesced_group coalesced_threads();
    friend _CG_QUALIFIER thread_group tiled_partition(const thread_group& parent, unsigned int tilesz);
    friend _CG_QUALIFIER coalesced_group tiled_partition(const coalesced_group& parent, unsigned int tilesz);
    friend class details::_coalesced_group_data_access;

    _CG_QUALIFIER unsigned int _packLanes(unsigned laneMask) const {
        unsigned int member_pack = 0;
        unsigned int member_rank = 0;
        for (int bit_idx = 0; bit_idx < 32; bit_idx++) {
            unsigned int lane_bit = _data.coalesced.mask & (1 << bit_idx);
            if (lane_bit) {
                if (laneMask & lane_bit)
                    member_pack |= 1 << member_rank;
                member_rank++;
            }
        }
        return (member_pack);
    }

    // Internal Use
    _CG_QUALIFIER coalesced_group _get_tiled_threads(unsigned int tilesz) const {
        const bool pow2_tilesz = ((tilesz & (tilesz - 1)) == 0);

        // Invalid, immediately fail
        if (tilesz == 0 || (tilesz > 32) || !pow2_tilesz) {
            details::abort();
            return (coalesced_group(0));
        }
        if (size() <= tilesz) {
            return (*this);
        }

        if ((_data.coalesced.is_tiled == true) && pow2_tilesz) {
            unsigned int base_offset = (thread_rank() & (~(tilesz - 1)));
            unsigned int masklength = min((unsigned int)size() - base_offset, tilesz);
            unsigned int mask = (unsigned int)(-1) >> (32 - masklength);

            mask <<= (details::laneid() & ~(tilesz - 1));
            coalesced_group coalesced_tile = coalesced_group(mask);
            coalesced_tile._data.coalesced.metaGroupSize = size() / tilesz;
            coalesced_tile._data.coalesced.metaGroupRank = thread_rank() / tilesz;
            coalesced_tile._data.coalesced.is_tiled = true;
            return (coalesced_tile);
        }
        else if ((_data.coalesced.is_tiled == false) && pow2_tilesz) {
            unsigned int mask = 0;
            unsigned int member_rank = 0;
            int seen_lanes = (thread_rank() / tilesz) * tilesz;
            for (unsigned int bit_idx = 0; bit_idx < 32; bit_idx++) {
                unsigned int lane_bit = _data.coalesced.mask & (1 << bit_idx);
                if (lane_bit) {
                    if (seen_lanes <= 0 && member_rank < tilesz) {
                        mask |= lane_bit;
                        member_rank++;
                    }
                    seen_lanes--;
                }
            }
            coalesced_group coalesced_tile = coalesced_group(mask);
            // Override parent with the size of this group
            coalesced_tile._data.coalesced.metaGroupSize = (size() + tilesz - 1) / tilesz;
            coalesced_tile._data.coalesced.metaGroupRank = thread_rank() / tilesz;
            return coalesced_tile;
        }
        else {
            // None in _CG_VERSION 1000
            details::abort();
        }

        return (coalesced_group(0));
    }

 protected:
    _CG_QUALIFIER coalesced_group(unsigned int mask) {
        _data.coalesced.mask = mask;
        _data.coalesced.size = __popc(mask);
        _data.coalesced.metaGroupRank = 0;
        _data.coalesced.metaGroupSize = 1;
        _data.coalesced.is_tiled = false;
    }

    _CG_QUALIFIER unsigned int get_mask() const {
        return (_data.coalesced.mask);
    }

 public:
    _CG_THREAD_SCOPE(cuda::thread_scope::thread_scope_block)

    _CG_QUALIFIER unsigned int size() const {
        return (_data.coalesced.size);
    }

    _CG_QUALIFIER unsigned int thread_rank() const {
        return (__popc(_data.coalesced.mask & details::lanemask32_lt()));
    }

    // Rank of this group in the upper level of the hierarchy
    _CG_QUALIFIER unsigned int meta_group_rank() const {
        return _data.coalesced.metaGroupRank;
    }

    // Total num partitions created out of all CTAs when the group was created
    _CG_QUALIFIER unsigned int meta_group_size() const {
        return _data.coalesced.metaGroupSize;
    }

    _CG_QUALIFIER void sync() const {
        __syncwarp(_data.coalesced.mask);
    }

#ifdef _CG_CPP11_FEATURES
    template <typename TyElem, typename TyRet = details::remove_qual<TyElem>>
    _CG_QUALIFIER TyRet shfl(TyElem&& elem, int srcRank) const {
        unsigned int lane = (srcRank == 0) ? __ffs(_data.coalesced.mask) - 1 :
            (size() == 32) ? srcRank : __fns(_data.coalesced.mask, 0, (srcRank + 1));

        return details::tile::shuffle_dispatch<TyElem>::shfl(
            _CG_STL_NAMESPACE::forward<TyElem>(elem), _data.coalesced.mask, lane, 32);
    }

    template <typename TyElem, typename TyRet = details::remove_qual<TyElem>>
    _CG_QUALIFIER TyRet shfl_down(TyElem&& elem, unsigned int delta) const {
        if (size() == 32) {
            return details::tile::shuffle_dispatch<TyElem>::shfl_down(
                _CG_STL_NAMESPACE::forward<TyElem>(elem), 0xFFFFFFFF, delta, 32);
        }

        unsigned int lane = __fns(_data.coalesced.mask, details::laneid(), delta + 1);

        if (lane >= 32)
            lane = details::laneid();

        return details::tile::shuffle_dispatch<TyElem>::shfl(
            _CG_STL_NAMESPACE::forward<TyElem>(elem), _data.coalesced.mask, lane, 32);
    }

    template <typename TyElem, typename TyRet = details::remove_qual<TyElem>>
    _CG_QUALIFIER TyRet shfl_up(TyElem&& elem, int delta) const {
        if (size() == 32) {
            return details::tile::shuffle_dispatch<TyElem>::shfl_up(
                _CG_STL_NAMESPACE::forward<TyElem>(elem), 0xFFFFFFFF, delta, 32);
        }

        unsigned lane = __fns(_data.coalesced.mask, details::laneid(), -(delta + 1));
        if (lane >= 32)
            lane = details::laneid();

        return details::tile::shuffle_dispatch<TyElem>::shfl(
            _CG_STL_NAMESPACE::forward<TyElem>(elem), _data.coalesced.mask, lane, 32);
    }
#else
    template <typename TyIntegral>
    _CG_QUALIFIER TyIntegral shfl(TyIntegral var, unsigned int src_rank) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        unsigned int lane = (src_rank == 0) ? __ffs(_data.coalesced.mask) - 1 :
            (size() == 32) ? src_rank : __fns(_data.coalesced.mask, 0, (src_rank + 1));
        return (__shfl_sync(_data.coalesced.mask, var, lane, 32));
    }

    template <typename TyIntegral>
    _CG_QUALIFIER TyIntegral shfl_up(TyIntegral var, int delta) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        if (size() == 32) {
            return (__shfl_up_sync(0xFFFFFFFF, var, delta, 32));
        }
        unsigned lane = __fns(_data.coalesced.mask, details::laneid(), -(delta + 1));
        if (lane >= 32) lane = details::laneid();
        return (__shfl_sync(_data.coalesced.mask, var, lane, 32));
    }

    template <typename TyIntegral>
    _CG_QUALIFIER TyIntegral shfl_down(TyIntegral var, int delta) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        if (size() == 32) {
            return (__shfl_down_sync(0xFFFFFFFF, var, delta, 32));
        }
        unsigned int lane = __fns(_data.coalesced.mask, details::laneid(), delta + 1);
        if (lane >= 32) lane = details::laneid();
        return (__shfl_sync(_data.coalesced.mask, var, lane, 32));
    }
#endif

    _CG_QUALIFIER int any(int predicate) const {
        return (__ballot_sync(_data.coalesced.mask, predicate) != 0);
    }
    _CG_QUALIFIER int all(int predicate) const {
        return (__ballot_sync(_data.coalesced.mask, predicate) == _data.coalesced.mask);
    }
    _CG_QUALIFIER unsigned int ballot(int predicate) const {
        if (size() == 32) {
            return (__ballot_sync(0xFFFFFFFF, predicate));
        }
        unsigned int lane_ballot = __ballot_sync(_data.coalesced.mask, predicate);
        return (_packLanes(lane_ballot));
    }

#ifdef _CG_HAS_MATCH_COLLECTIVE

    template <typename TyIntegral>
    _CG_QUALIFIER unsigned int match_any(TyIntegral val) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        if (size() == 32) {
            return (__match_any_sync(0xFFFFFFFF, val));
        }
        unsigned int lane_match = __match_any_sync(_data.coalesced.mask, val);
        return (_packLanes(lane_match));
    }

    template <typename TyIntegral>
    _CG_QUALIFIER unsigned int match_all(TyIntegral val, int &pred) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        if (size() == 32) {
            return (__match_all_sync(0xFFFFFFFF, val, &pred));
        }
        unsigned int lane_match = __match_all_sync(_data.coalesced.mask, val, &pred);
        return (_packLanes(lane_match));
    }

#endif /* !_CG_HAS_MATCH_COLLECTIVE */

};

_CG_QUALIFIER coalesced_group coalesced_threads()
{
    return (coalesced_group(__activemask()));
}

namespace details {
    template <unsigned int Size> struct verify_thread_block_tile_size;
    template <> struct verify_thread_block_tile_size<32> { typedef void OK; };
    template <> struct verify_thread_block_tile_size<16> { typedef void OK; };
    template <> struct verify_thread_block_tile_size<8>  { typedef void OK; };
    template <> struct verify_thread_block_tile_size<4>  { typedef void OK; };
    template <> struct verify_thread_block_tile_size<2>  { typedef void OK; };
    template <> struct verify_thread_block_tile_size<1>  { typedef void OK; };

#ifdef _CG_CPP11_FEATURES
    template <unsigned int Size>
    using _is_power_of_2 = _CG_STL_NAMESPACE::integral_constant<bool, (Size & (Size - 1)) == 0>;

    template <unsigned int Size>
    using _is_single_warp = _CG_STL_NAMESPACE::integral_constant<bool, Size <= 32>;
    template <unsigned int Size>
    using _is_multi_warp =
    _CG_STL_NAMESPACE::integral_constant<bool, (Size > 32) && (Size < 1024)>;

    template <unsigned int Size>
    using _is_valid_single_warp_tile =
        _CG_STL_NAMESPACE::integral_constant<bool, _is_power_of_2<Size>::value && _is_single_warp<Size>::value>;
    template <unsigned int Size>
    using _is_valid_multi_warp_tile =
        _CG_STL_NAMESPACE::integral_constant<bool, _is_power_of_2<Size>::value && _is_multi_warp<Size>::value>;
#else
    template <unsigned int Size>
    struct _is_multi_warp {
        static const bool value = false;
    };
#endif
}

template <unsigned int Size>
class __static_size_tile_base
{
protected:
    _CG_STATIC_CONST_DECL unsigned int numThreads = Size;

public:
    _CG_THREAD_SCOPE(cuda::thread_scope::thread_scope_block)

    // Rank of thread within tile
    _CG_STATIC_QUALIFIER unsigned int thread_rank() {
        return (details::cta::thread_rank() & (numThreads - 1));
    }

    // Number of threads within tile
    _CG_STATIC_QUALIFIER unsigned int size() {
        return (numThreads);
    }
};

template <unsigned int Size>
class __static_size_thread_block_tile_base : public __static_size_tile_base<Size>
{
    friend class details::_coalesced_group_data_access;
    typedef details::tile::tile_helpers<Size> th;

#ifdef _CG_CPP11_FEATURES
    static_assert(details::_is_valid_single_warp_tile<Size>::value, "Size must be one of 1/2/4/8/16/32");
#else
    typedef typename details::verify_thread_block_tile_size<Size>::OK valid;
#endif
    using __static_size_tile_base<Size>::numThreads;
    _CG_STATIC_CONST_DECL unsigned int fullMask = 0xFFFFFFFF;

 protected:
    _CG_STATIC_QUALIFIER unsigned int build_mask() {
        unsigned int mask = fullMask;
        if (numThreads != 32) {
            // [0,31] representing the current active thread in the warp
            unsigned int laneId = details::laneid();
            // shift mask according to the partition it belongs to
            mask = th::tileMask << (laneId & ~(th::laneMask));
        }
        return (mask);
    }

public:
    _CG_STATIC_QUALIFIER void sync() {
        __syncwarp(build_mask());
    }

#ifdef _CG_CPP11_FEATURES
    // PTX supported collectives
    template <typename TyElem, typename TyRet = details::remove_qual<TyElem>>
    _CG_QUALIFIER TyRet shfl(TyElem&& elem, int srcRank) const {
        return details::tile::shuffle_dispatch<TyElem>::shfl(
            _CG_STL_NAMESPACE::forward<TyElem>(elem), build_mask(), srcRank, numThreads);
    }

    template <typename TyElem, typename TyRet = details::remove_qual<TyElem>>
    _CG_QUALIFIER TyRet shfl_down(TyElem&& elem, unsigned int delta) const {
        return details::tile::shuffle_dispatch<TyElem>::shfl_down(
            _CG_STL_NAMESPACE::forward<TyElem>(elem), build_mask(), delta, numThreads);
    }

    template <typename TyElem, typename TyRet = details::remove_qual<TyElem>>
    _CG_QUALIFIER TyRet shfl_up(TyElem&& elem, unsigned int delta) const {
        return details::tile::shuffle_dispatch<TyElem>::shfl_up(
            _CG_STL_NAMESPACE::forward<TyElem>(elem), build_mask(), delta, numThreads);
    }

    template <typename TyElem, typename TyRet = details::remove_qual<TyElem>>
    _CG_QUALIFIER TyRet shfl_xor(TyElem&& elem, unsigned int laneMask) const {
        return details::tile::shuffle_dispatch<TyElem>::shfl_xor(
            _CG_STL_NAMESPACE::forward<TyElem>(elem), build_mask(), laneMask, numThreads);
    }
#else
    template <typename TyIntegral>
    _CG_QUALIFIER TyIntegral shfl(TyIntegral var, int srcRank) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        return (__shfl_sync(build_mask(), var, srcRank, numThreads));
    }

    template <typename TyIntegral>
    _CG_QUALIFIER TyIntegral shfl_down(TyIntegral var, unsigned int delta) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        return (__shfl_down_sync(build_mask(), var, delta, numThreads));
    }

    template <typename TyIntegral>
    _CG_QUALIFIER TyIntegral shfl_up(TyIntegral var, unsigned int delta) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        return (__shfl_up_sync(build_mask(), var, delta, numThreads));
    }

    template <typename TyIntegral>
    _CG_QUALIFIER TyIntegral shfl_xor(TyIntegral var, unsigned int laneMask) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        return (__shfl_xor_sync(build_mask(), var, laneMask, numThreads));
    }
#endif //_CG_CPP11_FEATURES

    _CG_QUALIFIER int any(int predicate) const {
        unsigned int lane_ballot = build_mask() & __ballot_sync(build_mask(), predicate);
        return (lane_ballot != 0);
    }
    _CG_QUALIFIER int all(int predicate) const {
        unsigned int lane_ballot = build_mask() & __ballot_sync(build_mask(), predicate);
        return (lane_ballot == build_mask());
    }
    _CG_QUALIFIER unsigned int ballot(int predicate) const {
        unsigned int lane_ballot = build_mask() & __ballot_sync(build_mask(), predicate);
        return (lane_ballot >> (details::laneid() & (~(th::laneMask))));
    }

#ifdef _CG_HAS_MATCH_COLLECTIVE
    template <typename TyIntegral>
    _CG_QUALIFIER unsigned int match_any(TyIntegral val) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        unsigned int lane_match = build_mask() & __match_any_sync(build_mask(), val);
        return (lane_match >> (details::laneid() & (~(th::laneMask))));
    }

    template <typename TyIntegral>
    _CG_QUALIFIER unsigned int match_all(TyIntegral val, int &pred) const {
        details::assert_if_not_arithmetic<TyIntegral>();
        unsigned int lane_match = build_mask() & __match_all_sync(build_mask(), val, &pred);
        return (lane_match >> (details::laneid() & (~(th::laneMask))));
    }
#endif

};

template <unsigned int Size, typename ParentT>
class __static_parent_thread_block_tile_base
{
public:
    // Rank of this group in the upper level of the hierarchy
    _CG_STATIC_QUALIFIER unsigned int meta_group_rank() {
        return ParentT::thread_rank() / Size;
    }

    // Total num partitions created out of all CTAs when the group was created
    _CG_STATIC_QUALIFIER unsigned int meta_group_size() {
        return (ParentT::size() + Size - 1) / Size;
    }
};

/**
 * class thread_block_tile<unsigned int Size, ParentT = void>
 *
 * Statically-sized group type, representing one tile of a thread block.
 * The only specializations currently supported are those with native
 * hardware support (1/2/4/8/16/32)
 *
 * This group exposes warp-synchronous builtins.
 * Can only be constructed via tiled_partition<Size>(ParentT&)
 */

template <unsigned int Size, typename ParentT = void>
class __single_warp_thread_block_tile :
    public __static_size_thread_block_tile_base<Size>,
    public __static_parent_thread_block_tile_base<Size, ParentT>
{
    typedef __static_parent_thread_block_tile_base<Size, ParentT> staticParentBaseT;
    friend class details::_coalesced_group_data_access;

protected:
    _CG_QUALIFIER __single_warp_thread_block_tile() { };
    _CG_QUALIFIER __single_warp_thread_block_tile(unsigned int, unsigned int) { };

    _CG_STATIC_QUALIFIER unsigned int get_mask() {
        return __static_size_thread_block_tile_base<Size>::build_mask();
    }
};

template <unsigned int Size>
class __single_warp_thread_block_tile<Size, void> :
    public __static_size_thread_block_tile_base<Size>,
    public thread_group_base<coalesced_group_id>
{
    _CG_STATIC_CONST_DECL unsigned int numThreads = Size;

    template <unsigned int, typename ParentT> friend class __single_warp_thread_block_tile;
    friend class details::_coalesced_group_data_access;

    typedef __static_size_thread_block_tile_base<numThreads> staticSizeBaseT;

protected:
    _CG_QUALIFIER __single_warp_thread_block_tile(unsigned int meta_group_rank, unsigned int meta_group_size) {
        _data.coalesced.mask = staticSizeBaseT::build_mask();
        _data.coalesced.size = numThreads;
        _data.coalesced.metaGroupRank = meta_group_rank;
        _data.coalesced.metaGroupSize = meta_group_size;
        _data.coalesced.is_tiled = true;
    }

    _CG_QUALIFIER unsigned int get_mask() const {
        return (_data.coalesced.mask);
    }

public:
    using staticSizeBaseT::sync;
    using staticSizeBaseT::size;
    using staticSizeBaseT::thread_rank;

    _CG_QUALIFIER unsigned int meta_group_rank() const {
        return _data.coalesced.metaGroupRank;
    }

    _CG_QUALIFIER unsigned int meta_group_size() const {
        return _data.coalesced.metaGroupSize;
    }
};

/**
 * Outer level API calls
 * void sync(GroupT) - see <group_type>.sync()
 * void thread_rank(GroupT) - see <group_type>.thread_rank()
 * void group_size(GroupT) - see <group_type>.size()
 */
template <class GroupT>
_CG_QUALIFIER void sync(GroupT const &g)
{
    g.sync();
}

// TODO: Use a static dispatch to determine appropriate return type
// C++03 is stuck with unsigned long long for now
#ifdef _CG_CPP11_FEATURES
template <class GroupT>
_CG_QUALIFIER auto thread_rank(GroupT const& g) -> decltype(g.thread_rank()) {
    return g.thread_rank();
}


template <class GroupT>
_CG_QUALIFIER auto group_size(GroupT const &g) -> decltype(g.size()) {
    return g.size();
}
#else
template <class GroupT>
_CG_QUALIFIER unsigned long long thread_rank(GroupT const& g) {
    return static_cast<unsigned long long>(g.thread_rank());
}


template <class GroupT>
_CG_QUALIFIER unsigned long long group_size(GroupT const &g) {
    return static_cast<unsigned long long>(g.size());
}
#endif


/**
 * tiled_partition
 *
 * The tiled_partition(parent, tilesz) method is a collective operation that
 * partitions the parent group into a one-dimensional, row-major, tiling of subgroups.
 *
 * A total of ((size(parent)+tilesz-1)/tilesz) subgroups will
 * be created where threads having identical k = (thread_rank(parent)/tilesz)
 * will be members of the same subgroup.
 *
 * The implementation may cause the calling thread to wait until all the members
 * of the parent group have invoked the operation before resuming execution.
 *
 * Functionality is limited to power-of-two sized subgorup instances of at most
 * 32 threads. Only thread_block, thread_block_tile<>, and their subgroups can be
 * tiled_partition() in _CG_VERSION 1000.
 */
_CG_QUALIFIER thread_group tiled_partition(const thread_group& parent, unsigned int tilesz)
{
    if (parent.get_type() == coalesced_group_id) {
        const coalesced_group *_cg = static_cast<const coalesced_group*>(&parent);
        return _cg->_get_tiled_threads(tilesz);
    }
    else {
        const thread_block *_tb = static_cast<const thread_block*>(&parent);
        return _tb->_get_tiled_threads(tilesz);
    }
}

// Thread block type overload: returns a basic thread_group for now (may be specialized later)
_CG_QUALIFIER thread_group tiled_partition(const thread_block& parent, unsigned int tilesz)
{
    return (parent._get_tiled_threads(tilesz));
}

// Coalesced group type overload: retains its ability to stay coalesced
_CG_QUALIFIER coalesced_group tiled_partition(const coalesced_group& parent, unsigned int tilesz)
{
    return (parent._get_tiled_threads(tilesz));
}

namespace details {
    template <unsigned int Size, typename ParentT>
    class internal_thread_block_tile : public __single_warp_thread_block_tile<Size, ParentT> {};

    template <unsigned int Size, typename ParentT>
    _CG_QUALIFIER internal_thread_block_tile<Size, ParentT> tiled_partition_internal() {
        return internal_thread_block_tile<Size, ParentT>();
    }

    template <typename TyVal, typename GroupT, typename WarpLambda, typename InterWarpLambda>
    TyVal _CG_QUALIFIER multi_warp_reduction_helper(
            const GroupT& group,
            WarpLambda warp_lambda,
            InterWarpLambda inter_warp_lambda);
}
/**
 * tiled_partition<tilesz>
 *
 * The tiled_partition<tilesz>(parent) method is a collective operation that
 * partitions the parent group into a one-dimensional, row-major, tiling of subgroups.
 *
 * A total of ((size(parent)/tilesz) subgroups will be created,
 * therefore the parent group size must be evenly divisible by the tilesz.
 * The allow parent groups are thread_block or thread_block_tile<size>.
 *
 * The implementation may cause the calling thread to wait until all the members
 * of the parent group have invoked the operation before resuming execution.
 *
 * Functionality is limited to native hardware sizes, 1/2/4/8/16/32.
 * The size(parent) must be greater than the template Size parameter
 * otherwise the results are undefined.
 */

#if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
template <unsigned int Size>
class __static_size_multi_warp_tile_base : public __static_size_tile_base<Size>
{
    static_assert(details::_is_valid_multi_warp_tile<Size>::value, "Size must be one of 64/128/256/512");

    template <typename TyVal, typename GroupT, typename WarpLambda, typename InterWarpLambda>
    friend TyVal details::multi_warp_reduction_helper(
            const GroupT& group,
            WarpLambda warp_lambda,
            InterWarpLambda inter_warp_lambda);
    template <unsigned int OtherSize>
    friend class __static_size_multi_warp_tile_base;
    using WarpType = details::internal_thread_block_tile<32, __static_size_multi_warp_tile_base<Size>>;
    using ThisType = __static_size_multi_warp_tile_base<Size>;
    _CG_STATIC_CONST_DECL int numWarps = Size / 32;

protected:
    details::multi_warp_scratch* const tile_memory;

    template <typename GroupT>
    _CG_QUALIFIER __static_size_multi_warp_tile_base(const GroupT& g) :
            tile_memory(g.tile_memory) {}


private:
    _CG_QUALIFIER unsigned int get_communication_size() const {
        return tile_memory->communication_size;
    }

    _CG_QUALIFIER unsigned int get_max_block_size() const {
        return tile_memory->max_block_size;
    }

    _CG_QUALIFIER unsigned int* get_sync_location() const {
        // First two elements are not used, for barriers, but to store config.
        unsigned int sync_id = (details::cta::thread_rank() / Size + (get_max_block_size() / Size) - 2);
        return &(reinterpret_cast<unsigned int*>(tile_memory->memory)[sync_id]);
    }

    template <typename T>
    _CG_QUALIFIER T* get_reduction_location(unsigned int warp_id) const {
        unsigned int sync_mem_size = (get_max_block_size() / 32 - 2) * sizeof(unsigned int);
        unsigned int reduction_id = (details::cta::thread_rank() - thread_rank()) / 32 + warp_id;
        return reinterpret_cast<T*>(&tile_memory->memory[sync_mem_size + reduction_id * get_communication_size()]);
    }

    template <typename T>
    _CG_QUALIFIER T* get_reduction_location() const {
        unsigned int sync_mem_size = (get_max_block_size() / 32 - 2) * sizeof(unsigned int);
        unsigned int reduction_id = details::cta::thread_rank() / 32;
        return reinterpret_cast<T*>(&tile_memory->memory[sync_mem_size + reduction_id * get_communication_size()]);
    }

    template <typename TyVal>
    _CG_QUALIFIER TyVal shfl_impl(TyVal val, unsigned int src) const {
        unsigned int src_warp = src / 32;
        auto warp = details::tiled_partition_internal<32, ThisType>();
        unsigned int* sync_location = get_sync_location();

        // Get warp slot of the source threads warp.
        TyVal* warp_reduction_location = get_reduction_location<TyVal>(src_warp);

        if (warp.meta_group_rank() == src_warp) {
            // Put shuffled value into my warp slot and let my warp arrive at the barrier.
            if (thread_rank() == src) {
                *warp_reduction_location = val;
            }
            unsigned int sync_ticket =
                details::sync_warps_arrive(numWarps, sync_location, thread_rank());
            TyVal result = *warp_reduction_location;
            details::sync_warps_wait(sync_location, sync_ticket, thread_rank());
            return result;
        }
        else {
            // Wait for the source warp to arrive on the barrier.
            details::sync_warps_wait_for_count(1, sync_location);
            TyVal result = *warp_reduction_location;
            details::sync_warps(numWarps, sync_location, thread_rank());
            return result;
        }
    }

    template <typename TyVal>
    _CG_QUALIFIER TyVal shfl_large_impl(TyVal val, unsigned int src) const {
        auto warp = details::tiled_partition_internal<32, ThisType>();

        details::copy_channel<numWarps> broadcast_channel{
            get_reduction_location<char>(0),
            get_sync_location(),
            get_communication_size() * numWarps};

        if (warp.meta_group_rank() == src / 32) {
#if _CG_ITERATIVE_BROADCAST_NO_SHFL
            unsigned int src_mask = 1 << (src % 32);
            if (thread_rank() == src) {
                broadcast_channel.template send_value<TyVal, 1, decltype(broadcast_channel)::send_one_to_many>(val, 0, true, src_mask);
            }
            else {
                broadcast_channel.template receive_value<TyVal>(val, warp.thread_rank() == ((src % 32) == 0 ? 1 : 0), true, ~src_mask);
            }
#else
            val = warp.shfl(val, src);
            broadcast_channel.template send_value<TyVal, 32, decltype(broadcast_channel)::send_many_to_many>(val, warp.thread_rank());
#endif
        }
        else {
            broadcast_channel.template receive_value<TyVal>(val, warp.thread_rank() == 0);
        }
        sync();
        return val;
    }

    template <typename TyVal, typename WarpLambda, typename InterWarpLambda>
    _CG_QUALIFIER TyVal reduction_scheme_impl(const WarpLambda& warp_lambda, const InterWarpLambda& inter_warp_lambda) const {
        auto warp = details::tiled_partition_internal<32, ThisType>();
        unsigned int* sync_location = get_sync_location();
        TyVal* warp_reduction_location = get_reduction_location<TyVal>();

        warp_lambda(warp, warp_reduction_location);

        if (details::sync_warps_last_releases(numWarps, sync_location, thread_rank())) {
            auto reduction_threads = details::tiled_partition_internal<numWarps, decltype(warp)>();
            if (reduction_threads.meta_group_rank() == 0) {
                TyVal* thread_reduction_location = get_reduction_location<TyVal>(reduction_threads.thread_rank());
                inter_warp_lambda(reduction_threads, thread_reduction_location);
            }
            warp.sync();
            details::sync_warps_release(sync_location, warp.thread_rank() == 0);
        }
        TyVal result = *warp_reduction_location;
        warp.sync();  // Added warpsync, if all collectives do sync before writing to reduce_location (they does right now),
                      // we could delete it.
        return result;
    }

    template <typename TyVal, typename WarpLambda, typename InterWarpLambda>
    _CG_QUALIFIER TyVal reduction_scheme_large_impl(
            const WarpLambda& warp_lambda,
            const InterWarpLambda& inter_warp_lambda) const {
        auto warp = details::tiled_partition_internal<32, ThisType>();
        unsigned int* sync_location = get_sync_location();
        details::copy_channel<numWarps> final_result_channel{
            get_reduction_location<char>(0),
            sync_location,
            get_communication_size() * numWarps};

        TyVal warp_result;
        warp_lambda(warp, &warp_result);

        if (warp.meta_group_rank() == 0) {
            auto reduction_threads = details::tiled_partition_internal<numWarps, decltype(warp)>();
            details::copy_channel<numWarps> partial_results_channel{
                get_reduction_location<char>(reduction_threads.thread_rank()),
                sync_location,
                get_communication_size()};

            partial_results_channel.template receive_value<TyVal>(
                    warp_result,
                    warp.thread_rank() == 0,
                    reduction_threads.thread_rank() != 0);
            inter_warp_lambda(reduction_threads, &warp_result);
            final_result_channel.template send_value<TyVal, 32, decltype(final_result_channel)::send_many_to_many>(
                    warp_result,
                    warp.thread_rank());
        }
        else {
            details::copy_channel<numWarps> partial_results_channel{get_reduction_location<char>(), sync_location, get_communication_size()};
            partial_results_channel.template send_value<TyVal, 32, decltype(partial_results_channel)::send_many_to_one>(
                    warp_result,
                    warp.thread_rank());
            final_result_channel.template receive_value<TyVal>(warp_result, warp.thread_rank() == 0);
        }
        sync();
        return warp_result;
    }

    template <typename TyVal, typename WarpLambda, typename InterWarpLambda>
    _CG_QUALIFIER TyVal reduction_scheme(const WarpLambda& warp_lambda, const InterWarpLambda& inter_warp_lambda) const {
        if (sizeof(TyVal) > get_communication_size()) {
            return reduction_scheme_large_impl<TyVal, WarpLambda, InterWarpLambda>(warp_lambda, inter_warp_lambda);
        }
        else {
            return reduction_scheme_impl<TyVal, WarpLambda, InterWarpLambda>(warp_lambda, inter_warp_lambda);
        }
    }

public:
    using __static_size_tile_base<Size>::thread_rank;

    template <typename TyVal>
    _CG_QUALIFIER TyVal shfl(TyVal val, unsigned int src) const {
        if (sizeof(TyVal) > get_communication_size()) {
            return shfl_large_impl(val, src);
        }
        else {
            return shfl_impl(val, src);
        }
    }

   _CG_QUALIFIER void sync() const {
        details::sync_warps(Size / 32, get_sync_location(), __static_size_tile_base<Size>::thread_rank());
    }

    _CG_QUALIFIER int any(int predicate) const {
        auto warp_lambda = [=] (WarpType& warp, int* warp_reduction_location) {
                *warp_reduction_location = __any_sync(0xFFFFFFFF, predicate);
        };
        auto inter_warp_lambda =
            [] (details::internal_thread_block_tile<numWarps, WarpType>& reduction_threads, int* thread_reduction_location) {
                *thread_reduction_location = __any_sync(0xFFFFFFFFU >> (32 - numWarps), *thread_reduction_location);
        };
        return reduction_scheme<int>(warp_lambda, inter_warp_lambda);
    }

    _CG_QUALIFIER int all(int predicate) const {
        auto warp_lambda = [=] (WarpType& warp, int* warp_reduction_location) {
                *warp_reduction_location = __all_sync(0xFFFFFFFF, predicate);
        };
        auto inter_warp_lambda =
            [] (details::internal_thread_block_tile<numWarps, WarpType>& reduction_threads, int* thread_reduction_location) {
                *thread_reduction_location = __all_sync(0xFFFFFFFFU >> (32 - numWarps), *thread_reduction_location);
        };
        return reduction_scheme<int>(warp_lambda, inter_warp_lambda);
    }
};


template <unsigned int Size, typename ParentT = void>
class __multi_warp_thread_block_tile :
    public __static_size_multi_warp_tile_base<Size>,
    public __static_parent_thread_block_tile_base<Size, ParentT>
{
    typedef __static_parent_thread_block_tile_base<Size, ParentT> staticParentBaseT;
    typedef __static_size_multi_warp_tile_base<Size> staticTileBaseT;
protected:
    _CG_QUALIFIER __multi_warp_thread_block_tile(const ParentT& g) :
        __static_size_multi_warp_tile_base<Size>(g) {}
};

template <unsigned int Size>
class __multi_warp_thread_block_tile<Size, void> : public __static_size_multi_warp_tile_base<Size>
{
    const unsigned int metaGroupRank;
    const unsigned int metaGroupSize;

protected:
    template <unsigned int OtherSize, typename ParentT>
    _CG_QUALIFIER __multi_warp_thread_block_tile(const __multi_warp_thread_block_tile<OtherSize, ParentT>& g) :
        __static_size_multi_warp_tile_base<Size>(g), metaGroupRank(g.meta_group_rank()), metaGroupSize(g.meta_group_size()) {}

public:
    _CG_QUALIFIER unsigned int meta_group_rank() const {
        return metaGroupRank;
    }

    _CG_QUALIFIER unsigned int meta_group_size() const {
        return metaGroupSize;
    }
};
#endif

template <unsigned int Size, typename ParentT = void>
class thread_block_tile;

namespace details {
    template <unsigned int Size, typename ParentT, bool IsMultiWarp>
    class thread_block_tile_impl;

    template <unsigned int Size, typename ParentT>
    class thread_block_tile_impl<Size, ParentT, false>: public __single_warp_thread_block_tile<Size, ParentT>
    {
    protected:
        template <unsigned int OtherSize, typename OtherParentT, bool OtherIsMultiWarp>
        _CG_QUALIFIER thread_block_tile_impl(const thread_block_tile_impl<OtherSize, OtherParentT, OtherIsMultiWarp>& g) :
            __single_warp_thread_block_tile<Size, ParentT>(g.meta_group_rank(), g.meta_group_size()) {}

        _CG_QUALIFIER thread_block_tile_impl(const thread_block& g) :
            __single_warp_thread_block_tile<Size, ParentT>() {}
    };

#if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
    template <unsigned int Size, typename ParentT>
    class thread_block_tile_impl<Size, ParentT, true> : public __multi_warp_thread_block_tile<Size, ParentT>
    {
        protected:
        template <typename GroupT>
        _CG_QUALIFIER thread_block_tile_impl(const GroupT& g) :
            __multi_warp_thread_block_tile<Size, ParentT>(g) {}
    };
#endif
}

template <unsigned int Size, typename ParentT>
class thread_block_tile : public details::thread_block_tile_impl<Size, ParentT, details::_is_multi_warp<Size>::value>
{
protected:
    _CG_QUALIFIER thread_block_tile(const ParentT& g) :
        details::thread_block_tile_impl<Size, ParentT, details::_is_multi_warp<Size>::value>(g) {}

public:
    _CG_QUALIFIER operator thread_block_tile<Size, void>() const {
        return thread_block_tile<Size, void>(*this);
    }
};

template <unsigned int Size>
class thread_block_tile<Size, void> : public details::thread_block_tile_impl<Size, void, details::_is_multi_warp<Size>::value>
{
    template <unsigned int, typename ParentT>
    friend class thread_block_tile;

protected:
    template <unsigned int OtherSize, typename OtherParentT>
    _CG_QUALIFIER thread_block_tile(const thread_block_tile<OtherSize, OtherParentT>& g) :
        details::thread_block_tile_impl<Size, void, details::_is_multi_warp<Size>::value>(g) {}

public:
    template <typename ParentT>
    _CG_QUALIFIER thread_block_tile(const thread_block_tile<Size, ParentT>& g) :
        details::thread_block_tile_impl<Size, void, details::_is_multi_warp<Size>::value>(g) {}
};

namespace details {
    template <typename TyVal, typename GroupT, typename WarpLambda, typename InterWarpLambda>
    TyVal _CG_QUALIFIER multi_warp_reduction_helper(
            const GroupT& group,
            WarpLambda warp_lambda,
            InterWarpLambda inter_warp_lambda) {
        return group.template reduction_scheme<TyVal>(warp_lambda, inter_warp_lambda);
    }

    template <unsigned int Size, typename ParentT>
    struct tiled_partition_impl;

    template <unsigned int Size>
    struct tiled_partition_impl<Size, thread_block> : public thread_block_tile<Size, thread_block> {
        _CG_QUALIFIER tiled_partition_impl(const thread_block& g) :
            thread_block_tile<Size, thread_block>(g) {}
    };

    // ParentT = static thread_block_tile<ParentSize, GrandParent> specialization
    template <unsigned int Size, unsigned int ParentSize, typename GrandParent>
    struct tiled_partition_impl<Size, thread_block_tile<ParentSize, GrandParent> > :
        public thread_block_tile<Size, thread_block_tile<ParentSize, GrandParent> > {
#ifdef _CG_CPP11_FEATURES
        static_assert(Size < ParentSize, "Tile size bigger or equal to the parent group size");
#endif
        _CG_QUALIFIER tiled_partition_impl(const thread_block_tile<ParentSize, GrandParent>& g) :
            thread_block_tile<Size, thread_block_tile<ParentSize, GrandParent> >(g) {}
    };

}

#if defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
namespace experimental {
    template <unsigned int Size, typename ParentT>
    _CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)
    {
        return details::tiled_partition_impl<Size, ParentT>(g);
    }

}
#endif

template <unsigned int Size, typename ParentT>
_CG_QUALIFIER thread_block_tile<Size, ParentT> tiled_partition(const ParentT& g)
{
#ifdef _CG_CPP11_FEATURES
    static_assert(details::_is_single_warp<Size>::value,
            "Tiled partition with Size > 32 supported only with experimental features enabled");
#endif
    return details::tiled_partition_impl<Size, ParentT>(g);
}

/**
 * <group_type>.sync()
 *
 * Executes a barrier across the group
 *
 * Implements both a compiler fence and an architectural fence to prevent,
 * memory reordering around the barrier.
 */
_CG_QUALIFIER void thread_group::sync() const
{
    switch (_data.group.type) {
    case coalesced_group_id:
        cooperative_groups::sync(*static_cast<const coalesced_group*>(this));
        break;
    case thread_block_id:
        cooperative_groups::sync(*static_cast<const thread_block*>(this));
        break;
    case grid_group_id:
        cooperative_groups::sync(*static_cast<const grid_group*>(this));
        break;
#if defined(_CG_HAS_MULTI_GRID_GROUP) && defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
    case multi_grid_group_id:
        cooperative_groups::sync(*static_cast<const multi_grid_group*>(this));
        break;
#endif
    default:
        break;
    }
}

/**
 * <group_type>.size()
 *
 * Returns the total number of threads in the group.
 */
_CG_QUALIFIER unsigned long long thread_group::size() const
{
    unsigned long long size = 0;
    switch (_data.group.type) {
    case coalesced_group_id:
        size = cooperative_groups::group_size(*static_cast<const coalesced_group*>(this));
        break;
    case thread_block_id:
        size = cooperative_groups::group_size(*static_cast<const thread_block*>(this));
        break;
    case grid_group_id:
        size = cooperative_groups::group_size(*static_cast<const grid_group*>(this));
        break;
#if defined(_CG_HAS_MULTI_GRID_GROUP) && defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
    case multi_grid_group_id:
        size = cooperative_groups::group_size(*static_cast<const multi_grid_group*>(this));
        break;
#endif
    default:
        break;
    }
    return size;
}

/**
 * <group_type>.thread_rank()
 *
 * Returns the linearized rank of the calling thread along the interval [0, size()).
 */
_CG_QUALIFIER unsigned long long thread_group::thread_rank() const
{
    unsigned long long rank = 0;
    switch (_data.group.type) {
    case coalesced_group_id:
        rank = cooperative_groups::thread_rank(*static_cast<const coalesced_group*>(this));
        break;
    case thread_block_id:
        rank = cooperative_groups::thread_rank(*static_cast<const thread_block*>(this));
        break;
    case grid_group_id:
        rank = cooperative_groups::thread_rank(*static_cast<const grid_group*>(this));
        break;
#if defined(_CG_HAS_MULTI_GRID_GROUP) && defined(_CG_CPP11_FEATURES) && defined(_CG_ABI_EXPERIMENTAL)
    case multi_grid_group_id:
        rank = cooperative_groups::thread_rank(*static_cast<const multi_grid_group*>(this));
        break;
#endif
    default:
        break;
    }
    return rank;
}

_CG_END_NAMESPACE

#include <cooperative_groups/details/partitioning.h>

# endif /* ! (__cplusplus, __CUDACC__) */

#endif /* !_COOPERATIVE_GROUPS_H_ */
