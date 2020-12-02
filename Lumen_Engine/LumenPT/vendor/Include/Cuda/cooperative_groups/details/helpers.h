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

#ifndef _COOPERATIVE_GROUPS_HELPERS_H_
# define _COOPERATIVE_GROUPS_HELPERS_H_

#include "info.h"
#include "sync.h"

_CG_BEGIN_NAMESPACE

namespace details {
#ifdef _CG_CPP11_FEATURES
    template <typename Ty> struct _is_float_or_half          : public _CG_STL_NAMESPACE::is_floating_point<Ty> {};
# ifdef _CG_HAS_FP16_COLLECTIVE
    template <>            struct _is_float_or_half<__half>  : public _CG_STL_NAMESPACE::true_type {};
    template <>            struct _is_float_or_half<__half2> : public _CG_STL_NAMESPACE::true_type {};
# endif
    template <typename Ty>
    using  is_float_or_half = _is_float_or_half<typename _CG_STL_NAMESPACE::remove_cv<Ty>::type>;
#endif

    template <typename TyTrunc>
    _CG_STATIC_QUALIFIER TyTrunc vec3_to_linear(dim3 index, dim3 nIndex) {
        return ((TyTrunc)index.z * nIndex.y * nIndex.x) +
               ((TyTrunc)index.y * nIndex.x) +
                (TyTrunc)index.x;
    }

#ifdef _CG_CPP11_FEATURES
    template <typename Ty>
    using remove_qual = typename _CG_STL_NAMESPACE::remove_cv<typename _CG_STL_NAMESPACE::remove_reference<Ty>::type>::type;
# endif

    class _coalesced_group_data_access {
    public:
        // Retrieve mask of coalesced groups
        template <typename TyGroup>
        _CG_STATIC_QUALIFIER unsigned int get_mask(const TyGroup &group) {
            return group.get_mask();
        }

        // Retrieve mask of tiles
        template <template <typename, typename> typename TyGroup, typename Sz, typename Parent>
        _CG_STATIC_QUALIFIER unsigned int get_mask(const TyGroup<Sz, Parent> &group) {
            return group.build_maks();
        }

        template <typename TyGroup>
        _CG_STATIC_QUALIFIER TyGroup construct_from_mask(unsigned int mask) {
            return TyGroup(mask);
        }

        template <typename TyGroup>
        _CG_STATIC_QUALIFIER void modify_meta_group(TyGroup &group, unsigned int mgRank, unsigned int mgSize) {
            group._data.coalesced.metaGroupRank = mgRank;
            group._data.coalesced.metaGroupSize = mgSize;
        }
    };

    namespace tile {
        template <unsigned int TileCount, unsigned int TileMask, unsigned int LaneMask, unsigned int ShiftCount>
        struct _tile_helpers{
            _CG_STATIC_CONST_DECL unsigned int tileCount = TileCount;
            _CG_STATIC_CONST_DECL unsigned int tileMask = TileMask;
            _CG_STATIC_CONST_DECL unsigned int laneMask = LaneMask;
            _CG_STATIC_CONST_DECL unsigned int shiftCount = ShiftCount;
        };

        template <unsigned int> struct tile_helpers;
        template <> struct tile_helpers<32> : public _tile_helpers<1,  0xFFFFFFFF, 0x1F, 5> {};
        template <> struct tile_helpers<16> : public _tile_helpers<2,  0x0000FFFF, 0x0F, 4> {};
        template <> struct tile_helpers<8>  : public _tile_helpers<4,  0x000000FF, 0x07, 3> {};
        template <> struct tile_helpers<4>  : public _tile_helpers<8,  0x0000000F, 0x03, 2> {};
        template <> struct tile_helpers<2>  : public _tile_helpers<16, 0x00000003, 0x01, 1> {};
        template <> struct tile_helpers<1>  : public _tile_helpers<32, 0x00000001, 0x00, 0> {};

#ifdef _CG_CPP11_FEATURES
        namespace shfl {
            /***********************************************************************************
             * Recursively Sliced Shuffle
             *  Purpose:
             *      Slices an input type a number of times into integral types so that shuffles
             *      are well defined
             *  Expectations:
             *      This object *should not* be used from a reinterpret_cast pointer unless
             *      some alignment guarantees can be met. Use a memcpy to guarantee that loads
             *      from the integral types stored within are aligned and correct.
             **********************************************************************************/
            template <unsigned int count, bool intSized = (count <= sizeof(int))>
            struct recursive_sliced_shuffle_helper;

            template <unsigned int count>
            struct recursive_sliced_shuffle_helper<count, true> {
                int val;

                template <typename TyFn>
                _CG_QUALIFIER void invoke_shuffle(const TyFn &shfl) {
                    val = shfl(val);
                }
            };

            template <unsigned int count>
            struct recursive_sliced_shuffle_helper<count, false> {
                int val;
                recursive_sliced_shuffle_helper<count - sizeof(int)> next;

                template <typename TyFn>
                _CG_QUALIFIER void invoke_shuffle(const TyFn &shfl) {
                    val = shfl(val);
                    next.invoke_shuffle(shfl);
                }
            };
        }

        struct _memory_shuffle {
            template <typename TyElem, typename TyShflFn>
            _CG_STATIC_QUALIFIER TyElem _shfl_internal(TyElem elem, const TyShflFn& fn) {
                static_assert(sizeof(TyElem) > 0, "in memory shuffle is not yet implemented");
                return TyElem{};
            }

            template <typename TyElem, typename TyRet = remove_qual<TyElem>>
            _CG_STATIC_QUALIFIER TyRet shfl(TyElem&& elem, unsigned int gMask, unsigned int srcRank, unsigned int threads) {
                auto shfl = [=](int val) -> int {
                    return 0;
                };

                return _shfl_internal<TyRet>(_CG_STL_NAMESPACE::forward<TyElem>(elem), shfl);
            }

            template <typename TyElem, typename TyRet = remove_qual<TyElem>>
            _CG_STATIC_QUALIFIER TyRet shfl_down(TyElem&& elem, unsigned int gMask, unsigned int delta, unsigned int threads) {
                auto shfl = [=](int val) -> int {
                    return 0;
                };

                return _shfl_internal<TyRet>(_CG_STL_NAMESPACE::forward<TyElem>(elem), shfl);
            }

            template <typename TyElem, typename TyRet = remove_qual<TyElem>>
            _CG_STATIC_QUALIFIER TyRet shfl_up(TyElem&& elem, unsigned int gMask, unsigned int delta, unsigned int threads) {
                auto shfl = [=](int val) -> int {
                    return 0;
                };

                return _shfl_internal<TyRet>(_CG_STL_NAMESPACE::forward<TyElem>(elem), shfl);
            }

            template <typename TyElem, typename TyRet = remove_qual<TyElem>>
            _CG_STATIC_QUALIFIER TyRet shfl_xor(TyElem&& elem, unsigned int gMask, unsigned int lMask, unsigned int threads) {
                auto shfl = [=](int val) -> int {
                    return 0;
                };

                return _shfl_internal<TyRet>(_CG_STL_NAMESPACE::forward<TyElem>(elem), shfl);
            }
        };

        /***********************************************************************************
         * Intrinsic Device Function Shuffle
         *  Purpose:
         *      Uses a shuffle helper that has characteristics best suited for moving
         *      elements between threads
         *  Expectations:
         *      Object given will be forced into an l-value type so that it can be used
         *      with a helper structure that reinterprets the data into intrinsic compatible
         *      types
         *  Notes:
         *      !! TyRet is required so that objects are returned by value and not as
         *      dangling references depending on the value category of the passed object
         **********************************************************************************/
        struct _intrinsic_compat_shuffle {
            template <unsigned int count>
            using shfl_helper = shfl::recursive_sliced_shuffle_helper<count>;

            template <typename TyElem, typename TyShflFn>
            _CG_STATIC_QUALIFIER TyElem _shfl_internal(TyElem elem, const TyShflFn& fn) {
                static_assert(__is_trivially_copyable(TyElem), "Type is not compatible with device shuffle");
                shfl_helper<sizeof(TyElem)> helper;
                memcpy(&helper, &elem, sizeof(TyElem));
                helper.invoke_shuffle(fn);
                memcpy(&elem, &helper, sizeof(TyElem));
                return elem;
            }

            template <typename TyElem, typename TyRet = remove_qual<TyElem>>
            _CG_STATIC_QUALIFIER TyRet shfl(TyElem&& elem, unsigned int gMask, unsigned int srcRank, unsigned int threads) {
                auto shfl = [=](int val) -> int {
                    return __shfl_sync(gMask, val, srcRank, threads);
                };

                return _shfl_internal<TyRet>(_CG_STL_NAMESPACE::forward<TyElem>(elem), shfl);
            }

            template <typename TyElem, typename TyRet = remove_qual<TyElem>>
            _CG_STATIC_QUALIFIER TyRet shfl_down(TyElem&& elem, unsigned int gMask, unsigned int delta, unsigned int threads) {
                auto shfl = [=](int val) -> int {
                    return __shfl_down_sync(gMask, val, delta, threads);
                };

                return _shfl_internal<TyRet>(_CG_STL_NAMESPACE::forward<TyElem>(elem), shfl);
            }

            template <typename TyElem, typename TyRet = remove_qual<TyElem>>
            _CG_STATIC_QUALIFIER TyRet shfl_up(TyElem&& elem, unsigned int gMask, unsigned int delta, unsigned int threads) {
                auto shfl = [=](int val) -> int {
                    return __shfl_up_sync(gMask, val, delta, threads);
                };

                return _shfl_internal<TyRet>(_CG_STL_NAMESPACE::forward<TyElem>(elem), shfl);
            }

            template <typename TyElem, typename TyRet = remove_qual<TyElem>>
            _CG_STATIC_QUALIFIER TyRet shfl_xor(TyElem&& elem, unsigned int gMask, unsigned int lMask, unsigned int threads) {
                auto shfl = [=](int val) -> int {
                    return __shfl_xor_sync(gMask, val, lMask, threads);
                };

                return _shfl_internal<TyRet>(_CG_STL_NAMESPACE::forward<TyElem>(elem), shfl);
            }
        };

        struct _native_shuffle {
            template <typename TyElem>
            _CG_STATIC_QUALIFIER TyElem shfl(
                    TyElem elem, unsigned int gMask, unsigned int srcRank, unsigned int threads) {
                return static_cast<TyElem>(__shfl_sync(gMask, elem, srcRank, threads));
            }

            template <typename TyElem>
            _CG_STATIC_QUALIFIER TyElem shfl_down(
                    TyElem elem, unsigned int gMask, unsigned int delta, unsigned int threads) {
                return static_cast<TyElem>(__shfl_down_sync(gMask, elem, delta, threads));
            }

            template <typename TyElem>
            _CG_STATIC_QUALIFIER TyElem shfl_up(
                    TyElem elem, unsigned int gMask, unsigned int delta, unsigned int threads) {
                return static_cast<TyElem>(__shfl_up_sync(gMask, elem, delta, threads));
            }

            template <typename TyElem>
            _CG_STATIC_QUALIFIER TyElem shfl_xor(
                    TyElem elem, unsigned int gMask, unsigned int lMask, unsigned int threads) {
                return static_cast<TyElem>(__shfl_xor_sync(gMask, elem, lMask, threads));
            }
        };

        // Almost all arithmetic types are supported by native shuffle
        // Vector types are the exception
        template <typename TyElem>
        using use_native_shuffle = _CG_STL_NAMESPACE::integral_constant<
            bool,
            _CG_STL_NAMESPACE::is_integral<
                remove_qual<TyElem>>::value ||
            details::is_float_or_half<
                remove_qual<TyElem>>::value
        >;

        constexpr unsigned long long _MemoryShuffleCutoff = 32;

        template <typename TyElem,
                  bool IsNative = use_native_shuffle<TyElem>::value,
                  bool InMem = (sizeof(TyElem) > _MemoryShuffleCutoff)>
        struct shuffle_dispatch;

        template <typename TyElem>
        struct shuffle_dispatch<TyElem, true, false> :  public _native_shuffle {};

        template <typename TyElem>
        struct shuffle_dispatch<TyElem, false, false> : public _intrinsic_compat_shuffle {};

        template <typename TyElem>
        struct shuffle_dispatch<TyElem, false, true> :  public _memory_shuffle {};

#endif //_CG_CPP11_FEATURES
    };

#ifdef _CG_CPP11_FEATURES
    template <unsigned int numWarps>
    struct copy_channel {
        char* channel_ptr;
        unsigned int* sync_location;
        size_t channel_size;

        // One thread sending to any all other threads, it has to wait for all warps (including its own).
        struct send_one_to_many {
            _CG_STATIC_CONST_DECL unsigned int wait_count = numWarps;
            _CG_STATIC_QUALIFIER void post_iter_release(unsigned int thread_idx, unsigned int* sync_location, unsigned int sync_mask) {
                __syncwarp(sync_mask);
                details::sync_warps_release(sync_location, thread_idx == 0);
            }
        };

        // One warp sending to all other warps, it has to wait for all other warps.
        struct send_many_to_many {
            _CG_STATIC_CONST_DECL unsigned int wait_count = numWarps - 1;
            _CG_STATIC_QUALIFIER void post_iter_release(unsigned int thread_idx, unsigned int* sync_location, unsigned int sync_mask) {
                __syncwarp(sync_mask);
                details::sync_warps_release(sync_location, thread_idx == 0);
            }
        };

        // One warp is receiving and all other warps are sending to that warp, they have to wait for that one warp.
        struct send_many_to_one {
            _CG_STATIC_CONST_DECL unsigned int wait_count = 1;
            _CG_STATIC_QUALIFIER void post_iter_release(unsigned int thread_idx, unsigned int* sync_location, unsigned int sync_mask) {
                // Wait for all warps to finish and let the last warp release all threads.
                if (details::sync_warps_last_releases(numWarps, sync_location, thread_idx)) {
                    details::sync_warps_release(sync_location, thread_idx == 0);
                }
            }
        };

        template <unsigned int ThreadCnt, size_t ValSize, typename SendDetails>
        _CG_QUALIFIER void _send_value_internal(char* val_ptr, unsigned int thread_idx, bool active, unsigned int sync_mask) {
            size_t thread_offset = thread_idx * sizeof(int);

            for (size_t i = 0; i < ValSize; i += channel_size) {
                size_t bytes_left = ValSize - i;
                size_t copy_chunk = min(bytes_left, channel_size);

                details::sync_warps_wait_for_count(SendDetails::wait_count, sync_location);
                if (active) {
                    for (size_t j = thread_offset; j < copy_chunk ; j += sizeof(int) * ThreadCnt) {
                        size_t my_bytes_left = copy_chunk - j;
                        memcpy(channel_ptr + j, val_ptr + i + j, min(my_bytes_left, sizeof(int)));
                    }
                }
                SendDetails::post_iter_release(thread_idx, sync_location, sync_mask);
            }
        }


        template <typename TyVal, unsigned int ThreadCnt, typename SendDetails>
        _CG_QUALIFIER void send_value(TyVal& val, unsigned int thread_idx, bool active = true, unsigned int sync_mask = 0xFFFFFFFF) {
            _send_value_internal<ThreadCnt, sizeof(TyVal), SendDetails>(reinterpret_cast<char*>(&val), thread_idx, active, sync_mask);
        }

        template <size_t ValSize>
        _CG_QUALIFIER void _receive_value_internal(char* val_ptr, bool warp_master, bool active, unsigned int sync_mask) {
            for (size_t i = 0; i < ValSize; i += channel_size) {
                size_t bytes_left = ValSize - i;
                details::sync_warps_wait_for_release(sync_location, warp_master, sync_mask);
                if (active) {
                    memcpy(val_ptr + i, channel_ptr, min(bytes_left, channel_size));
                }
            }
        }

        template <typename TyVal>
        _CG_QUALIFIER void receive_value(TyVal& val, bool warp_master, bool active = true, unsigned int sync_mask = 0xFFFFFFFF) {
            _receive_value_internal<sizeof(TyVal)>(reinterpret_cast<char*>(&val), warp_master, active, sync_mask);
        }
    };
#endif //_CG_CPP11_FEATURES

    namespace multi_grid {
        struct multi_grid_functions;
    };

    namespace grid {
        _CG_STATIC_QUALIFIER void sync(unsigned int *bar) {
            unsigned int expected = gridDim.x * gridDim.y * gridDim.z;

            details::sync_grids(expected, bar);
        }

        _CG_STATIC_QUALIFIER unsigned long long size()
        {
            // block.[yz] * grid.[yz] -> [max(65535) * max(~2048)] fits within 4b, promote after multiplication
            // block.x * grid.x -> [max(2^31-1) * max(~2048)] exceeds 4b, promote before multiplication
            return ((unsigned long long)(blockDim.z * gridDim.z)) *
                   ((unsigned long long)(blockDim.y * gridDim.y)) *
                   (((unsigned long long)blockDim.x * gridDim.x));
        }

        _CG_STATIC_QUALIFIER unsigned long long thread_rank()
        {
            unsigned long long blkIdx = vec3_to_linear<unsigned long long>(blockIdx, gridDim);
            return (blkIdx * (blockDim.x * blockDim.y * blockDim.z)) + vec3_to_linear<unsigned int>(threadIdx, blockDim);
        }

        _CG_STATIC_QUALIFIER dim3 grid_dim()
        {
            return (dim3(gridDim.x, gridDim.y, gridDim.z));
        }
    };


#if defined(_CG_HAS_MULTI_GRID_GROUP)

    namespace multi_grid {
        _CG_STATIC_QUALIFIER unsigned long long get_intrinsic_handle()
        {
            return (cudaCGGetIntrinsicHandle(cudaCGScopeMultiGrid));
        }

        _CG_STATIC_QUALIFIER void sync(const unsigned long long handle)
        {
            cudaError_t err = cudaCGSynchronize(handle, 0);
        }

        _CG_STATIC_QUALIFIER unsigned int size(const unsigned long long handle)
        {
            unsigned int numThreads = 0;
            cudaCGGetSize(&numThreads, NULL, handle);
            return numThreads;
        }

        _CG_STATIC_QUALIFIER unsigned int thread_rank(const unsigned long long handle)
        {
            unsigned int threadRank = 0;
            cudaCGGetRank(&threadRank, NULL, handle);
            return threadRank;
        }

        _CG_STATIC_QUALIFIER unsigned int grid_rank(const unsigned long long handle)
        {
            unsigned int gridRank = 0;
            cudaCGGetRank(NULL, &gridRank, handle);
            return gridRank;
        }

        _CG_STATIC_QUALIFIER unsigned int num_grids(const unsigned long long handle)
        {
            unsigned int numGrids = 0;
            cudaCGGetSize(NULL, &numGrids, handle);
            return numGrids;
        }

# ifdef _CG_CPP11_FEATURES
        struct multi_grid_functions {
            decltype(multi_grid::get_intrinsic_handle) *get_intrinsic_handle;
            decltype(multi_grid::sync) *sync;
            decltype(multi_grid::size) *size;
            decltype(multi_grid::thread_rank) *thread_rank;
            decltype(multi_grid::grid_rank) *grid_rank;
            decltype(multi_grid::num_grids) *num_grids;
        };

        template <typename = void>
        _CG_STATIC_QUALIFIER const multi_grid_functions* load_grid_intrinsics() {
            __constant__ static const multi_grid_functions mgf {
                &multi_grid::get_intrinsic_handle,
                &multi_grid::sync,
                &multi_grid::size,
                &multi_grid::thread_rank,
                &multi_grid::grid_rank,
                &multi_grid::num_grids
            };

            return &mgf;
        }
# endif
    };
#endif

    namespace cta {

        _CG_STATIC_QUALIFIER void sync()
        {
            __barrier_sync(0);
        }

        _CG_STATIC_QUALIFIER unsigned int size()
        {
            return static_cast<unsigned int>(blockDim.x * blockDim.y * blockDim.z);
        }

        _CG_STATIC_QUALIFIER  unsigned int thread_rank()
        {
            return static_cast<unsigned int>(vec3_to_linear<unsigned int>(threadIdx, blockDim));
        }

        _CG_STATIC_QUALIFIER dim3 group_index()
        {
            return (dim3(blockIdx.x, blockIdx.y, blockIdx.z));
        }

        _CG_STATIC_QUALIFIER dim3 thread_index()
        {
            return (dim3(threadIdx.x, threadIdx.y, threadIdx.z));
        }

        _CG_STATIC_QUALIFIER dim3 block_dim()
        {
            return (dim3(blockDim.x, blockDim.y, blockDim.z));
        }

    };

    _CG_STATIC_QUALIFIER unsigned int laneid()
    {
        unsigned int laneid;
        asm volatile("mov.u32 %0, %%laneid;" : "=r"(laneid));
        return laneid;
    }

    _CG_STATIC_QUALIFIER unsigned int warpsz()
    {
        unsigned int warpSize;
        asm volatile("mov.u32 %0, WARP_SZ;" : "=r"(warpSize));
        return warpSize;
    }

    _CG_STATIC_QUALIFIER unsigned int lanemask32_eq()
    {
        unsigned int lanemask32_eq;
        asm volatile("mov.u32 %0, %%lanemask_eq;" : "=r"(lanemask32_eq));
        return (lanemask32_eq);
    }

    _CG_STATIC_QUALIFIER unsigned int lanemask32_lt()
    {
        unsigned int lanemask32_lt;
        asm volatile("mov.u32 %0, %%lanemask_lt;" : "=r"(lanemask32_lt));
        return (lanemask32_lt);
    }

    _CG_STATIC_QUALIFIER void abort()
    {
        _CG_ABORT();
    }

    template <typename Ty>
    _CG_QUALIFIER void assert_if_not_arithmetic() {
#ifdef _CG_CPP11_FEATURES
        static_assert(
            _CG_STL_NAMESPACE::is_integral<Ty>::value ||
            details::is_float_or_half<Ty>::value,
            "Error: Ty is neither integer or float"
        );
#endif
    }

}; // !Namespace internal

_CG_END_NAMESPACE

#endif /* !_COOPERATIVE_GROUPS_HELPERS_H_ */
