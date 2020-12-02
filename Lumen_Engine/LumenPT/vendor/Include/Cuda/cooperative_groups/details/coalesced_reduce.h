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

#ifndef _CG_COALESCED_REDUCE_H_
#define _CG_COALESCED_REDUCE_H_

#include "info.h"
#include "helpers.h"
#include "partitioning.h"

_CG_BEGIN_NAMESPACE

namespace details {

template <typename TyVal, typename TyOp>
_CG_QUALIFIER auto shuffle_reduce_pow2(const coalesced_group& group, TyVal val, TyOp op) -> decltype(op(val, val)){
    using TyRet = decltype(op(val, val));
    TyRet out = val;

    for (int offset = group.size() >> 1; offset > 0; offset >>= 1)
        out = op(out, group.shfl_down(out, offset));

    return out;
}

template <typename TyVal, typename TyOp>
_CG_QUALIFIER auto coalesced_reduce(const coalesced_group& group, TyVal val, TyOp op) -> decltype(op(val, val)){
    using TyRet = decltype(op(val, val));
    const unsigned int groupSize = group.size();
    bool isPow2 = (groupSize & (groupSize-1)) == 0;
    TyRet out = val;

    // Normal shfl_down reduction if the group is a power of 2
    if (isPow2) {
        // Dispatch correct answer from lane 0 after performing the reduction
        return group.shfl(shuffle_reduce_pow2(group, val, op), 0);
    }
    else {
        const unsigned int mask = details::_coalesced_group_data_access::get_mask(group);
        unsigned int lanemask = details::lanemask32_lt() & mask;
        unsigned int srcLane = details::laneid();

        const unsigned int base = __ffs(mask)-1; /* lane with rank == 0 */
        const unsigned int rank = __popc(lanemask);

        for (unsigned int i = 1, j = 1; i < groupSize; i <<= 1) {
            if (i <= rank) {
                srcLane -= j;
                j = i; /* maximum possible lane */

                unsigned int begLane = base + rank - i; /* minimum possible lane */

                /*  Next source lane is in the range [ begLane .. srcLane ]
                    *  If begLane < srcLane then do a binary search.
                    */
                while (begLane < srcLane) {
                    const unsigned int halfLane = (begLane + srcLane) >> 1;
                    const unsigned int halfMask = lanemask >> halfLane;
                    const unsigned int d = __popc(halfMask);
                    if (d < i) {
                        srcLane = halfLane - 1; /* halfLane too large */
                    }
                    else if ((i < d) || !(halfMask & 0x01)) {
                        begLane = halfLane + 1; /* halfLane too small */
                    }
                    else {
                        begLane = srcLane = halfLane; /* happen to hit */
                    }
                }
            }

            TyVal tmp = details::tile::shuffle_dispatch<TyVal>::shfl(out, mask, srcLane, 32);
            if (i <= rank) {
                out = op(out, tmp);
            }
        }
        // Redistribute the value after performing all the reductions
        return group.shfl(out, groupSize-1);
    }
}

template <typename TyVal, typename TyOp, unsigned int TySize, typename ParentT>
_CG_QUALIFIER auto coalesced_reduce(const __single_warp_thread_block_tile<TySize, ParentT>& group, TyVal val, TyOp op) -> decltype(op(val, val)) {
    for (int mask = TySize >> 1; mask > 0; mask >>= 1)
        val = op(val, group.shfl_xor(val, mask));

    return val;
}

} // details

_CG_END_NAMESPACE

#endif // _CG_COALESCED_REDUCE_H_
