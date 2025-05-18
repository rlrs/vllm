#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"

#include "cutlass_extensions/gemm/dispatch_policy.hpp"
#include "cutlass_extensions/gemm/collective/collective_builder.hpp"

#include "cutlass_gemm_caller.cuh"

#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

// added these
#include "cutlass/util/device_memory.h"


namespace vllm {

using namespace cute;

using         ElementA    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using         LayoutATag  = cutlass::layout::RowMajor;
constexpr int AlignmentA  = 16;


using         ElementB    = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using         LayoutBTag  = cutlass::layout::ColumnMajor;
constexpr int AlignmentB  = 128;

using         ElementD    = cutlass::bfloat16_t;
using         ElementC    = cutlass::bfloat16_t;
using         LayoutCTag  = cutlass::layout::RowMajor;
using         LayoutDTag  = cutlass::layout::RowMajor;
constexpr int AlignmentD  = 128 / cutlass::sizeof_bits<ElementD>::value;
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;

using ElementAccumulator  = float;
using ArchTag             = cutlass::arch::Sm120;
using OperatorClass       = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape    = Shape<_128,_128,_128>;
using ClusterShape        = Shape<_1,_1,_1>; // Only <1,1,1> is supported

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,                      
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
  >::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
  >::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int,int,int,int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

// Reference device GEMM implementation type
using StrideA   = typename Gemm::GemmKernel::StrideA;
using LayoutA   = decltype(cute::make_layout(make_shape(0,0,0), StrideA{}));
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using LayoutB   = decltype(cute::make_layout(make_shape(0,0,0), StrideB{}));
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using LayoutC   = decltype(cute::make_layout(make_shape(0,0,0), StrideC{}));
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutD   = decltype(cute::make_layout(make_shape(0,0,0), StrideD{}));

template <typename Gemm>
void cutlass_gemm_caller_blockwise(torch::Tensor& out, torch::Tensor const& a,
                                   torch::Tensor const& b,
                                   torch::Tensor const& a_scales,
                                   torch::Tensor const& b_scales) {
//   using Adapter = typename Gemm::Adapter;
//   using GemmKernel = typename Adapter::GemmKernel;
//   using StrideA = typename Adapter::GemmKernel::StrideA;
//   using StrideB = typename Adapter::GemmKernel::StrideB;
//   using StrideD = typename Adapter::GemmKernel::StrideD;
//   using StrideC = typename Adapter::GemmKernel::StrideC;
//   using LayoutSFA = typename Adapter::GemmKernel::CollectiveMainloop::LayoutSFA;
//   using LayoutSFB = typename Adapter::GemmKernel::CollectiveMainloop::LayoutSFB;
  using ScaleConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

//   using ElementABDataType = typename Gemm::ElementAB::DataType;
//   using ElementABScaleFactorType = typename Gemm::ElementAB::ScaleFactorType;
//   using ElementC = typename Gemm::ElementC;
//   using ElementD = typename Gemm::ElementD;

  int32_t m = a.size(0), n = b.size(1), k = a.size(1);
  auto prob_shape = cute::make_shape(m, n, k, 1);

  StrideA a_stride;
  StrideB b_stride;
  StrideC c_stride;
  StrideD d_stride;
  a_stride =
      cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(m, k, 1));
  b_stride =
      cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(n, k, 1));
  c_stride =
      cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(m, n, 1));
  d_stride =
      cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(m, n, 1));

  LayoutSFA layout_SFA =
      ScaleConfig::tile_atom_to_shape_SFA(make_shape(m, n, k, 1));
  LayoutSFB layout_SFB =
      ScaleConfig::tile_atom_to_shape_SFB(make_shape(m, n, k, 1));

  auto a_ptr = static_cast<ElementA::DataType*>(a.data_ptr());
  auto b_ptr = static_cast<ElementB::DataType*>(b.data_ptr());
  auto a_scales_ptr = static_cast<ElementA::ScaleFactorType*>(a_scales.data_ptr());
  auto b_scales_ptr = static_cast<ElementB::ScaleFactorType*>(b_scales.data_ptr());

//   typename GemmKernel::MainloopArguments mainloop_args{
//       a_ptr,        a_stride,   b_ptr,        b_stride,
//       a_scales_ptr, layout_SFA, b_scales_ptr, layout_SFB};

//   auto c_ptr = static_cast<ElementD*>(out.data_ptr());
//   typename GemmKernel::EpilogueArguments epilogue_args{
//       {}, c_ptr, c_stride, c_ptr, c_stride};

    typename Gemm::Arguments args {
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},

        // Mainloop operands
        {   a_ptr, a_stride, 
            b_ptr, b_stride, 
            a_scales_ptr, layout_SFA,
            b_scales_ptr, layout_SFB
        },

        // Epilogue operands
        {   {1.0f, 0.0f},
            static_cast<ElementC*>(out.data_ptr()), c_stride, // not touched since beta=0
            static_cast<ElementD*>(out.data_ptr()), d_stride
        }
    };
    // auto arguments = make_args(options);

    Gemm gemm;
    size_t workspace_bytes = gemm.get_workspace_size(args);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_bytes);


//   c3x::cutlass_gemm_caller<GemmKernel>(a.device(), prob_shape, mainloop_args,
//                                        epilogue_args);
    CUTLASS_CHECK( gemm.can_implement(args) );   // fast sanity check
    CUTLASS_CHECK( gemm.initialize(args, workspace.get()) );
    CUTLASS_CHECK( gemm.run() );
}

template <typename OutType>
void cutlass_gemm_blockwise_sm120_fp8_dispatch(torch::Tensor& out,
                                               torch::Tensor const& a,
                                               torch::Tensor const& b,
                                               torch::Tensor const& a_scales,
                                               torch::Tensor const& b_scales) {
  cutlass_gemm_caller_blockwise<Gemm
//   <
//     OutType, Shape<_128, _128, _128>, Shape<_128, _1, _1>,
//     Shape<_1, _1, _1>, cutlass::epilogue::collective::EpilogueScheduleAuto, // cutlass::epilogue::TmaWarpSpecialized1Sm,
//     cutlass::gemm::collective::KernelScheduleAuto //cutlass::gemm::KernelTmaWarpSpecializedPingpongMxf8f6f4Sm120
//     >
    >(out, a, b, a_scales, b_scales);
}

}  // namespace vllm
