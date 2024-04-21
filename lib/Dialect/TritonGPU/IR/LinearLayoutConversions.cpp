#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include <numeric>

namespace mlir::triton::gpu {
namespace {

LinearLayout blockedToLinearLayout(ArrayRef<int64_t> shape,
                                   BlockedEncodingAttr blocked) {
  MLIRContext *ctx = blocked.getContext();

  assert(shape.size() == blocked.getOrder().size());
  const int rank = shape.size();

  // Create the StringAttrs we'll need for our layout.
  StringAttr kRegister = StringAttr::get(ctx, "register");
  StringAttr kThread = StringAttr::get(ctx, "thread");
  StringAttr kWarp = StringAttr::get(ctx, "warp");
  StringAttr kBlock = StringAttr::get(ctx, "block");
  std::vector<StringAttr> outDimNames;
  for (int i = 0; i < rank; i++) {
    outDimNames.push_back(StringAttr::get(ctx, "dim" + llvm::Twine(i)));
  }

  // The size of `blocked` (i.e. its register * thread * warp * block size) may
  // be different than the shape's size.  If `blocked` is larger than the shape,
  // it means that some data elements will be stored twice (or more) in
  // registers, i.e. the LinearLayout will map two or more (reg, thread, warp,
  // block) input tuples to the same (dim0,...,dimN) output tuple.
  //
  // We keep track of this in `shapeRemaining`, which tells us how much of the
  // shape has been "covered" by the layout.  We call layoutForDim starting with
  // the most minor input dimension (i.e. "register") and once we've "used up"
  // all of shapeRemaining, the LinearLayout broadcasts the remaining input
  // dimensions.
  std::vector<int32_t> shapeRemaining(shape.begin(), shape.end());
  auto layoutForDim = [&](StringAttr inDimName, ArrayRef<unsigned> sizes,
                          ArrayRef<unsigned> order) {
    LinearLayout ret = LinearLayout::empty();

    // Start with the most minor dimension, which is order[0].
    for (int i = 0; i < rank; i++) {
      int dim = order[i];

      int32_t size, zeros;
      if (shapeRemaining[dim] >= sizes[dim]) {
        size = sizes[dim];
        zeros = 0;
        shapeRemaining[dim] /= sizes[dim];
      } else {
        size = shapeRemaining[dim];
        zeros = size > 0 ? sizes[dim] / size : sizes[dim];
        shapeRemaining[dim] = 0;
      }

      // TODO: Can I add the zeros after the fact?
      ret *= LinearLayout::identity1D(size, inDimName, outDimNames[dim]) *
             LinearLayout::zeros1D(zeros, inDimName, outDimNames[dim]);
    }
    return ret;
  };

  // First the shape is split into CTASplitNum pieces, which are distributed
  // among the NumCTAs in the CTG.  Then it's distributed among the threads in
  // the block.
  LinearLayout ctgLayout =
      layoutForDim(kBlock, blocked.getCTASplitNum(), blocked.getCTAOrder());

  // CTASplitNum[i] != CTAsPerCGA[i] means we duplicate the layout along
  // dimension i so that there are CTAsPerCGA[i] / CTASplitNum[i] copies.
  for (int i = 0; i < rank; i++) {
    int dim = blocked.getCTAOrder()[i];
    unsigned splitNum = blocked.getCTASplitNum()[dim];
    unsigned CTAsPerCGA = blocked.getCTAsPerCGA()[dim];
    assert(CTAsPerCGA % splitNum == 0);
    ctgLayout *=
        LinearLayout::zeros1D(CTAsPerCGA / splitNum, kBlock, outDimNames[dim]);
  }

  // Now split the shape among the register+thread+warp.
  LinearLayout ctaLayout =
      layoutForDim(kRegister, blocked.getSizePerThread(), blocked.getOrder()) *
      layoutForDim(kThread, blocked.getThreadsPerWarp(), blocked.getOrder()) *
      layoutForDim(kWarp, blocked.getWarpsPerCTA(), blocked.getOrder());

  // Join the layouts, with the CTG layout being more minor and its being
  // transposed to match the order of the CTA layout.  (You can't multiply two
  // layouts with different relative orders for the dims they have in common.)
  LinearLayout ret =
      ctaLayout *
      ctgLayout.transposeOuts(llvm::to_vector(ctaLayout.getOutDimNames()));

  // If the shape per CTA is larger than the layout, we repeat the layout by
  // having each thread hold multiple elements, i.e. adding to the register
  // dimension.  (In a way, this means `register` is both the most minor and
  // most major input dimension of the layout within the block.)
  //
  // The `block` dimension is always more major than the repeats.  That is, we
  // repeat enough so that then when we tack on the multi-block dimension, we
  // fill the shape exactly.
  for (int i = 0; i < rank; i++) {
    int dim = blocked.getOrder()[i];
    int32_t layoutSize = ret.getOutDimSize(outDimNames[dim]);

    // Note we divide by getCTASplitNum(), not getCTASPerCGA.  CTASplitNum[i]
    // tells us how many unique copies of the reg+thread+warp layout there are
    // in the CGA.  This is broadcasted to the CTAsPerCGA[i] CTAs in the block.
    int32_t shapeSize = shape[dim] / blocked.getCTASplitNum()[dim];
    if (shapeSize <= layoutSize) {
      continue;
    }
    assert(shapeSize % layoutSize == 0);
    ret *= LinearLayout::identity1D(shapeSize / layoutSize, kRegister,
                                    outDimNames[dim]);
  }

  return ret;
}

LinearLayout sharedToLinearLayout(ArrayRef<int64_t> shape,
                                  SharedEncodingAttr layout) {
  return LinearLayout::empty();
}

} // anonymous namespace

LinearLayout toLinearLayout(ArrayRef<int64_t> shape, Attribute layout) {
  if (auto blocked = dyn_cast<BlockedEncodingAttr>(layout)) {
    return blockedToLinearLayout(shape, blocked);
  } else if (auto shared = dyn_cast<SharedEncodingAttr>(layout)) {
    return sharedToLinearLayout(shape, shared);
  }

  // TODO(jlebar): Other layouts
  llvm::llvm_unreachable_internal("Unsupported layout");
}

} // namespace mlir::triton::gpu
