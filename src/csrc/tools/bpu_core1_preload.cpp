// Small LD_PRELOAD shim for S600/HBRT loadability diagnostics and single-core
// deployment.
//
// HBRT 4.7.5's HBM loader asks libbpu how many BPU cores exist, then validates
// that one model buffer maps to identical IOVA addresses on every reported core.
// On the target S600 board, some valid single-core nash-p HBMs map at different
// IOVA addresses on other idle cores and fail during hbDNNInitializeFromFiles
// before scheduling can restrict execution to --core_id 1.  Overriding only
// hb_bpu_core_num() to report one visible core makes HBRT validate the core that
// is actually used; it does not change graph placement or HBM contents.
//
// Usage:
//   Set LD_PRELOAD=/path/to/libfoundationpose_bpu_core1_preload.so before
//   running /usr/hobot/bin/hrt_model_exec model_info --model_file model.hbm.

#include <cstdint>

extern "C" std::uint32_t hb_bpu_core_num() noexcept { return 1U; }
