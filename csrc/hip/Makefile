HIPCC ?= hipcc
# Use HIP_ARCHITECTURES env var if set, otherwise default to native
HIP_ARCH ?= $(if $(HIP_ARCHITECTURES),$(HIP_ARCHITECTURES),native)
# Extra compilation flags matching setup.py defaults
HIP_FLAGS ?= -O3 -fPIC -std=c++17 --offload-arch=$(HIP_ARCH) \
-U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF_CONVERSIONS__ \
-fgpu-flush-denormals-to-zero $(CC_FLAG)
INCLUDES ?= -I../

SRCS := causal_conv1d_fwd_hip.hip \
        causal_conv1d_bwd_hip.hip \
        causal_conv1d_update_hip.hip
OBJS := $(SRCS:.hip=.o)
LIB  := libcausal_conv1d_hip.so
TEST := testbench.out

all: $(LIB)

$(TEST): testbench.hip $(LIB)
	$(HIPCC) $(HIP_FLAGS) $(INCLUDES) testbench.hip -L. -lcausal_conv1d_hip -o $@

testbench: $(TEST)

run-testbench: $(TEST)
	LD_LIBRARY_PATH=. ./$(TEST)

%.o: %.hip
	$(HIPCC) $(HIP_FLAGS) $(INCLUDES) -c $< -o $@

$(LIB): $(OBJS)
	$(HIPCC) -shared -o $@ $(OBJS)

clean:
	rm -f $(OBJS) $(LIB) $(TEST)
.PHONY: all clean testbench run-testbench

