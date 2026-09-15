# Include Vortex configuration
UNITTEST_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
ROOT_DIR := $(abspath $(UNITTEST_DIR)/../../..)
include $(ROOT_DIR)/build/config.mk

RTL_DIR := $(VORTEX_HOME)/hw/rtl
SRC_DIR := $(VORTEX_HOME)/hw/unittest/fpint_emul

# Force use of gcc-11/g++-11 for C++20 features
export CC := /usr/bin/gcc-11
export CXX := /usr/bin/g++-11

SIM_EXEC ?= vcs

TIME_SCALE=1ns/1ps
TOP_MODULE := tb_fpint_emul

TB   := $(SRC_DIR)/tb_fpint_emul.sv
RTLS := $(RTL_DIR)/VX_gpu_pkg.sv \
        $(RTL_DIR)/verification/cf_math_pkg.sv \
        $(RTL_DIR)/verification/fpint_emul.sv

DPI_SRCS := $(VORTEX_HOME)/hw/dpi/float_dpi.cpp \
            $(VORTEX_HOME)/sim/common/util.cpp \
            $(VORTEX_HOME)/sim/common/softfloat_ext.cpp \
            $(VORTEX_HOME)/sim/common/rvfloats.cpp

SOFTFLOAT_LIB := $(THIRD_PARTY_DIR)/softfloat/build/Linux-x86_64-GCC/softfloat.a
						

INCDIRS_ := $(VORTEX_HOME)/build/hw \
            $(VORTEX_HOME)/hw \
            $(VORTEX_HOME)/sim/common \
            $(RTL_DIR) \
            $(VORTEX_HOME)/hw/dpi \
            $(THIRD_PARTY_DIR)/softfloat/source/include

DEFINES_ := SIMULATION DEBUG_LEVEL=3 

CFLAGS  := "-std=c++17 -Wall -Wextra -Wno-array-bounds -Wno-maybe-uninitialized -fPIC -I$(VORTEX_HOME)/build/hw -I$(VORTEX_HOME)/hw -I$(VORTEX_HOME)/sim/common -I$(VORTEX_HOME)/hw/dpi -I$(RTL_DIR) -I$(THIRD_PARTY_DIR)/softfloat/source/include -DXLEN_$(XLEN)"
LDFLAGS := 

PARAMS_  := 

COMPILE_LOG := logs/compile.log
SIM_LOG     := logs/sim.log

VL_INCDIRS  := $(foreach dir,$(INCDIRS_),-I$(dir))
VC_INCDIRS  := $(foreach dir,$(INCDIRS_),+incdir+$(dir))

empty :=
space := $(empty) $(empty)
DEFINES  := +define+$(subst $(space),+,$(DEFINES_))

PARAMS := $(foreach p,$(PARAMS_),-pvalue+$(p))

setup:
	mkdir -p logs reports

include $(SIM_EXEC).mk

fst2vcd:
	for file in reports/*.fst; do \
	  fst2vcd $$file -o reports/$${file##*/}.vcd; \
	done

vcd2fsdb:
	for file in reports/*.vcd; do \
	  vcd2fsdb $$file -o reports/$${file##*/}.fsdb; \
	done

fst2fsdb:
	for file in reports/*.fst; do \
	  fst2vcd $$file -o reports/$${file##*/}.vcd; \
	  vcd2fsdb  reports/$${file##*/}.vcd -o reports/$${file##*/}.fsdb; \
	  rm reports/$${file##*/}.vcd; \
	done

clean:
	rm -rf obj_dir 
	rm -rf reports logs 
	rm -rf simv novas* ucli* verdi* csrc* simv.daidir*
	rm -rf vcd2fsdb*