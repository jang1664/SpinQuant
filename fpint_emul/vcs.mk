compile: setup
	vcs \
	-V \
	-kdb \
	-sverilog \
	-full64 \
	-debug_access+all \
	-l $(COMPILE_LOG) \
	-timescale=$(TIME_SCALE) \
	+libext+.v+ \
	+ntb_random_seed=999 \
	${VC_INCDIRS} \
	$(DEFINES) \
	-CFLAGS $(CFLAGS) \
	$(RTLS) \
	$(TB) \
	$(DPI_SRCS) \
	$(SOFTFLOAT_LIB) \
	$(PARAMS)

sim: compile
	./simv -reportstats -l $(SIM_LOG)