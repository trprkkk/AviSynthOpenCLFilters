# Convenience build/test entry points (no OpenCL needed for the reference test).
CC ?= g++
CXXFLAGS := -O2 -std=c++17 -Wall

REF_BIN := build/ktgmc_cpu_ref

.PHONY: all test lint ref clean

all: test lint

lint:
	./lint/lint_opencl.sh

ref:
	-@mkdir build 2>nul
	$(CXX) $(CXXFLAGS) -w sim/ktgmc_cpu_ref.cpp -o $(REF_BIN)

test: ref lint
	python3 python/run_validation.py
	python3 python/run_motion_core.py
	python3 python/run_mv_aux.py
	python3 python/run_mv_interp.py
	python3 python/run_mv_mean.py
	python3 python/run_mv_searchprep.py
	python3 python/run_mv_search.py
	python3 python/run_mv_io.py
	python3 python/run_mv_rb2b.py
	python3 python/run_mv_rb2b_pad.py
	python3 python/run_mv_degrain.py
	python3 python/run_kfm_deband.py
	python3 python/run_kfm_edgelevel.py
	python3 python/run_kfm_temporalnr.py
	python3 python/run_kfm_mergestatic.py
	python3 python/run_kfm_filterbase.py
	python3 python/run_kfm_noiseclip.py
	python3 python/run_kfm_deblock.py
	python3 python/run_kfm_deblock_qp.py
	python3 python/run_kfm_deblock_aux.py
	python3 python/run_kfm_combinganalyze.py
	python3 python/run_kfm_decombeucf.py
	python3 python/run_avscuda_merge.py
	python3 python/run_avscuda_filters.py
	python3 python/run_avscuda_convert.py
	python3 python/run_avscuda_conditional.py
	python3 python/run_avscuda_resample.py
	python3 python/run_nnedi3_pad.py
	python3 python/run_nnedi3_prescreen.py
	python3 python/run_nnedi3_compute.py
	python3 python/run_masktools_lut.py
	python3 python/run_mv_calc_all_sad.py
	python3 python/run_mv_prepare.py
	python3 lint/audit_inventory.py

clean:
	rm -rf build
