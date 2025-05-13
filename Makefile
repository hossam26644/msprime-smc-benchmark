
.PHONY: all setup data figure clean mrproper

all: data figure

setup:
	virtualenv -p python3 .venv
	source .venv/bin/activate
	pip install -r requirements.txt

data-perf:
	mkdir -p data
	python3 evaluation/generate_ancestry_perf_data.py

figure-perf:
	mkdir -p figures
	python3 evaluation/plot.py ancestry-perf-hudson
	python3 evaluation/plot.py ancestry-perf-smc

data-varying-seq:
	mkdir -p data
	python3 evaluation/generate_ancestry_perf_data.py run-varying-seq-len-sims
	python3 evaluation/generate_ancestry_perf_data.py run-varying-seq-len-sims-smc

data-varying-k:
	mkdir -p data
	python3 evaluation/generate_ancestry_perf_data.py run-varying-k-sims
data-hybrid:
	mkdir -p data
	python3 evaluation/generate_ancestry_perf_data.py run-hybrid-sims

plot-hybrid:
	mkdir -p figures
	python3 evaluation/plot.py hybrid

data-panels:
	mkdir -p data
	python3 evaluation/generate_ancestry_perf_data.py run-panels

plot-panels:
	mkdir -p figures
	python3 evaluation/plot.py panels

plot-fixed:
	mkdir -p figures
	python3 evaluation/plot.py sequence-length-vs-time

data-varying-sample-size:
	mkdir -p data
	python3 evaluation/generate_ancestry_perf_data.py run-varying-sample-size
	python3 evaluation/generate_ancestry_perf_data.py run-varying-sample-size-smc

plot-varying-sample-size:
	mkdir -p figures
	python3 evaluation/plot.py sample-size-vs-time

plot-varying-k:
	mkdir -p figures
	python3 evaluation/plot.py k-vs-time

cleanfigures:
	rm figures/*

cleandata:
	rm data/*

clean:
	cleanfigures
	cleandata
