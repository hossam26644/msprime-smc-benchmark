
.PHONY: all setup data figure clean mrproper

all: data figure

setup:
	virtualenv -p python3 .venv
	source .venv/bin/activate
	pip install -r requirements.txt

data:
	mkdir -p data
	python3 evaluation/generate_ancestry_perf_data.py

figure:
	mkdir -p figures
	python3 evaluation/plot.py ancestry-perf-hudson
	python3 evaluation/plot.py ancestry-perf-smc

cleanfigures:
	rm figures/*

cleandata:
	rm data/*

clean:
	cleanfigures
	cleandata
