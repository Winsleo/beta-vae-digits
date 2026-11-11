# Usage: make test | train | evaluate | plots | extras | multiseed | all
PY     ?= python
EPOCHS ?= 500
RUN    ?= runs/main

test:
	$(PY) tests/test_core.py

train:
	$(PY) src/train.py --epochs $(EPOCHS) --out $(RUN) --snapshots 100

evaluate:
	$(PY) src/evaluate.py --models $(RUN)/models --out results/main.json
	$(PY) src/evaluate.py --models $(RUN)/models_ep100 --out results/ep100.json

plots:
	$(PY) src/plots.py --models $(RUN)/models --early $(RUN)/models_ep100 --history $(RUN)/history --out results/figs

extras:
	$(PY) src/q2_leakage.py --out results/q2
	$(PY) src/q4_autoencoder.py --out results/q4

multiseed:
	$(PY) src/multiseed.py --seeds 0 1 2 3 4 --epochs $(EPOCHS) --out results/multiseed.json

all: test train evaluate plots extras multiseed
