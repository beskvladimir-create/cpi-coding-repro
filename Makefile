PY ?= python

.PHONY: all data trie confirm consensus audit clean

all:
	$(PY) src/run_all.py

data:
	$(PY) src/make_synth.py

trie: data
	$(PY) src/trie_classifier.py

confirm: data
	$(PY) src/confirm_models.py

consensus:
	$(PY) src/consensus_sim.py

audit: data
	cd src && $(PY) leakage_audit.py && $(PY) significance.py && $(PY) aggregation_to_training.py

clean:
	rm -f data/synth/*.parquet data/synth/manifest.json
	rm -f results/*.csv results/*.md
	rm -f figures/*.png
