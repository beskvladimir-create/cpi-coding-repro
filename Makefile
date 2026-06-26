PY ?= python

.PHONY: all data trie confirm consensus clean

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

clean:
	rm -f data/synth/*.parquet data/synth/manifest.json
	rm -f results/*.csv results/*.md
	rm -f figures/*.png
