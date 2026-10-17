PY ?= python

.PHONY: all data trie confirm consensus audit cost robust sensitivity grade-ablation noise bands numbers figures pretrained pretrained-analysis clean

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

cost: data
	cd src && $(PY) cost_benchmark.py

robust: data
	cd src && $(PY) robustness_checks.py && $(PY) aggregation_passes.py

# sensitivity checks of round 2: grade-block ablation, generator noise,
# final reliability weights, annotator-population sensitivity. The corpus
# variants are generated in memory; data/synth is never rewritten. Run one at
# a time (noise_sensitivity.py takes about 15 minutes on one core).
sensitivity: data
	cd src && $(PY) grade_token_ablation.py && $(PY) noise_sensitivity.py \
	  && $(PY) consensus_weights.py && $(PY) consensus_sensitivity.py

grade-ablation: data
	cd src && $(PY) grade_token_ablation.py

noise: data
	cd src && $(PY) noise_sensitivity.py

# descriptive bands and supplementary table of every model pair, from the
# released pairwise results (needs `audit` and `pretrained-analysis` outputs)
bands:
	cd src && $(PY) pairwise_bands.py

# derived figures quoted in the text (ranges, medians, differences)
numbers:
	cd src && $(PY) manuscript_numbers.py

# accuracy-cost figure from results/confirm_metrics.csv + cost_benchmark_summary.csv
figures:
	$(PY) src/make_cost_figure.py

# pretrained-encoder baselines: needs `pip install transformers`, GPU advisable;
# writes results/pretrained/rerun/
pretrained: data
	$(PY) src/pretrained_baselines.py

# paired comparisons of the pretrained baselines against the seven models
pretrained-analysis:
	$(PY) src/analyze_pretrained.py

clean:
	rm -f data/synth/*.parquet data/synth/manifest.json
	rm -f results/*.csv results/*.md results/*.json
	rm -f figures/*.png
