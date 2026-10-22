# Computational Username Analysis: Classes A, B, and C

This repository organizes the original three-class username research and subsequent supplementary analyses. **Use as a PRIVATE research repository unless the datasets have been cleared for public redistribution.** Class A includes names and case-related information, while Class B includes public-figure identifiers and biographical metadata.

## Data and research design

- **Class A:** 236 source rows; 234 unique username strings. Source: `data/class_a_master.csv`.
- **Class B:** 20,000 X/YouTube account rows with `handle`, `qid`, platform, and other fields. Source: `data/class_b_x_youtube.csv`.
- **Class C:** Reddit username corpus analyzed in aggregate (25,865,739 usernames in the report). **The underlying Class C `users.csv` is not included**; notebook 03 downloads it through KaggleHub.

The principal predictive comparisons are separate binary classifiers: A vs B, and A vs C. The Class C full-corpus aggregate analysis is distinct from the smaller sampled A-vs-C classifier. The NER workflow is a completed three-class analysis, with supplied statistical results and interpretations. The updated A-vs-B notebook adds structural tests and exploratory within-Class-B gender associations.

## Files

| File | Purpose |
|---|---|
| `notebooks/01_class_a_descriptives.ipynb` | Class A exploration and features |
| `notebooks/02_class_b_analysis.ipynb` | Class B X/YouTube descriptive/statistical work |
| `notebooks/03_class_c_aggregate.ipynb` | Chunked large-scale Reddit descriptive analysis |
| `notebooks/04_class_a_vs_b_original.ipynb` | Original Class A versus B classifier |
| `notebooks/05_class_a_vs_c.ipynb` | Class A versus C classifier |
| `notebooks/06_three_class_ner_workflow.ipynb` | Exploratory NER analyses across three classes |
| `notebooks/07_a_vs_b_supplemental_statistics_gender.ipynb` | Expanded A/B notebook with supplementary structural and gender analyses |
| `reports/final_technical_report.docx` / `.pdf` | Final consolidated technical report (including Class A recalculations and NER results) |
| `reports/NER_Report.html` | Detailed NER experiment report from the supplied results package |
| `results/ner/` | NER statistical tables, graphs, configuration, and script for reproducing aggregate statistical calculations |
| `results/class_b_gender_supplement_statistics.csv` | Previously produced gender comparison outputs |
| `data/` | Original Class A/B CSV datasets |

## Execution notes

The notebooks were originally authored in Google Colab and may request interactive file uploads. They are **not yet a unified, end-to-end executable pipeline**. Run the notebook workflows individually, reviewing cell order and file-path assumptions. Results in the consolidated report are previously reported, not newly reproduced as part of this packaging. The NER statistics script reproduces aggregate calculations from supplied counts, not a fresh run of spaCy inference.

Install typical dependencies with `pip install -r requirements.txt`. The transformer NER notebook additionally downloads spaCy's `en_core_web_trf` model and may require substantial memory and compute.

## Methodological cautions

The classification task examines statistical separation between assembled username samples, **not criminality determination for individual usernames**. Observed gender values in Class B should be treated as source metadata, not a basis for inferring an unobserved individual's gender from their username. Cross-platform differences and source selection confound interpretation.

## Reproducibility and publication

- Project code uses random seeds in several notebooks; reruns can still differ if package versions or source corpora change.
- Check dataset provenance, permitted redistribution, personal information, and notes before making this repository public.
- Consider releasing only code, aggregate statistics, and a de-identified sample publicly while keeping raw data private.

## NER data provenance

The `results/ner/` tables are the original supplied reference outputs. The supplied reproduction script calculates aggregate tests from stored counts; it does not rerun the transformer model over username records. The plots are from the supplied reproduced outputs. Original NER report and configuration are included.
