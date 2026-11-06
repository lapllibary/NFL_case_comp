# Oyunkhand — WR and TE research code

Source-code snapshot from the existing Combine-to-NFL research archive, prepared for team review.

## Start here

- `notebooks/wr_combine_bayesian_v3.ipynb`: WR Bayesian V3 model and diagnostics.
- `notebooks/te_combine_to_separation.ipynb`: initial TE separation analysis.
- `notebooks/te_phase_v2_development.ipynb`: TE V2 development comparison.
- `notebooks/TE_scouting_submission.ipynb`: latest scouting extension.
- `analysis/`: Python sources for preprocessing, WR models, TE models and scouting.
- `requirements.txt`: dependencies from the original archive; versions are not pinned.

## Reproduction and scope

Obtain competition CSVs separately from NFL Big Data Bowl 2027 under the competition terms. Raw competition data, model binaries, generated result tables and notebook outputs are not included here. Notebook outputs were already cleared in the source archive. This is a set of stage-specific research scripts, not one automated end-to-end application.

Run from this directory where relative paths are used. Review the paths and environment settings in each stage before execution. The scouting extension supports `NFL_DATA_DIR`, `TE_EXTERNAL_DIR`, and `TE_SCOUT_OUT`; its notebook prepares external inputs. V2 requires development-only `tracking.csv`, `player_features.csv`, and `routes.csv` from the pilot preprocessing and the `TE_V2_INPUT` setting. Earlier scripts may retain Colab-specific paths and need configuration for another environment.

External pressure data are FTN Data via nflverse, CC BY-SA 4.0. See `reports/TE_data_attribution.txt` and `results/scouting/source_manifest.json` for provenance. Preserve attribution and applicable source-specific terms; no blanket license is granted to this mixed research.

## Interpretation

Movement features have not demonstrated a reliable improvement in separation prediction. Archetype rules and team-fit comparisons are provisional, not validated scouting rankings. The Hard-Brake proxy selected 40 of 41 players. The historical cohorts are 2023–2025; these are not a 2027 prospect board, and 2025 has already been inspected. No fresh model training was performed when importing these sources into the team repository.

## Import validation

All ten Python source files were parsed for syntax. All eight notebooks were parsed as JSON and checked to contain no cell outputs. Model training and full data-dependent execution were not rerun.

