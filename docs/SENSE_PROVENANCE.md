# Sensory mapping provenance

Minecraft is mapped onto **named Drosophila sensory populations**, not a wood-only feature vector.

## Sources (read locally in `.research/`, not vendored)

- [blendi-remade/fly-brain-minecraft](https://github.com/blendi-remade/fly-brain-minecraft)
  `SensoryFrame.java`, `WorldSenses.java`, `SensoryEncoders.java`, `OdorTable`, `TasteTable`
- [mlabonne/chessfly](https://huggingface.co/mlabonne/chessfly)
  frozen \(W_{ij}=\mathrm{sign}_{ij}\exp\theta_{ij}\), 5-step rate settle, learnable encoder/gains/decoder
- Male CNS: Berg et al., *Cell* 2026, neuPrint `male-cns:v1.0` (CC BY 4.0)
- Shiu et al., *Nature* 2024 (LIF / sign convention)
- [TuragaLab/flybody](https://github.com/TuragaLab/flybody) — MuJoCo body + IL/RL locomotion; **not** used as the Minecraft body

## Frame fields (match fly-brain-minecraft)

`luminance[]`, `objects[]`, `odor` + `odorBearingDeg`, `taste`, JO wind L/R, sounds, bristle regions, groomDust, damage, thermo/hygro, airborne / legsOnGround, analytic `objectChannels` (LC4, LPLC2, LC11, LC10a, HS).

Hill law: \(r \propto S^{1.5}/(0.2^{1.5}+S^{1.5})\). Odor falloff \(\exp(-d/\lambda)\), \(\lambda=6\) blocks.

## Config tables

| File | Role |
|------|------|
| `configs/sense/block_odor.json` | any block class → glomeruli |
| `configs/sense/item_odor.json` | dropped items |
| `configs/sense/entity_odor.json` | mobs / player CO2 |
| `configs/sense/taste_table.json` | contact → GRNs |
| `configs/sense/retina.json` | raycast columns |
| `configs/sense/mechano.json` | JO / bristle / weather |
| `configs/sense/population_index.json` | logical channel → neuPrint type |

Wood logs are **rows** in the odor tables (green-leaf / terpene glomeruli), not a special API.

## Goals

`configs/goals/*.json` compile to **additive** `SensoryInjection`s (`js/sense/goalToSense.js`, `python/sense/goal_to_sense.py`). Success rules are for RL/eval only.

## Connectome

Default training graph: `data/connectome/mini_male_cns.npz` from `python/tools/build_mini_graph.py`. Real ORN/GRN/LC/DN **names**; pruned hidden core. Full 176k graph is **not** redistributed — see `python/tools/fetch_connectome.py`.
