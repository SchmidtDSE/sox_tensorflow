# Model Comparisons: PNW-Cnet v4 — Pacific Northwest Owl Species Classifier

When comparing the differences between `sox-tensorflow` and `sox` its useful to not only look at the differences in the spectrograms, but also the how these differences might propagate through a model. To this end we are including the script [model_comparison_cli](https://github.com/SchmidtDSE/sox_tensorflow/blob/main/sox_comparison/model_comparison_cli.py)

PNW-Cnet is a convolutional neural network for classifying wildlife sounds from
passive acoustic recordings, with a focus on owl and other bird species in the
Pacific Northwest. It takes 257×1000 grayscale spectrograms (12-second segments
at 8 kHz) as input and outputs probabilities across 51 sound classes.

**source**:
- Paper: https://www.sciencedirect.com/science/article/pii/S2352711023001693
- Interactive app: https://github.com/zjruff/Shiny_PNW-Cnet

---

## Files

| File | Description |
|---|---|
| `PNW-Cnet_v4_TF.h5` | Trained Keras model weights (TensorFlow HDF5 format) |
| `target_classes.csv` | 51-class label mapping with species name and category |

---

## PNW-OWL Citation

If you use PNW-Cnet in your work, please cite the original paper:

> Ruff, Z. J., Lesmeister, D. B., Duchac, L. S., Padmaraju, B. K., & Sullivan, C. M. (2023).
> **Automated species detection: An end-to-end pipeline utilizing PNW-Cnet and
> automated recording units.**
> *SoftwareX*, 22, 101414.
> https://doi.org/10.1016/j.softx.2023.101414

**BibTeX:**
```bibtex
@article{ruff2023pnwcnet,
  title   = {Automated species detection: An end-to-end pipeline utilizing
             {PNW-Cnet} and automated recording units},
  author  = {Ruff, Zachary J. and Lesmeister, Damon B. and Duchac, Leila S.
             and Padmaraju, Bhanu K. and Sullivan, Christopher M.},
  journal = {SoftwareX},
  volume  = {22},
  pages   = {101414},
  year    = {2023},
  doi     = {10.1016/j.softx.2023.101414},
  url     = {https://www.sciencedirect.com/science/article/pii/S2352711023001693}
}
```

---

## Classes

The model classifies 51 sound classes across owls, other birds, mammals, and
nuisance sounds:

| Code | Sound | Species | Category |
|---|---|---|---|
| AEAC | Northern saw-whet owl | *Aegolius acadicus* | Owl |
| BRCA | Canada goose | *Branta canadensis* | Game bird |
| BRMA | Marbled murrelet | *Brachyramphus marmoratus* | Other bird |
| BUVI | Great horned owl | *Bubo virginianus* | Owl |
| CAGU | Hermit thrush | *Catharus guttatus* | Songbird |
| CALU | Wolf howl | *Canis lupus* | Mammal |
| CAUS | Swainson's thrush | *Catharus ustulatus* | Songbird |
| CCOO | Olive-sided flycatcher | *Contopus cooperi* | Songbird |
| CHFA | Wrentit | *Chamaea fasciata* | Songbird |
| CHMI | Common nighthawk call | *Chordeiles minor* | Other bird |
| CHMI_IRREG | Common nighthawk boom | *Chordeiles minor* | Other bird |
| COAU | Northern flicker series | *Colaptes auratus* | Woodpecker |
| COAU2 | Northern flicker 'skew' | *Colaptes auratus* | Woodpecker |
| COCO | Common raven | *Corvus corax* | Corvid |
| CYST | Steller's jay | *Cyanocitta stelleri* | Corvid |
| DEFU | Sooty grouse | *Dendragapus fuliginosus* | Game bird |
| DOG | Dog barks | — | Nuisance |
| DRPU | Downy woodpecker call | *Dryobates pubescens* | Woodpecker |
| DRUM | Woodpecker spp. drum | — | Woodpecker |
| FLY | Insect buzz | — | Nuisance |
| FROG | Frog chorus | — | Nuisance |
| GLGN | Northern pygmy-owl | *Glaucidium gnoma* | Owl |
| HOSA | Human speech | — | Nuisance |
| HYPI | Pileated woodpecker call | *Dryocopus pileatus* | Woodpecker |
| INSP | Barred owl inspection call | *Strix varia* | Owl |
| IXNA | Varied thrush | *Ixoreus naevius* | Songbird |
| MEKE | Western screech-owl | *Megascops kennicotti* | Owl |
| MYTO | Townsend's solitaire | *Myadestes townsendi* | Songbird |
| NUCO | Clark's nutcracker | *Nucifraga columbiana* | Songbird |
| OCPR | American pika | *Ochotona princeps* | Mammal |
| ORPI | Mountain quail | *Oreortyx pictus* | Game bird |
| PAFA | Band-tailed pigeon | *Patagioenas fasciata* | Game bird |
| PECA | Canada jay | *Perisoreus canadensis* | Corvid |
| PHNU | Common poorwill | *Phalaenoptilus nuttallii* | Other bird |
| PIMA | Spotted towhee | *Pipilo maculatus* | Songbird |
| POEC | Chickadee song | *Poecile* sp. | Songbird |
| PSFL | Flammulated owl | *Psiloscops flammeolus* | Owl |
| SHOT | Gunshot | — | Nuisance |
| SITT | Nuthatch | *Sitta* sp. | Songbird |
| SPRU | Sapsucker drum | *Sphyrapicus* spp. | Woodpecker |
| STOC | Spotted owl location call | *Strix occidentalis* | Owl |
| STOC_IRREG | Spotted owl series call | *Strix occidentalis* | Owl |
| STVA | Barred owl eight-note call | *Strix varia* | Owl |
| STVA_IRREG | Barred owl series call | *Strix varia* | Owl |
| TADO1 | Douglas' squirrel rattle | *Tamasciurus douglasii* | Mammal |
| TADO2 | Douglas' squirrel chirp | *Tamasciurus douglasii* | Mammal |
| TAMI | Chipmunk chirp | *Neotamias* sp. | Mammal |
| TUMI | American robin whinny | *Turdus migratorius* | Songbird |
| WHIS | Strix owl contact whistle | *Strix* sp. | Owl |
| YARD | Yarder (machine) | — | Nuisance |
| ZEMA | Mourning dove | *Zenaida macroura* | Game bird |



