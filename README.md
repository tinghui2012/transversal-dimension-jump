# Transversal dimension jump for product qLDPC codes
This repo contains the code for simulations in "Transversal dimension jump for product qLDPC codes"
Preprint: https://arxiv.org/abs/2510.07269

Includes:
- constructions of product qLDPC codes, including HGP and lifted product
- homomorphic CNOT construction
- CCZ operator construction
- memory experiment
- teleporation experiment
- CCZ error detection experiment

## Repository structure

```text
.
├── README.md
├── data
│   ├── gamma_0_dict.pkl
│   ├── gamma_1_dict.pkl
│   ├── gamma_2_dict.pkl
│   ├── gamma_3_dict.pkl
│   ├── hx_dict_2d.pkl
│   ├── hx_dict_3d.pkl
│   ├── hz_dict_2d.pkl
│   └── hz_dict_3d.pkl
├── notebooks
│   ├── 0_code_generation.ipynb
│   ├── 1_memory_experiments.ipynb
│   ├── 2_teleportation_experiments.ipynb
│   └── 3_ccz_error_detection_simulations.ipynb
├── requirements_py312.txt
├── requirements_py37.txt
└── src
    ├── BP_codes_sage.py
    ├── BP_cup_prod.py
    ├── CCZCircuit.py
    ├── CircuitScheduling.py
    ├── CodeGeneration.py
    ├── Decoders_SpaceTime.py
    ├── DistanceEst.py
    ├── ErrorPlugin.py
    ├── PCCZMatrix.py
    ├── QECCircuits.py
    ├── SingleStageDecoding.py
    ├── TeleportationCircuit.py
    └── utilities.py
```
## Requirements

- 0_code_generation.ipynb: Python 3.12, requirements_py312.txt, SageMath, GAP, QDistRnd
- 1_memory_experiments.ipynb: Python 3.12, requirements_py312.txt
- 2_teleportation_experiments: Python 3.7, requirements_py37.txt
- 3_ccz_error_detection_simulations: Python 3.12, requirements_py312.txt

## Citing

```bibtex
@misc{li2025transversaldimensionjumpproduct,
      title={Transversal dimension jump for product qLDPC codes}, 
      author={Christine Li and John Preskill and Qian Xu},
      year={2025},
      eprint={2510.07269},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2510.07269}, 
}
```
