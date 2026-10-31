# Data License & Attribution

All **source code** in this repository is licensed under the [MIT License](LICENSE).

The small committed demo bundle (`web/data/` and `coord/artifacts/`) contains sample laparoscopic cholecystectomy frames and evaluation metadata derived from **CholecSeg8k** (itself derived from **Cholec80**), which is licensed under the **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License ([CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/))**.

## Required Attribution (CC BY-NC-SA 4.0)

1. **CholecSeg8k** (held-out video frames in `web/data/frames/` and `web/data/dense/`, and ground-truth evaluation masks in `web/data/gt_masks/` used strictly post-hoc for scoring):
   - W.-Y. Hong, C.-L. Kao, Y.-H. Kuo, J.-R. Wang, W.-L. Chang, and C.-S. Shih, *"CholecSeg8k: A Semantic Segmentation Dataset for Laparoscopic Cholecystectomy Based on Cholec80,"* arXiv:2012.12453, 2020.
   - Dataset URL: <https://www.kaggle.com/datasets/newslab/cholecseg8k>
   - License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

2. **Cholec80** (underlying surgical videos):
   - A. P. Twinanda, S. Shehata, D. Mutter, J. Marescaux, M. de Mathelin, and N. Padoy, *"EndoNet: A Deep Architecture for Recognition Tasks on Laparoscopic Videos,"* IEEE Transactions on Medical Imaging, vol. 36, no. 1, pp. 86–97, 2017.
   - Dataset URL: <http://camma.u-strasbg.fr/datasets>
   - License: [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)

3. **Endoscapes2023 & CholecT50** (external evaluation benchmarks referenced in `out/results.json` and `out/student.json`; raw frames are not redistributed in this repository):
   - CAMMA Research Group, University of Strasbourg & IHU Strasbourg (<http://camma.u-strasbg.fr/datasets>), licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## License Terms Summary for Demo Data

Under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/), the surgical video frames and ground-truth masks in `web/data/` may be shared and adapted for **non-commercial** research and demonstration purposes only, provided appropriate credit is given, a link to the license is included, changes (such as JPEG compression and AI annotation overlays) are indicated, and any redistributed derivatives of the dataset are shared under the same CC BY-NC-SA 4.0 license.
