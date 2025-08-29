# Learned Off-Grid Imager for Low-Altitude Economy with Cooperative ISAC Network (LAEImager)

This is the codes for the paper "**Learned Off-Grid Imager for Low-Altitude Economy with Cooperative ISAC Network**" in *IEEE Transactions on Wireless Communications*. Its conference version has been presented in VTC2025-Spring, Oslo, Norway, June 2025.
Arxiv link: https://arxiv.org/abs/2506.07799

This paper realizes low-altitude surveillance by employing hybrid model and data knowledge.
Model-based on-grid imager in MATLAB and learned physics-embedded off-grid imager in Python are provided.

# MATLAB version and Python Packages

- MATLAB: R2021b

- python==3.11.5
- pytorch==2.0.1
- numpy==1.23.5
- wandb==0.17.9

# Implementation

MATLAB: Run `main.m`

Python: Run `train.py` and `pred.py` for training and testing

Note: To run Python codes, a dataset in `.mat` format should be first generated based on the MATLAB codes. Potential modifications of the MATLAB codes are needed for specfic dataset generation requirements. The dataset must include *input data*, which is calculated by multiplicating Hermitian transpose of the sensing matrix and CSI measurement vector and named by 'data_herm_image' in `dataloader` at `./Pythpn/utils.py`, and *output data*, which is the ground truth image and named by 'data_true_image' in `dataloader` at `./Pythpn/utils.py`. For more information, refer to the related papers.

# Citation

```

@article{huang2025learned,
  title={Learned Off-Grid Imager for Low-Altitude Economy with Cooperative {ISAC} Network},
  author={Huang, Yixuan and Yang, Jie and Xia, Shuqiang and Wen, Chao-Kai and Jin, Shi},
  journal={IEEE Trans. Wireless Commun.},
  year={early access, Aug. 2025},
  publisher={IEEE}
}

@inproceedings{huang2025cooperative,
  title={Cooperative {ISAC} Network for Off-Grid Imaging-based Low-Altitude Surveillance},
  author={Huang, Yixuan and Yang, Jie and Wen, Chao-Kai and Xia, Shuqiang and Li, Xiao and Jin, Shi},
  booktitle={Proc. IEEE 101st Veh. Technol. Conf. (VTC2025-Spring)},
  pages={1--7},
  year={Jun. 2025}
}
```

