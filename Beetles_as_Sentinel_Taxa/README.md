# Imageomics HDR Scientific Mood Challenge Sample

This repository contains sample training code and submissions for the [2025 HDR Scientific Mood (Modeling out of distribution) Challenge: Beetles as Sentinel Taxa](https://www.codabench.org/competitions/9854/). It is designed to give participants a reference for both working on the challenge, and also the expected publication of their submissions following the challenge (i.e., how to open-source your submission).

## Repository Structure

For your repository, you will want to complete the structure information below and add other files (e.g., training code):
```
submission/
  <model weights>
  model.py
  requirements.txt
```
We also recommend that you include a [CITATION.cff](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files) for your work.

**Note:** If you have requirements not included in the [whitelist](https://github.com/Imageomics/HDR-SMood-challenge/blob/main/ingestion_program/whitelist.txt), please check the [issues](https://github.com/Imageomics/HDR-SMood-challenge/issues) on the [challenge GitHub](https://github.com/Imageomics/HDR-SMood-Challenge) to see if someone else has requested it before making your own issue.

### Structure of this Repository
```
HDR-SMood-challenge-sample/
│
├── baselines/
│   ├── submissions/
│   │   └── <MODEL NAME>/
│   │       ├── model.pth
│   │       ├── model.py
│   │       ├── train.py
│   │       └── requirements.txt
│   └── training/
│       └── <MODEL NAME>/
│           ├── evaluation.py
│           ├── model.py
│           ├── train.py
│           └── utils.py
├── notebooks/
|   └── train-data-exploration.ipynb
├── .gitignore
├── LICENSE
└── README.md
```

> [!IMPORTANT]  
> Do not zip the whole folder submission folder when submitting your model to Codabench. ***Only*** select the `model.py` and relevant weight and requirements files to make the tarball.

#### Training Data Exploration

This repository also includes `train-data-exploration.ipynb` which loads the training data from Hugging Face to perform various data analytics. Specifically, it looks at distributions of images, species, SPEI values, etc. over the various domains. To run this notebook, first clone this repository and create a fresh conda environment, then install the requirements file:

```console
conda create -n beetle-sample -c conda-forge pip -y
conda activate beetle-sample
pip install -r requirements.txt
jupyter lab
```

## Installation & Running (for Training)

### Installation
If you have `uv` simply run `uv sync`, otherwise you can use the `requirements.txt` file with either `conda` or `pip`.

### Training
An example training run can be executed by running the following:
```
python baselines/training/train.py
```

with `uv` do:
```
uv run python baselines/training/train.py
```

The training data is available on the Imageomics Hugging Face: [Sentinel Beetles Dataset](https://huggingface.co/datasets/imageomics/sentinel-beetles). 

### Evaluation
Aftering training, you can locally evaluate your model by running the following:
```
python baselines/training/evaluation.py
```

with `uv` do:
```
uv run python baselines/training/evaluation.py
```

## References
Baselines built off of BioCLIPv2 and Dinov2:

Gu, Jianyang, et al. "Bioclip 2: Emergent properties from scaling hierarchical contrastive learning." arXiv preprint arXiv:2505.23883 (2025).

Oquab, Maxime, et al. "Dinov2: Learning robust visual features without supervision." arXiv preprint arXiv:2304.07193 (2023).

[Sample repo from Y1 challenge](https://github.com/Imageomics/HDR-anomaly-challenge-sample).

