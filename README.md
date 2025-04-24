## 
<h1 align="center">💡LucidPPN: Unambiguous Prototypical Parts Networks for User-centric Interpretable Computer Vision</h1>

<div align="center">
<a href="https://www.eml-munich.de/people/mateusz-pach">Mateusz Pach</a>,
<a href="https://neuroscience.ips.uj.edu.pl/team/koryna-lewandowska">Koryna Lewandowska</a>,
<a href="https://matinf.uj.edu.pl/en_GB/pracownicy/wizytowka?person_id=Jacek_Tabor">Jacek Tabor</a>,
<a href="https://bartoszzielinski.github.io/">Bartosz Zieliński</a>,
<a href="https://dawrym.github.io/">Dawid Rymarczyk</a>
<br>
<br>

[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-%3CCOLOR%3E.svg)](https://openreview.net/pdf?id=BM9qfolt6p)
</div>

<h3 align="center">Abstract</h3>

<p align="justify">
Prototypical parts networks combine the power of deep learning with the explainability of case-based reasoning to make accurate, interpretable decisions. They follow the this looks like that reasoning, representing each prototypical part with patches from training images. However, a single image patch comprises multiple visual features, such as color, shape, and texture, making it difficult for users to identify which feature is important to the model.
To reduce this ambiguity, we introduce the Lucid Prototypical Parts Network (LucidPPN), a novel prototypical parts network that separates color prototypes from other visual features. Our method employs two reasoning branches: one for non-color visual features, processing grayscale images, and another focusing solely on color information. This separation allows us to clarify whether the model's decisions are based on color, shape, or texture. Additionally, LucidPPN identifies prototypical parts corresponding to semantic parts of classified objects, making comparisons between data classes more intuitive, e.g., when two bird species might differ primarily in belly color.
Our experiments demonstrate that the two branches are complementary and together achieve results comparable to baseline methods. More importantly, LucidPPN generates less ambiguous prototypical parts, enhancing user understanding.</p>
<br>
<div align="center">
    <img src="assets/teaser.svg" alt="Teaser" width="1000">
</div>

---
### Setup
Install required PIP packages.
```bash
conda create --name lucidppn --file requirements.txt
```
Then prepare datasets with the following steps.
- download CUB from https://www.vision.caltech.edu/datasets/cub_200_2011/,
- download CARS from https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset,
- download FLOWER from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/,
- download DOGS from http://vision.stanford.edu/aditya86/ImageNetDogs/,
- run `preprocess_data/prepare_dogs.py` to organize DOGS,
- run `preprocess_data/prepare_flowers.py` to organize FLOWER,
- convert CARS into ImageDataset format i.e into two directories: `train/` and `test/` with subdirectory for each class
- cutout CUB images as in https://github.com/cfchen-duke/ProtoPNet,
- set dataset locations in `PIPNet/util/data.py` and `MetiNet/util/data.py` and `run_training.sh` files.

### Running Experiments

To train and evaluate the LucidPPN run `MetiNet/run_training.sh` by uncommenting desired experiments.

To train and evaluate the PIP-Net run `PIPNet/run_training.sh` by uncommenting desired experiments.

For LucidPPN, remember to first run `part_detection/run_training.sh` to generate segmentation masks.

Hint: Before running you may want to update wandb project and entity names found in the source code.
### Citation
```bibtex
@inproceedings{
    pach2025lucidppn,
    title={Lucid{PPN}: Unambiguous Prototypical Parts Network for User-centric Interpretable Computer Vision},
    author={Mateusz Pach and Koryna Lewandowska and Jacek Tabor and Bartosz Micha{\l} Zieli{\'n}ski and Dawid Damian Rymarczyk},
    booktitle={The Thirteenth International Conference on Learning Representations},
    year={2025},
    url={https://openreview.net/forum?id=BM9qfolt6p}
}
```
