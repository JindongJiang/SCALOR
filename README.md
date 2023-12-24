## SCALOR

This repository is the official implementation of ["SCALOR: Generative World Models with Scalable Object Representations"](https://arxiv.org/abs/1910.02384) by [Jindong Jiang](https://www.jindongjiang.me)\*, [Sepehr Janghorbani](http://janghorbani.net)\*, [Gerard de Melo](http://gerard.demelo.org), and [Sungjin Ahn](https://sungjinahn.com/); accepted at the *International Conference on Learning Representations* (*ICLR*) 2020. [Project Website](https://sites.google.com/view/scalor/home)

![Architecture](./figures/architecture.png)


## Requirements

To install requirements:

```bash
conda env create -f environment.yml
```

To activate:

```bash
conda activate scalor_env
```

## Dataset

The "Grand Central Station" dataset can be downloaded [here](https://bit.ly/36tckTa). After downloading the file, extract the dataset using this command:

```bash
tar -xf grandcentralframes.tar.gz -C /path/to/dataset/
```



## Training

To train SCALOR with default settings, run this command:

```bash
python train.py --data-dir /path/to/dataset/
```



## Results

![toy](./figures/toy.gif)

![natural-scene](./figures/natural-scene.gif)

## Using SCALOR in your project

### Foreground not working

If you find the background module explains everything in the image and the foreground module is turned off, first check the following two settings:

1. The `num_cell_h` and `num_cell_w` in [common.py](common.py). If the objects in the scene are densely positioned in a local area, the number of cells (num_cell_h and num_cell_w) should be larger to provide enough cells in that local area.

2. The `max_num_obj` in [common.py](common.py). In the early training stage, this number is higher the better (smaller than the total number of cells) since it allows more activated cells to accelerate the foreground training. Feel free to reduce it later.

Additionally, I also added the following two settings in the code. Feel free to try any of them:

1. Using a weaker background decoder, one option is to set the `using_bg_sbd` flag to True in [common.py](common.py).

2. Using a training curriculum in the early training stage. This can be done by setting the `phase_bg_alpha_curriculum` to True in [common.py](common.py).

Feel free to let me know if you face any other problems when adopting SCALOR in your project.

## Citation

```
@inproceedings{JiangJanghorbaniDeMeloAhn2020SCALOR,
  title={SCALOR: Generative World Models with Scalable Object Representations},
  author={Jiang, Jindong and Janghorbani, Sepehr and De Melo, Gerard and Ahn, Sungjin},
  booktitle={International Conference on Learning Representations},
  year={2019}
}
```

