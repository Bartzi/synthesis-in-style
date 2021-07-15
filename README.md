# Synthesis in Style

This repository contains the code for our paper "Synthesis in Style: Semantic Segmentation of Historical Documents using Synthetic Data".
You can read a Preprint [on Arxiv](https://arxiv.org/abs/2107.06777).

This document provides all information you need to do your own experiments with our proposed approach.
Currently, our method is not simple to use and to understand.
We are working on improving the method to reduce the required manual intervention to a minimum!

The current state of the work can be seen as WIP, it might not be simple to use.
Furthermore, it might also not provide the best results, yet.
We are working on improving the method.
If you have any good ideas, please let us know!

## Installation

Installation and preparation of our code for use is simple:
1. Make sure that you have an Nvidia GPU in your machine!
1. Clone the Repo
1. Build a Docker image:
   ```shell
   docker build --build-arg UNAME=$(whoami) --build-arg UID=$(id -u) --build-arg GID=$(id -g) -t synthesis_in_style:$(whoami) .
   ```
1. Done!

## Building a Model

If you wish to build a model on your own, you can do so by performing the following steps:
1. Gather your dataset
1. Prepare your dataset for training
1. Train a StyleGAN Model on your Dataset
1. Synthesize images and extract clusters from feature activations
1. Annotate extracted clusters
1. Adapt semantic segmentation algorithm
1. Synthesize training dataset for segmentation network
1. Train semantic segmentation network
1. Profit!?

### Gathering a Dataset

You can choose whatever dataset you like.
As example for this README, we will use the [HORAE](https://github.com/oriflamms/HORAE/) Dataset. 
You can get the dataset using the script `download_horae_dataset.py` in the directory `scripts`.
Just run `python3 download_horae_dataset.py` to get usage information for this script.
The download of the full dataset might take some time since it contains more than 2TB of data!

### Preparing Dataset for Training

To prepare your dataset, you have multiple options.
You can crop patches or use the available images directly.
If you wish to crop patches, you can use the script `create_stylegan_train_dataset.py` in the `scripts` directory.
Assuming you saved the HORAE dataset under `/data/horae`, you could run it like so:
```shell
python3 create_stylegan_train_dataset.py /data/horae /data/horae_crops 50000
```
This will create patches of size `256 x 256` in the directory `/data/horae_crops`.
Please run `python3 create_stylegan_train_dataset.py -h` for further options.

At the end of this step you'll need a `json` file that contains a list of all image files, you want to use for the training
of StyleGAN.
Here, it makes sense to already split into a train and validation split!
The paths of the files need to be relative to the location of the `json` file.
In our case the json file resides in `/data`.
Thus, the content of the file would be the following:
```json
[
  "horae_crops/0.png",
  "horae_crops/1.png",
  "[...]"
]
```

### Train StyleGAN

Now that you have a prepared dataset, we can train a StyleGAN model.
We recommend to use a machine with 8 GPUs for this.
However, one GPU also suffices.
To train a StyleGAN model, use the script `train_stylegan_2.py`.
Let's assume we want to train on a machine with 8 GPUs.
We could run the training out of the box like this:
```shell
python3 -m torch.distributed.launch --nproc_per_node=8 train_stylegan_2.py configs/stylegan_256px.yaml --images /data/train.json --val-images /data/val.json -l <name of your series of experiments> -ln <name of this experiment>
```
There are further options you can use to adapt the training.
Run `python3 train_stylegan_2.py -h` to get an overview of all options.
You can modify training hyperparameters by adapting the config yaml file (see examples in the dir `configs`).
You can monitor the train progress using Weights and Biases.
To log to your account, you can set the environment variable `WANDB_API_KEY` to your API Key.

You can find the resulting model and training logs in the dir `logs` that will appear once you started the training. 

### Synthesize Images and Extract Clusters

Once you obtained a StyleGAN Model, you'll need to perform all necessary steps to build the segmentation branch of your model.
You'll have to do the following steps for each StyleGAN model again because each model is different.

To perform this step, you can use the script `create_semantic_segmentation.py`.
Assuming, we trained a model, which was saved in `logs/training/horae`, we could do it the following way:
```shell
python3 create_semantic_segmentation.py logs/training/horae/checkpoints/100000.pt
```
There are further options, you can view them with `python3 create_semantic_segmentation.py -h`.

Once this script is done, you will find the result in the directory `logs/training/horae/semantic_segmentation`.
In its default configuration, the script produces several clusterings with a different number of clusters.
The number of clusters ranges from 3 to 24 in the default setting.
You can see the clustering result by looking at the `.png` files in the segmentation dir.

### Annotation of Clusters

Once you obtained clusters you will need to find the correct number of clusters for your use case.
You should do so by inspecting the images in the segmentation dir.
We found that `20` is a reasonable number of cluster to distinguish between printed and handwritten text.
Once you know the number of clusters, you can use our annotation tool to easily annotate the clusters.
You can find the tool in the directory `semantic_labeller`.
Before starting the tool, you'll need to adapt the config in `semantic_labeller/configs/server_config.json`.
Here, you'll need to enter the **absolute** path to the directory with your cluster results from the last step.
The number of clusters you wish to annotate and also a json file where you define the class names and the color 
for each class name (see our examples in the `config` dir).
You can then run the server by running:
```shell
export FLASK_APP=app
flask run
```
This will run a flask app on your localhost.
If you are running in our Docker Container you might need to forward port 5000.
Connect to the web app using your WebBrowser and start annotating.
We prepared a small [video](TODO) (To be done!) that shows you how to use the tool.

### Adaptation of the Algorithm

Now that you have annotated the clusters, you'll need to perform some programming and adapt the algorithm that is used
to combine feature layers to get a meaningful segmentation branch.
You can find an example in the dir `segmentation`.
Here, the file `gan_segmenter.py` is the base class that you can use for the adaption of the algorithm.
Further Instructions: **TODO**

### Synthesize Training Data

Now, you can finally synthesize some training data.
You can use the script `create_dataset.py` already introduced in the last section.
Just run it and create a nice dataset of around 100 000 (or more) images.

### Train Semantic Segmentation Network

Let's assume your synthetic dataset is in `/data/horae_train`.
You can now use the `train_segmentation.py` script to train a model for semantic segmentation.
For example if you wish to train on two GPUs:
```shell
python3 -m torch.dsitributed.launch --n_proc_per_node 2 train_segmentation.py configs/segmenter.yaml --images /data/horae_train/train.json --val-images /data/horae_train/val.json --coco-gt /data/horae_train/coco_gt.json -ln segmentation_train
``` 
Check further options with `python3 train_segmentation.py -h`.

### Segmenting Original Images

You can now use the segmentation model to segment real images :tada:.
To do so, use the `segment_image.py` script.
Using the model trained in the last section on some hypothetical document images residing in `/data/images`:
```shell
python3 segment_image.py logs/training/segmentation_train/checkpoints/100000.pt /data/images/image_0.png --output-dir /data/output
```

### Questions

If you have questions, feel free to reach out and open an Issue!

### Citation

If you find the code useful, please cite our paper!

```bibtex
@misc{bartz2021synthesis,
      title={Synthesis in Style: Semantic Segmentation of Historical Documents using Synthetic Data}, 
      author={Christian Bartz and Hendrik Rätz and Haojin Yang and Joseph Bethge and Christoph Meinel},
      year={2021},
      eprint={2107.06777},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
