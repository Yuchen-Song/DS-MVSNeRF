# DS-MVSNeRF
This repository combines two pieces of work, [MVSNeRF](https://apchenstu.github.io/mvsnerf/) and [DS-NeRF](https://www.cs.cmu.edu/~dsnerf/). To be more specific, I added colmap depth supervision on the finetuning function in the original MVSNeRF, and got enhanced outcomes.  
### MVSNeRF
![MVSNeRF](./demos/ft_t-rex_36views.gif)
### DS-MVSNeRF
![MVSNeRF](./demos/DSft_t-rex_36views.gif)
***

## Installation

Tested on Ubuntu 20.04 + Pytorch 1.11.0 + Pytorch Lignting 1.3.5

### Install environment:
```
conda create -n dsmvsnerf python=3.9
conda activate dsmvsnerf
pip install conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

### Install [mogrify](https://imagemagick.org/script/mogrify.php)
### Install [colmap](http://colmap.github.io/)

## Prepare Your Dataset
### Images preparation
Notice that this work can only support images scale by ***16:9*** and ***4:3***, using [ffmpeg](https://ffmpeg.org/)  to resize them if your images are not in correct resolusion scale.   

First, place your scene directory somewhere (better in the same project directory). See the following directory structure for an example:
```
├── nerf_llff_data
│   ├── {scene name}
│   ├── ├── images
│   ├── ├── ├── 0001.png
│   ├── ├── ├── 0002.png
```
You can also generate images from video input by using [ffmpeg](https://ffmpeg.org/)  

For those who wants to test on public dataset, download `nerf_llff_data.zip` from [here](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1)

### Poses generation
To generate the poses and sparse point cloud:
```
python imgs2poses.py <your_scenedir>
```
## Training
This work uses pretrained model from MVSNeRF, for more details of model training, please refer to [MVSNeRF](https://github.com/apchenstu/mvsnerf)

## Finetuning
This work focus on llff datatype, which can be generated from your own images.

```
CUDA_VISIBLE_DEVICES=0  python train_finetuning_pl.py  \
    --datadir /path/to/your/dataset/{scene_name} \
    --expname {scene_name}  --with_rgb_loss  --batch_size 512  \
    --num_epochs 1 --imgScale_test 1.0  --pad 24 \
    --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1 --depth_loss
```


## Rendering
To render videos only, please add `--render_only` as argument. You should also include `--is_finetuned` if you want to render from a finetuned model.

## Citation
If you find my code helps, please consider staring and forking.

My code is adapted from [**MVSNeRF**](https://github.com/apchenstu/mvsnerf) and [**DS-NeRF**](https://github.com/dunbar12138/DSNeRF).
