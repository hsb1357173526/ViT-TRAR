# Vision Transformer
* We add [TRAR](https://github.com/rentainhe/TRAR-VQA) to the repo [ViT-pytorch](https://github.com/rentainhe/ViT.pytorch) in ImageNet1K finetune.

* TRAR: Routing the Attention Spans in Transformer
for Visual Question Answering [[pdf](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhou_TRAR_Routing_the_Attention_Spans_in_Transformer_for_Visual_Question_ICCV_2021_paper.pdf)]
## Note
* We have only done ViT-B/16 + TRAR experiments in ImageNet21K pretraning + ImageNet2012 finetune.
* Add [test.py](test.py), you can test one image to output its label index.

## Usage
<details>
<summary> <b> Data Preparation </b> </summary>

### Download Pre-trained model (Google's Official Checkpoint)
* [Available models](https://console.cloud.google.com/storage/vit_models/): ViT-B_16(**85.8M**)
```python
# imagenet21k pre-train
wget https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz
```
Put `ViT_B_16.npy` into `./output/` folder after downloading.

### Imagenet2012 dataset preparation
Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

</details>

<details>
<summary> <b> Train Model </b> </summary>

#### Run `train.sh` for quick start
```
$ bash train.sh
```
you can customize `train.sh` by yourself, relative configs:
- `DATASET`: we only support `cifar10`, `cifar100`, `imagenet` now , but our experiment did in iamgenet.
- `MODEL_TYPE`: ViT-B_16, we haven't test other models.
- `IMG_SIZE`: input image size (224 or 384).
- `NAME`: name for this experiment.
- `GPUS`: choose the specific GPUs for training.
- `TRAIN_BATCH_SIZE`: batch size for training.
- `EVAL_BATCH_SIZE`: batch size for evaluation.
- `GRAD_STEPS`: accumulation gradient steps for saving gpu memory cost.
- `NUM_STEPS`: total training steps.
- `WARMUP_STEPS`: warm up steps.
- `DECAY_TYPE`: lr-scheduler, we only support `linear` and `cosine` now.
- `RESUME_PATH`: checkpoint path for resume training.
- `PRETRAINED_DIR`: path to load pretrained weight.

CIFAR-10 and CIFAR-100 are automatically download and train. In order to use a different dataset you need to customize [data_utils.py](./utils/data_utils.py).

The default batch size is 512. When GPU memory is insufficient, you can proceed with training by adjusting the value of `GRAD_STEPS` in [train.sh](train.sh).

You should change image_scale to the square root of image_size. (image_size should be a number's square) 

Also you can change TRAR's orders in [model/config.py](model/config.py) to get various mask weights.
```python
def get_b16_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12
    config.transformer.num_layers = 12
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.transformer.orders=[0,1,2] # You can DIY orders here.
    config.transformer.pooling='avg'
    config.transformer.img_scale = 24# You should change image_scale here.
    config.classifier = 'token'
    config.representation_size = None
    return config
```

#### Run `train_fp16.sh` for mixed precision training
Also can use [Automatic Mixed Precision(Amp)](https://nvidia.github.io/apex/amp.html) to reduce memory usage and train faster
```
$ bash train_fp16.sh
```
Additional configs:
- `FP16_OPT_LEVEL`: mixed precision training level from {`O0`, `O1`, `O2`, `O3`}

</details>


## Results
Although TRAR+ViT takes a long time to train, but we can observe that it(about 3K step) can convergence more easlily than Vit(about 6K step).

Note that it just is a small attempt experiment, there can be many problems in this idea.

But we will improve it in the future and learn from it in latest experiments.


|image resolution|  orders    |  dataset     | total_steps /warmup_steps | acc   |  cost_time    |  memory       | 
|:--------------:|:----------:|:------------:|:-------------------------:|:-----:|:-------------:|:-------------:|
| 224*224        |     -      | imagenet2012 |          20000/1000       | 81.62%|    2d0h58m24s |   3414MiB     |
| 224*224        | [0,1,2,3]  | imagenet2012 |          20000/1000       | 81.60%|    4d9h51m1s  |   4457MiB     |
| 384*384        |     -      | imagenet2012 |          20000/1000       | 83.62%|    3d0h6m49s  |   5903MiB     |
| 384*384        | [0,1,2]    | imagenet2012 |          20000/1000       | 83.77%|   17d5h47m49s |    12061MiB   |
| 384*384        | [0,1,2,3]  | imagenet2012 |          20000/1000       | 83.71%|    18d17h8m58s|   14055MiB    |

#### Run `test.py` for a easy test
Put the picture that you want to test to `./pic` folder. 
```python
pic_dir='./pic/1.jpg'
```
You can use model that your trained to test or download our trained weight [TRAR+ViT weight](https://pan.baidu.com/s/19ESXrnHA8uv-kSE0fymBlQ) `code:h8d6` and change the path:



```python
ckpt = torch.load('./ViT_TRAR_012.ckpt', map_location=torch.device("cpu"))
```
Run the test stage:
```python
CUDA_VISIBLE_DEVICES=0 python test.py
```




We provide three types: 
* ViT_TRAR_224(order=[0,1,2,3])
* ViT_TRAR_384(order=[0,1,2])
* ViT_TRAR_384(order=[0,1,2,3])
