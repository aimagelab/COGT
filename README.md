# Causal Graphical Models for Vision-Language Compositional Understanding

## Abstract

Recent work has empirically shown that Vision-Language Models (VLMs) struggle to fully understand the compositional properties of the human language, usually modeling an image caption as a "bag of words". As a result, they perform poorly on compositional tasks, which require a deeper understanding of the different entities of a sentence (subject, verb, etc.) jointly with their mutual relationships in order to be solved. In this paper, we model  the dependency relations among textual and visual tokens using a **Causal Graphical Model (CGM)**, built using a *dependency parser*, and we train a decoder conditioned by the VLM visual encoder. 
Differently from standard autoregressive or parallel predictions,
our decoder's generative process is partially-ordered following the CGM structure. This structure encourages the decoder to learn only the main causal dependencies in a sentence discarding spurious correlations.
Using extensive experiments on five compositional benchmarks, we show that our method significantly outperforms all the state-of-the-art compositional approaches, 
usually by a large margin, and it also improves over  methods trained  using much larger datasets.

## Create the environment
```
conda create -y -n "cogt" python=3.9.13
conda activate cogt
pip install -r requirements.txt
```

# Edit some files before running the code based on your local path
- `xvlm_dir/configs/config_swinB_224.json` Edit the field "ckpt" with your local path.
- `paths.py` You need to customize the paths once you have downloaded the data.


## Dataset 
We evaluate our model with five different dataset. Please download it from the original source.
- [ARO](https://arxiv.org/pdf/2210.01936)
- [SugarCrepe](https://arxiv.org/pdf/2306.14610)
- [VL-CheckList](https://arxiv.org/pdf/2207.00221)
- [ColorSwap](https://arxiv.org/pdf/2402.04492)

We propose an additional benchmark commonly used to evaluate the ability of open-vocabulary object detectors to discern fine-grained object properties. We use it as a compositional benchmark to challenge models in recognizing attributes of common objects that rarely appear in the image foreground:
- [FG-OVD](https://arxiv.org/pdf/2311.17518)

## Training
We train our models on custom COCO split dataset defined by [NegCLIP](https://arxiv.org/pdf/2210.01936).
Use these scripts to train the models:
```
scripts/COGT_CLIP_train.sh
scripts/COGT_X-VLM_train.sh
```

## Inference
To evaluate our model:
```
scripts/COGT_CLIP_inference.sh
scripts/COGT_X-VLM_inference.sh
```

## COGT Weights
- [COGT-CLIP_ViT/B-32] The checkpoint is soon available.
- [COGT-X-VLM] The checkpoint is soon available.

Soon available also on HuggingFace Hub 🤗
