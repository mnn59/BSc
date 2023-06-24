<div align="center">
 
<!--![logo](https://github.com/mnn59/BSc/blob/main/assets/logo.jpeg)  -->

<h1 align="center"><strong>🩺 Medical Image Segmentation System (MISS) <h6 align="center">My BSc project</h6></strong></h1>

![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2.0+-red?style=for-the-badge&logo=pytorch)
![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
[![GitHub Issues](https://img.shields.io/github/issues/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch.svg?style=for-the-badge)](https://github.com/mnn59/BSc/issues)

</div>

----
 
## 📚 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Getting Started](#getting-started)
  - [Download ViT model](#vit-model)
  - [Prepare Data](#prepare-data)
  - [Train/Test](#traintest)
  - [Running the Gradio WebApp](#running-the-app)
- [Acknowledgements](#acknowledgements)

----

## 📌 Overview <a name="overview"></a>
An end-to-end Computer Vision project focused on the topic of <a href="https://en.wikipedia.org/wiki/Image_segmentation" target="_blank">Image Segmentation</a> (specifically Semantic Segmentation). Although this project has primarily been built with the <a href="https://gradio.app/" target="_blank">Gradio</a>.

Rapid advances in the field of medical imaging are revolutionizing medicine. For example, the diseases diagnosis with the help of computers, where the segmentation of medical images plays an important role, has become more accurate. Although CNN-based methods have achieved excellent performance in recent years, but due to the intrinsic locality of convolution operations, they cannot learn explicit global and long-range semantic information well. Given the increased interest in self-attention mechanisms in computer vision and their ability to overcome this problem, the TransUNet architecture was proposed as the first medical image segmentation framework using Vision Transformer as a strong encoder in a U-shaped architecture.

TransUNet achieves good results compared to different architectures; therefore, in this project, we use it as the base model that has a hybrid CNN-Transformer architecture. this architecture is able to leverage both detailed high-resolution spatial information from CNN features and the global context encoded by Transformers. All experiments are conducted on Kvasir-SEG, CVC-ClinicDB and Ph2 datasets. First, we reproduce the results in the original paper, and then we proceed to improve the architecture by making appropriate changes and check the results. Some of these changes have been successful and others have been unsuccessful. Finally, we created a web-based system based on the new architecture.

----

## 💫 Demo <a name="demo"></a>
 <p align="center">
  <img width="60%" src="https://github.com/mnn59/BSc/blob/main/assets/demo.jpg">
 </p>

---

## 🚀 Getting Started <a name="getting-started"></a>

### Download Google pre-trained ViT model <a name="vit-model"></a>
* [Get models in this link](https://console.cloud.google.com/storage/vit_models/): R50-ViT-B_16
```bash
wget https://storage.googleapis.com/vit_models/imagenet21k/R50-ViT-B_16.npz &&
mkdir ../model/vit_checkpoint/imagenet21k &&
mv R50-ViT-B_16.npz ../model/vit_checkpoint/imagenet21k/R50-ViT-B_16.npz
```

### Prepare data <a name="prepare-data"></a>

Go to these links to download these 3 datasets, which used in this project:
- <b>CVC-ClinicDB:</b> <a href="https://polyp.grand-challenge.org/CVCClinicDB/">link</a>
- <b>Kvasir-SEG:</b> <a href="https://datasets.simula.no/kvasir-seg/">link</a>
- <b>Ph2:</b> <a href="https://www.fc.up.pt/addi/ph2%20database.html">link</a>

### Environment

Please prepare an environment with python=3.9, and then use the command "pip install -r requirements.txt" for the dependencies.


### 💻 Train/Test <a name="traintest"></a>
 
Run the train script on each dataset. Use `--arch` to choose architecture between <b>Transunet</b> and <b>Transcascade</b>.

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --dataset Kvasir --arch Transunet
```

Run the test script on each dataset. Use the same architecture you used for training.

```bash
python test.py --dataset Kvasir --arch Transunet
```


### 🤖 Running the Gradio WebApp <a name="running-the-app"></a>

After running this script, You will be directed to the app page via a unique link.

 ```bash
python gradio_test.py
```

----


## 👏 Acknowledgements <a name="acknowledgements"></a>
We are very grateful for these excellent works [TransUNet](https://github.com/Beckschen/TransUNet/) and [CASCADE](https://github.com/SLDGroup/CASCADE), which have provided the basis for our framework.

<p align="right">
 <a href="#top"><b>🔝 Return </b></a>
</p>

---

