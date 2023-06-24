<div align="center">
 
![logo](https://github.com/mnn59/BSc/blob/main/assets/app.png)  

<h1 align="center"><strong>🩺 Medical Image Segmentation System (MISS) <h6 align="center">My BSc project</h6></strong></h1>

![PyTorch - Version](https://img.shields.io/badge/PYTORCH-2.0+-red?style=for-the-badge&logo=pytorch)
![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
[![GitHub Issues](https://img.shields.io/github/issues/souvikmajumder26/Land-Cover-Semantic-Segmentation-PyTorch.svg?style=for-the-badge)](https://github.com/mnn59/BSc/issues)
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg?style=for-the-badge)

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
- [License](#license)
- [Acknowledgements](#acknowledgements)

----

## 📌 Overview <a name="overview"></a>
An end-to-end Computer Vision project focused on the topic of <a href="https://en.wikipedia.org/wiki/Image_segmentation" target="_blank">Image Segmentation</a> (specifically Semantic Segmentation). Although this project has primarily been built with the <a href="https://landcover.ai.linuxpolska.com/" target="_blank">LandCover.ai dataset</a>, the project template can be applied to train a model on any semantic segmentation dataset and extract inference outputs from the model in a <b>promptable</b> fashion. Though this is not even close to actual promptable AI, the term is being used here because of a specific functionality that has been integrated here.

The model can be trained on any or all the classes present in the semantic segmentation dataset with the ability to customize the model architecture, optimizer, learning rate, and a lot more parameters directly from the config file, giving it an <b>exciting AutoML</b> aspect. Thereafter while testing, the user can pass the prompt (in the form of the config variable '<b>test_classes</b>') of the selected classes that the user wants to be present in the masks predicted by the trained model.

For example, suppose the model has been trained on all the 30 classes of the <a href="https://www.cityscapes-dataset.com/" target="_blank">CityScapes dataset</a> and while inferencing, the user only wants the class <b>'parking'</b> to be present in the predicted mask due to a specific use-case application. Therefore, the user can provide the prompt as '<b>test_classes = ['parking']</b>' in the config file and get the desired output.

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

