# ASFM-Net
🔥🔥🔥 This repository is the official implementation for ACM Multimedia 2021 paper '[**ASFM-Net: Asymmetrical Siamese Feature Matching Network for Point Completion**](https://dl.acm.org/doi/abs/10.1145/3474085.3475348)'. 

🔥🔥🔥 ASFM-Net achieves the **1st** place in the [leaderboard of Completion3D](https://completion3d.stanford.edu/results) from April, 2021. 

## 1. Getting Started Instructions.
+ **Clone this project**
```
git clone --recursive https://github.com/Yan-Xia/ASFM-Net.git
```
### **Step-1: Environment Setup** 
+ Install Docker Engine with NVIDIA GPU Support **[Toturial](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)**. We use the following commands
```
curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker

distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```
+ `cd envs` 
+ Run `sudo ./build.sh`, it may takes a while (around 10 minutes) for building the container.
+ Run `sudo ./launch.sh` then it will bring up an new interactive command line interface.
> + if your enounter problem below,
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].
```
then you need to 
```
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```
> + The defualt GPU is all gpus. If you want to run on different deivce. Consider using this command in `./launch.sh`, e.g., using device:1
>  
```
docker run -itd --rm --gpus device=1 -p 6006:6006 -v $PWD/../:/ASFM-Net --name asfm-net asfm-net /bin/bash
```
### **Step-2: Download the point cloud datasets.**
+ We use the same dataset as [PCN: Point Cloud Completion Network](https://arxiv.org/abs/1808.00671).
+ Download the ShapeNet dataset in the `shapent` folder on [Google Drive](https://drive.google.com/drive/folders/1Y_tx3lrA2ivvM-bGxRO-TvbVBE8HoNPu) provided by PCN and unzip the dataset to folder `data`.
### **Step-3: ShapeNet Completion.**
Download our trained model from [Googe Drive](https://drive.google.com/drive/folders/1r8x6jq1QCWJ9fvep604nMkexykqQGpT0?usp=sharing) for testing.  
Run `./shapenet_test` to test the accuracy of the model.   
Run `./shapent_completion.sh` to train and test ASFM-Net automatically. More specifically, the script will
+ compile the point cloud distance ops, grouping ops and sampling ops  
+ train ASFM-Net
+ test the accuracy of the saved model

 ## 2. Completion Results.
|            |                            Input                             |                            TopNet                            |                             PCN                              |                             RFA                              |                           ASFM-Net                           |                         Ground Truth                         |
| :--------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  Airplane  | <img src="https://media.giphy.com/media/BwuglRouFr0m7PFIvp/giphy.gif" width="100"> | <img src=https://media.giphy.com/media/O95xLKJ5mRaEH7c23M/giphy.gif width="100"> | <img src="https://media.giphy.com/media/okbXGXRU1KtJVcwqbT/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/lD560QbTlowMwu3HPp/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/KierHhUIvxOXV7M11F/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/LbVrAxfkLDZzQNkFdq/giphy.gif" width="100"> |
|  Cabinet   | <img src="https://media.giphy.com/media/bWFx9wSN0rHDcchmaR/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/hFs6bpa20wbYXiB8Mw/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/Q4AyvG6zmETUAdiOfv/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/a8j10mxICEDAqp5Zic/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/Zjs3hM5Vlv8rxR7uTu/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/Nfltt5WAT60K1jvDZq/giphy.gif" width="100"> |
|    Car     | <img src="https://media.giphy.com/media/jIR9NTY9juedzE6rJv/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/IPsgasO6BM4wTVNB3K/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/JPsg9pnP4hTPpZZ3pT/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/AYLDultVEaydAzt8Kx/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/opcWCY7lL73HaJNQh4/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/INGnIpNZzjqtk6Evjs/giphy.gif" width="100"> |
|   Chair    | <img src="https://media.giphy.com/media/83Kq8O4gzftrINisk3/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/vt83rCObKnwAXt44u4/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/EojmZeRpNF6sm2RqBA/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/inNTaK6L8BAiofrfvH/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/k6S9gyIBu9YqyomplF/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/D21kTahYDkzIGOEiS1/giphy.gif" width="100"> |
|    Lamp    | <img src="https://media.giphy.com/media/KMKucLVw5QcSbSvaoM/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/qqRUfkCpCITiL987ht/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/5O1dYxQCxur24rMgQK/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/KPMooJPWLXjINQJgsh/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/KdkSNUnlJtn6ortbuZ/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/F7HiwV3dXee2cAg3Jm/giphy.gif" width="100"> |
|    Sofa    | <img src="https://media.giphy.com/media/Fsd573x4JiPNh5JwTH/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/I9MB3lhX4o3nvFuGWN/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/xF0VbZLZ2fAk4m3BiM/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/8Qs8IkLoFy8XtLLA8K/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/GhxgOC7cvbHJrQlTpv/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/kHXDof9xXO1DqfBN4C/giphy.gif" width="100"> |
|   Table    | <img src="https://media.giphy.com/media/in7Rxi29QK0lvxDVtr/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/Pnl8X6c9RSoYiStbLn/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/ibGhm0uyDGSQApdrvB/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/o9zLjH8VJqP5xRvzED/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/Qr5iZHViIXruDhFxVZ/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/Cuxi11w0UYSGBcdPpJ/giphy.gif" width="100"> |
| Watercraft | <img src="https://media.giphy.com/media/vvi1LhMoeZD37D0EHD/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/8datRP61I2YfFX64ns/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/2vajWS3M5rY1aaaiqc/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/bzqqDBvufaVgZfb2Sg/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/ET5p8X4tuKE5PPlMHc/giphy.gif" width="100"> | <img src="https://media.giphy.com/media/LrQrGRd6tL18h4LiUz/giphy.gif" width="100"> |



## 3.Citation

------

If you find our work useful in your research, please consider citing:

```
@inproceedings{xia2021asfm ,
  title={ASFM-Net: Asymmetrical Siamese Feature Matching Network for Point Completion},
  author={Yaqi, Xia and Yan, Xia and Wei, Li and Rui, Song and Kailang, Cao and Uwe, Stilla},
  booktitle={Proceedings of the 29th ACM International Conference on Multimedia},
  year={2021}
}
```

