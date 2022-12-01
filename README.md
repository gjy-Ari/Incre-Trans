## Rapid Identification of Damaged Buildings using Incremental Learning with Transferred Data from Historical Natural Disaster Cases
  
<img src="https://user-images.githubusercontent.com/67847241/196411214-f2b5a07b-5e44-44ca-9e15-4e1d648dfa33.jpg" width="830" height="570"><br/>
  
**The application process of the Incre-Trans framework.** End-to-end gradient boosting networks with assemble-decision strategy (EGB-A) as an incremental learning framework for mergency response, where the historical natural disaster data are transferred into the same style of post-disaster images by using cycle-consistent generative adversarial networks (CycleGAN). 

## Requirements
TensorFlow == 1.12.0  
keras == 2.2.4  
python == 3.6.7  
skimage  
osgeo  
matplotlib  
cv2  

## Usage
```Execute main.py to train base learners with adaptive learning rates.```

```When the trained model is applied, execute predict_probability.py to predict remote sensing images.```

```Execute assemble.py to fuse multi-stage predictions to generate recognition results.```

You can refer to [GeoBoost](https://doi.org/10.3390/rs12111794) for more information about EGB.

## Style transfer
After the disaster, [CycleGAN](https://doi.org/10.1109/ICCV.2017.244) is employed to transfer the historical data to the style of current disaster images.

For more information about it, please refer to [https://junyanz.github.io/CycleGAN/](https://junyanz.github.io/CycleGAN/).

## Data
Several datasets are used in this work.


You can find the data of the Haiti and Nepal cases we used in this shared folder: 
[https://drive.google.com/drive/folders/1Um9poJPwbrVRE1ge01prrKOONot0XfM6?usp=sharing](https://drive.google.com/drive/folders/1Um9poJPwbrVRE1ge01prrKOONot0XfM6?usp=sharing)  


Link to open access xBD dataset: [https://xview2.org/dataset](https://xview2.org/dataset)

## Citation
If you use the method or data of this work, please cite the following paper:  
[https://doi.org/10.1016/j.isprsjprs.2022.11.010](https://doi.org/10.1016/j.isprsjprs.2022.11.010)

##
If you have any questions, please feel free to contact: 202021051203@mail.bnu.edu.cn
