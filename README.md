# Sign Language Transformers improved using Keypoints features.

This repo contains some adaptions and improvements (attempts) I made regarding the code of the paper [Sign Language Transformers: Sign Language Transformers: Joint End-to-end Sign Language Recognition and Translation](https://www.cihancamgoz.com/pub/camgoz2020cvpr.pdf). 

This work was made in the context of a project of the course "**Object Recognition and Computer Vision**" by Jean Ponce, Ivan Laptev, Cordelia Schmid and Josef Sivic at **ENS - ULM**.

This code is based on [Joey NMT](https://github.com/joeynmt/joeynmt) but modified to realize joint continuous sign language recognition and translation. 
 
My contribution was to add pose estimations keypoints (2D or 3D) of the 2014T-phoenix dataset using the DOPE algorithm (https://github.com/naver/dope) https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123710375.pdf. 

# Improvements 

I described and implemented 3 different fusion types namely Early fusion, Late fusion and Mid-fusion in order to merge the information of the different channels (keypoints hand, body, face, and images) at different stages of the algorithm.

The three fusion architecture are shown bellow. Basically, they represent three ways of merging the additionnal pose estimation keypoints in the SLT model. 
A more detailed description of the architectures can be found on the report.

## Early Fusion :
![diag_early](https://user-images.githubusercontent.com/38350776/117323342-17ee9480-ae8f-11eb-8b3a-5f0f821fff78.png)

## Mid Fusion :
![diag_mid](https://user-images.githubusercontent.com/38350776/117323375-1e7d0c00-ae8f-11eb-84dc-89e3e1b0abb9.png)

## Late Fusion :
![diag_late](https://user-images.githubusercontent.com/38350776/117323406-263cb080-ae8f-11eb-822b-9fda89250b28.png)





A more detailed version of my work and the associated results can be found on the report report.pdf
 
 
## Requirements
* Download the feature files using the `data/download.sh` script.

* [Optional] Create a conda or python virtual environment.

* Install required packages using the `requirements.txt` file.

    `pip install -r requirements.txt`

## Usage

  `python -m signjoey train configs/sign.yaml` 

! Note that the default data directory is `./data`. If you download them to somewhere else, you need to update the `data_path` parameters in your config file.   
