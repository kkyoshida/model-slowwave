# model-slowwave

## Overview
These codes are accompanying the manuscript:

"Prefrontal synaptic regulation of homeostatic sleep pressure revealed through synaptic chemogenetics"

Takeshi Sawada, Yusuke Iino, Kensuke Yoshida, Hitoshi Okazaki, Shinnosuke Nomura, Chika Shimizu, Tomoki Arima, Motoki Juichi, Siqi Zhou, Nobuhiro Kurabayashi, Takeshi Sakurai, Sho Yagishita, Masashi Yanagisawa, Taro Toyoizumi, Haruo Kasai, Shoi Shi, 

Science, 2024. https://doi.org/10.1126/science.adl3043


## Requirements
Simulations for the paper above were conducted in the following setup:
- Mac Pro (2019)
- CPU: 3.2 GHz 16 core Intel Xeon W
- RAM: 96 GB
- python 3.9.16, numpy 1.24.3, matplotlib 3.7.1

## Usage
For basic usage, refer to the Jupyter notebook:
- 'slowwave_demo.ipynb'

The descriptions of the model are in 'models.py'.


To generate the figures in the paper, execute the following scripts:
- 'runcodetwopopu.py'
- 'runcodesinglepopu1.py'
- 'runcodesinglepopu2.py'

## License
This project is licensed under the MIT License (see LICENSE.txt for details).
