%
% Two-Stream Translating LSTM Network for Mangroves Mapping Using Sentinel-2 Multivariate Time Series
%
%    This demo shows the TSTLN model for mangrove mapping.
%
%    main.py ....... A main script executing experiments upon mapping mangroves.
%    utils.py ....... A script implementing the precision calculation, claasificaiton map drawing, and etc.
%    predict.py ....... A script implementing test the precision of the model.
%    /Data ............... The folder including the code of data preprocessing.
%    /model ............... The folder containing the code of the TSTLN model.
%
%   --------------------------------------
%   Note: Required core python libraries
%   --------------------------------------
%   1. python 3.6
%   2. pytorch 1.8.0
%   3. torchvision 0.11.3

%   --------------------------------------
%   Cite:
%   --------------------------------------
%
%   [1] Z. Xue and S. Qian, "Two-Stream Translating LSTM Network for Mangroves Mapping Using Sentinel-2 Multivariate Time Series," in IEEE Transactions on Geoscience and Remote Sensing, vol. 61, pp. 1-16, 2023, Art no. 4401416, doi: 10.1109/TGRS.2023.3249179.
%   --------------------------------------
%   Copyright & Disclaimer
%   --------------------------------------
%
%   The programs contained in this package are granted free of charge for
%   research and education purposes only. 
%
%   Copyright (c) 2023 by Zhaohui Xue & Siyu Qian
%   zhaohui.xue@hhu.edu.cn & qsy108@163.com
%   --------------------------------------
%   For full package:
%   --------------------------------------
%   https://sites.google.com/site/zhaohuixuers/
