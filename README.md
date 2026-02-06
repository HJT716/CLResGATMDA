## CLResGATMDA

Code and Datasets for "An Improved Graph Attention Network with Residual Connections for Accurate Disease-Microorganism Association Prediction"

### Developers

Jieteng Hou (hjt568124@gmail.com), Henan Agricultural University and Hefei Wang (mxiao5028@gmail.com) College of Forestry, Henan Agricultural University
### Datasets

HMDAD/adj.txt:Indices corresponding to the known microbe-disease associations within the dataset of HMDAD that records validated microbe-disease correlations
HMDAD/interaction.mat:A dataset containing known microbe-disease associations between microbes and diseases across different taxa
HMDAD/microbe_features.txt:microbe-microbe similarity
HMDAD/disease_features.txt:disease-disease similarity
Disbiome/adj.txt:Indices corresponding to the known microbe-disease associations within the dataset of HMDAD that records validated microbe-disease correlations
Disbiome/interaction.mat:A dataset containing known microbe-disease associations between microbes and diseases across different taxa
Disbiome/microbe_features.txt:microbe-microbe similarity
Disbiome/disease_features.txt:disease-disease similarity




### Environment Requirement
The code has been tested running under Python 3.10 The required packages are as follows:

*numpy: 2.2.6
*pandas: 2.3.2
*torch: 2.10.0.dev20250910+cu128
*torch_geometric: 2.6.1
*scikit-learn: 1.7.1
*matplotlib: 3.10.6
*scipy: 1.15.3
*openpyxl: 3.2.0b1

### Usage

Users can use their own data to train prediction models. If you want to train CLResGATMDA with your own dataset, you need to go through the following two processes: **1 Prepare the dataset** and **2 Microbe-disease interaction prediction**. In the meantime, we have provided the data that has been processed and you can implement CLResGATMDA directly using procedure: **2 Microbe-disease interaction prediction**.
 
### 1 Prepare the dataset and Predicting

```
git clone https://github.com/HJT716/CLResGATMDA
cd code/
python ResCLGATEns.py   
python inits.py   
```

1. Download all microbe-disease association information from the NCBI Batch Entrez tool, the HMDAD database (http://www.cuilab.cn/hmdad), and the Disbiome database (https://disbiome.ugent.be/home), name the file interaction.mat, and save them into the HMDAD and Disbiome folders respectively.
2. Run the `ResCLGATEns.py` script, which will perform a series of operations including calculating microbe-disease similarity, constructing a heterogeneous network, and predicting microbe-disease associations.


### Contact

Please feel free to contact us if you need raw data or any other help.

