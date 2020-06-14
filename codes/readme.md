# 基于决策树方法的PM2.5浓度预测

## 代码环境和软件版本
代码环境为python3.6, 代码格式为ipython, 安装jupyter notebook即可运行

## 各文件含义及作用
1. preprocess.ipynb 数据预处理文件，运行前需要在上级文件夹新建“data_mix_clean”和“data_mix_clean_pca”文件夹。运行时会读取data文件夹中的所有csv文件，并将预处理结果保存到“data_mix_clean”文件夹,“data_mix_clean_pca”文件夹和"data_mix_clean_all.csv"文件。

2. regression.ipynb 基于回归树的的模型的训练与测试文件，在运行preprocess.ipynb文件后再运行此文件，可以完成Bagging GBRT的训练及测试。若修改此文件的相应参数，则可以训练并测试其它基于回归树的模型。

3. classification.ipynb 基于分类树的的模型的训练与测试文件，在运行preprocess.ipynb文件后再运行此文件，可以完成分类决策树的训练及测试。若修改此文件的相应参数，则可以训练并测试其它基于分类树的模型。

4. BestModels  文件夹内是训练好的模型

4. test_classification.ipynb 在运行preprocess.ipynb文件后再运行此文件，可以完成分类决策树的测试，测试结果会在文件中以图片形成呈现。

5. test_regression.ipynb 在运行preprocess.ipynb文件后再运行此文件，可以完成Bagging GBRT的测试，测试结果会在文件中以图片形成呈现。