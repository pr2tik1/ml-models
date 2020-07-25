# AWS SageMaker Projects

Amazon SageMaker is a fully managed service that provides every developer and data scientist with the ability to build, train, and deploy machine learning (ML) models quickly. SageMaker removes the heavy lifting from each step of the machine learning process to make it easier to develop high quality models. 

The primary steps are:
- Download data
- Process the data
- Upload the processed data to S3 Bucket
- Data Modelling
- Training 
- Deploying the model
- Predicting
- Deleting the endpoints after use 

# Description

The notebook requires following :

* AWS SageMaker
* AWS S3
* AWS Notebook instances - ml.t2.medium, ml.p2.xlarge, ml.m4.xlarge
* Jupyter Lab/Notebook
* Python 3+

# Installation
Run the files by either command or Jupyter lab with AWS required tokens. Libraries required: 

```
boto3
sagemaker
seaborn
matplotlib
numpy
pandas
```
# Directory
```bash
|-- aws-sagemaker
|   |-- Pop_Segmentation.ipynb
|   |-- README.md
|   |-- energy-consumption.ipynb
|   |-- fraud-detection.ipynb
|   `-- txt_preprocessing.py
```
# Caution
Delete the endpoint after the process of prediction/analysis is done.

# Acknowledgements
These are my project submissions at Udacity's course Machine Learning Engineer Nanodegree.
