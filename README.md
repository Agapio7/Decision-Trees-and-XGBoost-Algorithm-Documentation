# XGBOOST
## Overview

*  The extended form of XGBoost is Extreme Gradient Boosting.
* XGBoost uses supervised learning algorithm  to accurately predict a target variable by combining an ensemble of estimates from a set of simpler and weaker models.
* XGBoost is an optimized distributed gradient boosting library designed to be highly efficient, flexible and portable.
* It implements Machine Learning algorithms under the Gradient Boosting framework.
* XGBoost is an extension to gradient boosted decision trees (GBM) and specially purposed to improve speed and performance.
* The XGBoost algorithm performs well in machine learning competitions because of its robust handling of a variety of data types, relationships, distributions, and the variety of hyperparameters that can be fine-tuned.
*  XGBoost is used for regression, classification (binary and multiclass), and ranking problems.

## Origin of XGBOOST
* The term “Gradient Boosting” was coined by Jerome H. Friedman from his paper Greedy Function Approximation: A Gradient Boosting Machine.

## Brief illustration on how gradient tree boosting works

![image](https://user-images.githubusercontent.com/91752852/142379264-f5141ccb-0eb3-42e8-aae3-822c811e856d.png)
Source: 

## XGBoost Features

### Regularized Learning: 

* It penalizes more complex models through both LASSO (L1) and Ridge (L2) regularization to prevent overfitting.
* Smooth the final learnt weights to avoid over-fitting. 
* The regularized objective is to select a model employing simple and predictive functions.

### Gradient Tree Boosting: 

* The tree ensemble model cannot be optimized using traditional optimization methods in Euclidean space. 
* Instead, the model is trained in an additive manner.

### Shrinkage and Column Subsampling: 
* Two additional techniques Shrinkage and Column Subsampling are used after regularization objective to further prevent overfitting.
#### Shrinkage Subsampling
* Shrinkage scales newly added weights by a factor η after each step of tree boosting. 
* Similar to a learning rate in stochastic optimization, shrinkage reduces the influence of each tree and leaves space for future trees to improve the model. 
#### Column Subsampling
* Column (feature) subsampling uses Random Forest. 
* Column sub-sampling prevents over-fitting even more so than the traditional row sub-sampling. 
* Column sub-samples also boosts up computations of the parallel algorithm.

### SPLITTING ALGORITHMS
#### Exact Greedy Algorithm: 
* The main problem in tree learning is to find the best split. 
* This algorithm enumerates over all the possible splits on all the features. 
* It is computationally demanding to enumerate all the possible splits for continuous features.
#### Approximate Algorithm:
* The exact greedy algorithm enumerates over all possible splitting points greedily but it is not possible when the data does not fit entirely into memory. 
* It proposes candidate splitting points according to percentiles of feature distribution. 
* The algorithm then plans the continuous features into buckets split by these candidate points, aggregates the statistics and search for the best solution among proposals based on the aggregated statistics.
#### Weighted Quantile Sketch: 
* One important theme in the approximate algorithm is to propose candidate split points. 
* XGBoost has a distributed weighted quantile sketch algorithm to effectively handle weighted data.

#### Sparsity-aware Split Finding: 
* there are many spare such as missing values in the data, frequent zero entries in the statistics and artifacts of feature engineering such as one-hot encoding.
* XGBoost obviously admits sparse features for inputs by automatically ‘learning’ best missing value depending on training loss and handles all sparsity patterns in a unified way.


## System Optimization

The library provides a system for use in a range of computing environments.

**Parallelization** of tree construction using all of your CPU cores during training.

**Distributed Computing** for training very large models using a cluster of machines.

**Out-of-Core Computing** optimizes available disk space while handling big data-frames that do not fit into memory.

**Cache Optimization** of data structures and algorithm to make best use of hardware to store gradient statistics.

**Block structure for Parallel Learning**: XGBoost can make use of multiple cores on the CPU for faster computing. This is due to block architecture in its system design. Data is sorted and stored in in-memory units called blocks.

## Goal of XGBOOST

### Execution Speed: 
* XGBoost is almost always faster than the other benchmarked implementations from R, Python Spark and H2O 
* It is really 10 times faster when compared to the other algorithms.

    

Fig: Benchmark Performance of XGBoost (Source: Benchmarking Random Forest Implementations)

### High Model Performance: 
* XGBoost dominates structured or tabular datasets on classification and regression predictive modelling problems.

## System Implementation
XGBoost is an open source package which is portable and reusable. 
* XGBOOST has various applications to solve problems such as regression, classification, ranking, and user-defined prediction problems.
* Feasible on running smoothly on Windows, Linux, and OS X.
* Supports all popular programming languages such as C++, Python, R, Java, Scala, and Julia and integrates naturally with language native data science pipelines such as scikitlearn.
* Cloud Integration such as AWS, Azure, Tianchi and Yarn clusters and works well with Flink, Spark, Hadoop, MPI Sun Grid engine and other ecosystems.

## XGBoost Parameters

 Three types of parameters in XGBoost : general parameters, booster parameters and task parameters as mentioned below.

**General parameters** is used for boosting, commonly tree or linear model

**Booster parameters** depend on users preference

**Learning task parameters** decide on the learning scenario. For example, regression tasks may use different parameters with ranking tasks.

**Command line parameters** relate to behavior of CLI version of XGBoost.



## How to Use Sage Maker XGBoost?

 We can use XGBoost in SageMaker in built-in algorithm or framework.
.








# Amazon Sage Maker on  XGBoost 

 We can use XGBoost in SageMaker by using built-in algorithm or framework.


## Use XGBoost as a framework

In the following example.Python SDK provides the XGBoost API as a framework while it provides other framework APIs, such as TensorFlow, MXNet, and PyTorch.


### SageMaker Python SDK 1
```bash
import boto3
import sagemaker
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.session import s3_input, Session

# initialize hyperparameters
hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "verbosity":"1",
        "objective":"reg:linear",
        "num_round":"50"}

# set an output path where the trained model will be saved
bucket = sagemaker.Session().default_bucket()
prefix = 'DEMO-xgboost-as-a-framework'
output_path = 's3://{}/{}/{}/output'.format(bucket, prefix, 'abalone-xgb-framework')

# construct a SageMaker XGBoost estimator
# specify the entry_point to your xgboost training script
estimator = XGBoost(entry_point = "your_xgboost_abalone_script.py", 
                    framework_version='1.2-2',
                    hyperparameters=hyperparameters,
                    role=sagemaker.get_execution_role(),
                    instance_count=1,
                    instance_type='ml.m5.2xlarge',
                    output_path=output_path)

# define the data type and paths to the training and validation datasets
content_type = "libsvm"
train_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'train'), content_type=content_type)
validation_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'validation'), content_type=content_type)

# execute the XGBoost training job
estimator.fit({'train': train_input, 'validation': validation_input})
```

### SageMaker Python SDK 2


```bash
import boto3
import sagemaker
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput

# initialize hyperparameters
hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "verbosity":"1",
        "objective":"reg:linear",
        "num_round":"50"}

# set an output path where the trained model will be saved
bucket = sagemaker.Session().default_bucket()
prefix = 'DEMO-xgboost-as-a-framework'
output_path = 's3://{}/{}/{}/output'.format(bucket, prefix, 'abalone-xgb-framework')

# construct a SageMaker XGBoost estimator
# specify the entry_point to your xgboost training script
estimator = XGBoost(entry_point = "your_xgboost_abalone_script.py", 
                    framework_version='1.2-2',
                    hyperparameters=hyperparameters,
                    role=sagemaker.get_execution_role(),
                    instance_count=1,
                    instance_type='ml.m5.2xlarge',
                    output_path=output_path)

# define the data type and paths to the training and validation datasets
content_type = "libsvm"
train_input = TrainingInput("s3://{}/{}/{}/".format(bucket, prefix, 'train'), content_type=content_type)
validation_input = TrainingInput("s3://{}/{}/{}/".format(bucket, prefix, 'validation'), content_type=content_type)

# execute the XGBoost training job
estimator.fit({'train': train_input, 'validation': validation_input})
```


## Use XGBoost as a built-in algorithm

Amazon Sage Maker uses the XGBoost built-in algorithm to build an XGBoost training container as shown in the following code example.

### SageMaker Python SDK 1

```bash
import sagemaker
import boto3
from sagemaker.amazon.amazon_estimator import get_image_uri 
from sagemaker.session import s3_input, Session
            
# initialize hyperparameters
hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"reg:squarederror",
        "num_round":"50"}

# set an output path where the trained model will be saved
bucket = sagemaker.Session().default_bucket()
prefix = 'DEMO-xgboost-as-a-built-in-algo'
output_path = 's3://{}/{}/{}/output'.format(bucket, prefix, 'abalone-xgb-built-in-algo')

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.
xgboost_container = get_image_uri(boto3.Session().region_name,
                          'xgboost', 
                          repo_version='1.2-2')

# construct a SageMaker estimator that calls the xgboost-container
estimator = sagemaker.estimator.Estimator(image_name=xgboost_container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          instance_count=1, 
                                          instance_type='ml.m5.2xlarge', 
                                          train_volume_size=5, # 5 GB 
                                          output_path=output_path)

# define the data type and paths to the training and validation datasets
content_type = "libsvm"
train_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'train'), content_type=content_type)
validation_input = s3_input("s3://{}/{}/{}/".format(bucket, prefix, 'validation'), content_type=content_type)

# execute the XGBoost training job
estimator.fit({'train': train_input, 'validation': validation_input})

```
### SageMaker Python SDK 2

```bash
  import sagemaker
import boto3
from sagemaker import image_uris
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput

# initialize hyperparameters
hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"reg:squarederror",
        "num_round":"50"}

# set an output path where the trained model will be saved
bucket = sagemaker.Session().default_bucket()
prefix = 'DEMO-xgboost-as-a-built-in-algo'
output_path = 's3://{}/{}/{}/output'.format(bucket, prefix, 'abalone-xgb-built-in-algo')

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
# specify the repo_version depending on your preference.
xgboost_container = sagemaker.image_uris.retrieve("xgboost", region, "1.2-2")

# construct a SageMaker estimator that calls the xgboost-container
estimator = sagemaker.estimator.Estimator(image_uri=xgboost_container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          instance_count=1, 
                                          instance_type='ml.m5.2xlarge', 
                                          volume_size=5, # 5 GB 
                                          output_path=output_path)

# define the data type and paths to the training and validation datasets
content_type = "libsvm"
train_input = TrainingInput("s3://{}/{}/{}/".format(bucket, prefix, 'train'), content_type=content_type)
validation_input = TrainingInput("s3://{}/{}/{}/".format(bucket, prefix, 'validation'), content_type=content_type)

# execute the XGBoost training job
estimator.fit({'train': train_input, 'validation': validation_input})
```


# Regression with Amazon SageMaker XGBoost algorithm

## Introduction
This project presents the use of Amazon SageMaker XGBoost to train and host a regression model.

 Abalone data has been used  originally from the UCI data repository.

 Here,the nominal feature(Male/Female/Infant) has been converted into a real valued feature as required by XGBoost.Age of abalone is to be predicted from eight physical measurements.

 ## Set Up

 This project was tested in Amazon SageMaker Studio on a ml.t3.medium instance with Python 3  kernel.

 First sepcify the The S3 bucket and prefix that we want to use for training and model data. It should be within the same region as the Notebook Instance, training, and hosting.
 
The IAM role arn used to give training and hosting access to our data.

```bash
%%time

import os
import boto3
import re
import sagemaker

# Get a SageMaker-compatible role used by this Notebook Instance.
role = sagemaker.get_execution_role()
region = boto3.Session().region_name

### update below values appropriately ###
bucket = sagemaker.Session().default_bucket()
prefix = "sagemaker/DEMO-xgboost-dist-script"
####

print(region)
```

## Fetch Dataset
Split the data into train/test/validation datasets and upload files to S3 as shown below.

```bash
%%time

import io
import boto3
import random


def data_split(
    FILE_DATA,
    DATA_DIR,
    FILE_TRAIN_BASE,
    FILE_TRAIN_1,
    FILE_VALIDATION,
    FILE_TEST,
    PERCENT_TRAIN_0,
    PERCENT_TRAIN_1,
    PERCENT_VALIDATION,
    PERCENT_TEST,
):
    data = [l for l in open(FILE_DATA, "r")]
    train_file_0 = open(DATA_DIR + "/" + FILE_TRAIN_0, "w")
    train_file_1 = open(DATA_DIR + "/" + FILE_TRAIN_1, "w")
    valid_file = open(DATA_DIR + "/" + FILE_VALIDATION, "w")
    tests_file = open(DATA_DIR + "/" + FILE_TEST, "w")

    num_of_data = len(data)
    num_train_0 = int((PERCENT_TRAIN_0 / 100.0) * num_of_data)
    num_train_1 = int((PERCENT_TRAIN_1 / 100.0) * num_of_data)
    num_valid = int((PERCENT_VALIDATION / 100.0) * num_of_data)
    num_tests = int((PERCENT_TEST / 100.0) * num_of_data)

    data_fractions = [num_train_0, num_train_1, num_valid, num_tests]
    split_data = [[], [], [], []]

    rand_data_ind = 0

    for split_ind, fraction in enumerate(data_fractions):
        for i in range(fraction):
            rand_data_ind = random.randint(0, len(data) - 1)
            split_data[split_ind].append(data[rand_data_ind])
            data.pop(rand_data_ind)

    for l in split_data[0]:
        train_file_0.write(l)

    for l in split_data[1]:
        train_file_1.write(l)

    for l in split_data[2]:
        valid_file.write(l)

    for l in split_data[3]:
        tests_file.write(l)

    train_file_0.close()
    train_file_1.close()
    valid_file.close()
    tests_file.close()


def write_to_s3(fobj, bucket, key):
    return (
        boto3.Session(region_name=region)
        .resource("s3")
        .Bucket(bucket)
        .Object(key)
        .upload_fileobj(fobj)
    )


def upload_to_s3(bucket, channel, filename):
    fobj = open(filename, "rb")
    key = prefix + "/" + channel
    url = "s3://{}/{}/{}".format(bucket, key, filename)
    print("Writing to {}".format(url))
    write_to_s3(fobj, bucket, key)
```

## Data ingestion

* Read the dataset from the existing repository into memory, for preprocessing prior to training. 

* It can be done in situ by Amazon Athena, Apache Spark in Amazon EMR, Amazon Redshift, etc.We believd that the dataset is present in the right location. 

* After that, we transfer the data to S3 for use in training. For small datasets, such as this one, reading into memory isn’t onerous, though it would be for larger datasets.


```bash
%%time
s3 = boto3.client("s3")

# Load the dataset
FILE_DATA = "abalone"
s3.download_file(
    "sagemaker-sample-files", f"datasets/tabular/uci_abalone/abalone.libsvm", FILE_DATA
)

# split the downloaded data into train/test/validation files
FILE_TRAIN_0 = "abalone.train_0"
FILE_TRAIN_1 = "abalone.train_1"
FILE_VALIDATION = "abalone.validation"
FILE_TEST = "abalone.test"
PERCENT_TRAIN_0 = 35
PERCENT_TRAIN_1 = 35
PERCENT_VALIDATION = 15
PERCENT_TEST = 15

DATA_DIR = "data"

if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)

data_split(
    FILE_DATA,
    DATA_DIR,
    FILE_TRAIN_0,
    FILE_TRAIN_1,
    FILE_VALIDATION,
    FILE_TEST,
    PERCENT_TRAIN_0,
    PERCENT_TRAIN_1,
    PERCENT_VALIDATION,
    PERCENT_TEST,
)
```

```bash
# upload the files to the S3 bucket
upload_to_s3(bucket, "train/train_0.libsvm", DATA_DIR + "/" + FILE_TRAIN_0)
upload_to_s3(bucket, "train/train_1.libsvm", DATA_DIR + "/" + FILE_TRAIN_1)
upload_to_s3(bucket, "validation/validation.libsvm", DATA_DIR + "/" + FILE_VALIDATION)
upload_to_s3(bucket, "test/test.libsvm", DATA_DIR + "/" + FILE_TEST)
```

## Create a XGBoost script to train with

* SageMaker can now run an XGboost script using the XGBoost estimator.

* Two input channels, ‘train’ and ‘validation’, were used in the call to the XGBoost estimator’s fit() method.

* A typical training script loads data from the input channels, configures training with hyperparameters, trains a model, and saves a model to model_dir so that it can be hosted later. 

* Hyperparameters are passed to the script as arguments and can be retrieved with an argparse.ArgumentParser instance. 

* For instance, the script run in this notebook is provided as the accompanying file (abalone.py) and also shown below:



```bash
import argparse
import json
import logging
import os
import pandas as pd
import pickle as pkl

from sagemaker_containers import entry_point
from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container import distributed

import xgboost as xgb


def _xgb_train(params, dtrain, evals, num_boost_round, model_dir, is_master):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training,
                        or is running single node training job.
                        Note that rabit_run will include this argument.
    """
    booster = xgb.train(params=params,
                        dtrain=dtrain,
                        evals=evals,
                        num_boost_round=num_boost_round)

    if is_master:
        model_location = model_dir + '/xgboost-model'
        pkl.dump(booster, open(model_location, 'wb'))
        logging.info("Stored trained model at {}".format(model_location))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.
    parser.add_argument('--max_depth', type=int,)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--gamma', type=int)
    parser.add_argument('--min_child_weight', type=int)
    parser.add_argument('--subsample', type=float)
    parser.add_argument('--verbosity', type=int)
    parser.add_argument('--objective', type=str)
    parser.add_argument('--num_round', type=int)
    parser.add_argument('--tree_method', type=str, default="auto")
    parser.add_argument('--predictor', type=str, default="auto")

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output_data_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))
    parser.add_argument('--sm_hosts', type=str, default=os.environ.get('SM_HOSTS'))
    parser.add_argument('--sm_current_host', type=str, default=os.environ.get('SM_CURRENT_HOST'))

    args, _ = parser.parse_known_args()

    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host

    dtrain = get_dmatrix(args.train, 'libsvm')
    dval = get_dmatrix(args.validation, 'libsvm')
    watchlist = [(dtrain, 'train'), (dval, 'validation')] if dval is not None else [(dtrain, 'train')]

    train_hp = {
        'max_depth': args.max_depth,
        'eta': args.eta,
        'gamma': args.gamma,
        'min_child_weight': args.min_child_weight,
        'subsample': args.subsample,
        'verbosity': args.verbosity,
        'objective': args.objective,
        'tree_method': args.tree_method,
        'predictor': args.predictor,
    }

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=args.model_dir)

    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_train,
            args=xgb_train_args,
            include_in_training=(dtrain is not None),
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True
        )
    else:
        # If single node training, call training method directly.
        if dtrain:
            xgb_train_args['is_master'] = True
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError("Training channel must have data to train model.")


def model_fn(model_dir):
    """Deserialize and return fitted model.

    Note that this should have the same name as the serialized model in the _xgb_train method
    """
    model_file = 'xgboost-model'
    booster = pkl.load(open(os.path.join(model_dir, model_file), 'rb'))
    return booster
```

Because the container imports our training script, always put  training  code in a main guard [if __name__=='__main__':] so that the container does not inadvertently run our training code at the wrong point in execution.

## Traini the XGBoost model

* After setting training parameters, we start training, and poll for status until training is completed as shown below.
* We construct a sagemaker.xgboost.estimator to run our training script on SageMaker.
```bash
hyperparams = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.7",
    "objective": "reg:squarederror",
    "num_round": "50",
    "verbosity": "2",
}

instance_type = "ml.m5.2xlarge"
output_path = "s3://{}/{}/{}/output".format(bucket, prefix, "abalone-dist-xgb")
content_type = "libsvm"
```

```bash
  # Open Source distributed script mode
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput
from sagemaker.xgboost.estimator import XGBoost

boto_session = boto3.Session(region_name=region)
session = Session(boto_session=boto_session)
script_path = "abalone.py"

xgb_script_mode_estimator = XGBoost(
    entry_point=script_path,
    framework_version="1.3-1",  # Note: framework_version is mandatory
    hyperparameters=hyperparams,
    role=role,
    instance_count=2,
    instance_type=instance_type,
    output_path=output_path,
)

train_input = TrainingInput(
    "s3://{}/{}/{}/".format(bucket, prefix, "train"), content_type=content_type
)
validation_input = TrainingInput(
    "s3://{}/{}/{}/".format(bucket, prefix, "validation"), content_type=content_type
)
```

## Train XGBoost Estimator on abalone data

Training is as simple as calling fit on the Estimator. It start a SageMaker Training job that will download the data, invoke the entry point code in the provided script file, and save any model artifacts that the script creates.

```bash
xgb_script_mode_estimator.fit({"train": train_input, "validation": validation_input})
```

## Deploy the XGBoost model
Finally, we can use the estimator to create an Amazon SageMaker endpoint  a hosted and managed prediction service  to perform inference.

```bash
predictor = xgb_script_mode_estimator.deploy(
    initial_instance_count=1, instance_type="ml.m5.2xlarge"
)
```

```bash
test_file = DATA_DIR + "/" + FILE_TEST
with open(test_file, "r") as f:
    payload = f.read() 
```
```bash
runtime_client = boto3.client("runtime.sagemaker", region_name=region)
response = runtime_client.invoke_endpoint(
    EndpointName=predictor.endpoint_name, ContentType="text/libsvm", Body=payload
)
result = response["Body"].read().decode("ascii")
print("Predicted values are {}.".format(result)) 
```

## Delete the Endpoint

After completion, we can run the delete_endpoint line in the cell as given. It will remove the hosted endpoint and avoid paying from a stray instance being left on.

```bash
predictor.delete_endpoint()
```
