# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 09:11:57 2021

@author: matts
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
import seaborn as sns
from IPython.display import display

#%%

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

def get_train_file_path(image_id):
    return "./train/{}.jpg".format(image_id)

def get_test_file_path(image_id):
    return "./test/{}.jpg".format(image_id)

train['file_path'] = train['Id'].apply(get_train_file_path)
test['file_path'] = test['Id'].apply(get_test_file_path)

display(train.head())
display(test.head())

#%%

# ====================================================
# Directory settings
# ====================================================
import os

OUTPUT_DIR = './cnn_ML/lgb_train'
MODEL_DIR = './cnn_ML/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
#%%

# ====================================================
# CFG
# ====================================================
class CFG:
    num_workers=0
    size=512
    batch_size=16
    model_name='tf_efficientnet_b3_ns'
    seed=42
    target_size=1
    target_col='Pawpularity'
    n_fold=5
    
#%%

# ====================================================
# Library
# ====================================================

import os
import gc
import sys
import math
import time
import pickle
import random
import shutil
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict, Counter

import scipy as sp
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

from tqdm.auto import tqdm
from functools import partial

import cv2
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, SGD
import torchvision.models as models
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau

import albumentations as A
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform

sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import timm

import lightgbm as lgb

from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

from glob import glob
from math import sqrt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%

# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    score = mean_squared_error(y_true, y_pred, squared=False)
    return score


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)

#%%

def get_train_file_path(image_id):
    return "./train/{}.jpg".format(image_id)

train = pd.read_csv('./train.csv')
train = train[train['Pawpularity'] != 100].reset_index().drop(columns=['index'])
train['file_path'] = train['Id'].apply(get_train_file_path)

num_bins = int(np.floor(1 + np.log2(len(train))))
train["bins"] = pd.cut(train[CFG.target_col], bins=num_bins, labels=False)

Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for n, (train_index, val_index) in enumerate(Fold.split(train, train["bins"])):
    train.loc[val_index, 'fold'] = int(n)
train['fold'] = train['fold'].astype(int)

display(train.groupby(['fold', "bins"]).size())

#%%

# ====================================================
# Dataset
# ====================================================

class TestDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        return image
    
#%%

# ====================================================
# Transforms
# ====================================================

def get_transforms(*, data):
    
    if data == 'train':
        return A.Compose([
            A.RandomResizedCrop(CFG.size, CFG.size, scale=(0.85, 1.0)),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

    elif data == 'valid':
        return A.Compose([
            A.Resize(CFG.size, CFG.size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])

#%%

# ====================================================
# MODEL
# ====================================================

class CustomModel(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.model = timm.create_model(self.cfg.model_name, pretrained=pretrained)
        self.n_features = self.model.classifier.in_features
        self.model.classifier = nn.Identity()
        self.fc = nn.Linear(self.n_features, self.cfg.target_size)

    def feature(self, image):
        feature = self.model(image)
        return feature
        
    def forward(self, image):
        feature = self.feature(image)
        output = self.fc(feature)
        return output
    
#%%

# ====================================================
# Helper functions
# ====================================================

def get_features(test_loader, model, device):
    model.eval()
    features = []
    tk0 = tqdm(enumerate(test_loader), total=len(test_loader))
    for step, (images) in tk0:
        images = images.to(device)
        batch_size = images.size(0)
        with torch.no_grad():
            feature = model.feature(images)
        features.append(feature.to('cpu').numpy())
    features = np.concatenate(features)
    return features

#%%

IMG_FEATURES = []
test_dataset = TestDataset(train, transform=get_transforms(data='valid'))
test_loader = DataLoader(test_dataset, 
                         batch_size=CFG.batch_size * 2, 
                         shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
for fold in range(CFG.n_fold):
    model = CustomModel(CFG, pretrained=False)
    state = torch.load(MODEL_DIR+f'{CFG.model_name}_fold{fold}_best.pth', 
                       map_location=torch.device('cpu'))['model']
    model.load_state_dict(state)
    model.to(device)
    features = get_features(test_loader, model, device)
    IMG_FEATURES.append(features)
    del state; gc.collect()
    torch.cuda.empty_cache()

#%%

# ====================================================
# Model
# ====================================================

def run_single_lightgbm(param, train, features, target, fold=0, categorical=[]):
    
    train[[f"img_{i}" for i in np.arange(1536)]] = IMG_FEATURES[fold]
    
    trn_idx = train[train.fold != fold].index
    val_idx = train[train.fold == fold].index

    if categorical == []:
        trn_data = lgb.Dataset(train.iloc[trn_idx][features].values, label=target.iloc[trn_idx].values)
        val_data = lgb.Dataset(train.iloc[val_idx][features].values, label=target.iloc[val_idx].values)
    else:
        trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx].values, categorical_feature=categorical)
        val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx].values, categorical_feature=categorical)
        
    num_round = 10000
    clf = lgb.train(param, 
                    trn_data,
                    num_round,
                    valid_sets=[trn_data, val_data],
                    verbose_eval=10,
                    early_stopping_rounds=10)
    with open(OUTPUT_DIR+f'lightgbm_fold{fold}.pkl', 'wb') as fout:
        pickle.dump(clf, fout)
    
    oof = np.zeros(len(train))
    oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    score = get_score(target.iloc[val_idx].values, oof[val_idx])
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["Feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold

    return oof, fold_importance_df, val_idx


def run_kfold_lightgbm(param, train, features, target, n_fold=5, categorical=[]):
    
    oof = np.zeros(len(train))
    feature_importance_df = pd.DataFrame()
    val_idxes = []
    
    for fold in range(n_fold):
        _oof, fold_importance_df, val_idx = run_single_lightgbm(param, 
                                                                train, features, target, 
                                                                fold=fold, categorical=categorical)
        oof += _oof
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        val_idxes.append(val_idx)
    
    val_idxes = np.concatenate(val_idxes)
    score = get_score(target.iloc[val_idxes].values, oof[val_idxes])
    
    return oof, feature_importance_df, val_idxes


def show_feature_importance(feature_importance_df):
    cols = (feature_importance_df[["Feature", "importance"]]
                .groupby("Feature").mean().sort_values(by="importance", ascending=False)[:50].index)
    best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]
    plt.figure(figsize=(8, 16))
    sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))
    plt.title('Features importance (averaged/folds)')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR+'feature_importance_df_lightgbm.png')
    
#%%

print(len(IMG_FEATURES[1]))

#%%
target = train['Pawpularity']
features = ['Subject Focus', 'Eyes', 'Face', 'Near', 'Action', 'Accessory',
            'Group', 'Collage', 'Human', 'Occlusion', 'Info', 'Blur'] + [f"img_{i}" for i in np.arange(1536)]

lgb_param = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'seed': 42,
    'max_depth': -10,
    'min_data_in_leaf': 10,
    'verbosity': -1,
}

oof, feature_importance_df, _ = run_kfold_lightgbm(lgb_param, 
                                                   train, features, target, 
                                                   n_fold=5, categorical=[])

show_feature_importance(feature_importance_df)
feature_importance_df.to_csv(OUTPUT_DIR+f'feature_importance_df.csv', index=False)

#%%

train['pred'] = oof
score = get_score(train['Pawpularity'].values, train['pred'].values)
train[['Id', 'Pawpularity', 'pred']].to_pickle(OUTPUT_DIR+'oof.pkl')

#%%

# setting dummy array for apples-to-apples comparison using same split

train_path = './train_resized'
train_jpg = glob(train_path + "/*.jpg")
train_images = [cv2.imread(file) for file in train_jpg]

X = np.array(train_images)
X = X / 255
X = train
Y = target
Y2 = oof

#%%

train_data, test_data, train_labels, test_labels = train_test_split(X,Y, test_size = .2, random_state = 7)
train_data, test_data, pred_train_labels_oof, pred_test_labels_oof = train_test_split(X,Y2, test_size = .2, random_state = 7)

#%%

print("For the EfficientNet b3 model augmented using Light GBM, the RMSE is ", 
      round(sqrt(mean_squared_error(test_labels, pred_test_labels_oof)),4), ".", sep="")


#%%



test = pd.read_csv('./test.csv')

def get_test_file_path(image_id):
    return "./test/{}.jpg".format(image_id)

test['file_path'] = test['Id'].apply(get_test_file_path)

display(test.head())

#%%

LGB_MODEL_DIR = './cnn_ML/'

def inference_single_lightgbm(test, features, model_path, fold):
    test[[f"img_{i}" for i in np.arange(1536)]] = IMG_FEATURES[fold]
    with open(model_path, 'rb') as fin:
        clf = pickle.load(fin)
    prediction = clf.predict(test[features], num_iteration=clf.best_iteration)
    return prediction

model_paths = [(fold, LGB_MODEL_DIR+f'lightgbm_fold{fold}.pkl') for fold in range(5)]
predictions = [inference_single_lightgbm(test, features, model_path, fold) for fold, model_path in model_paths]
predictions = np.mean(predictions, 0)

test['Pawpularity'] = predictions
test[['Id', 'Pawpularity']].to_csv('submission.csv', index=False)
display(test[['Id', 'Pawpularity']].head())

#%%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow.keras import datasets, layers, models
from matplotlib import image
from glob import glob
import cv2

import time
from matplotlib.ticker import MultipleLocator
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#%%

path = './'

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_path = './train'
train_resized_path = './train_resized'
train_bw_path = './train_resized_bw'
test_path = './test'

train_jpg = glob(train_path + "/*.jpg")
train_bw_jpg = glob(train_bw_path + "/*.jpg")
test_jpg = glob(test_path + "/*.jpg")

len(train_jpg)


train_images = [cv2.imread(file) for file in train_jpg]
train_bw_images_1d = [cv2.imread(file, 0).flatten(order = 'C') for file in train_bw_jpg] # 0 for grayscale, C for row-style flattening
test_images = [cv2.imread(file) for file in test_jpg]

#%%

X = np.array(train_bw_images_1d)
X = X / 255
Y = np.array(train_df['Pawpularity'])

#%%

train_bw_images_1d, test_data, train_labels, test_labels = train_test_split(X,Y, test_size = .2, random_state = 42)
print(train_bw_images_1d.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

#%%

def KNN(k_values):
    for val in k_values:
        KNN_model = KNeighborsClassifier(n_neighbors=val)
        KNN_model.fit(train_bw_images_1d, train_labels)
        test_predict = KNN_model.predict(test_data)
        print("k = ", val, "accuracy is: ", round(KNN_model.score(test_data, test_labels), 3), "\n")
        
k_values = [1, 3, 5, 7, 9]
KNN(k_values)

#%%

KNN_model = KNeighborsClassifier(n_neighbors=3)
KNN_model.fit(train_bw_images_1d, train_labels)
test_predict = KNN_model.predict(test_data)

#%%

test_SSE = (test_predict - test_labels) ** 2
sum_test_SSE = np.mean(test_SSE)

#%%

oof_df = pd.DataFrame(oof, columns=['prediction'])

oof_df.to_csv(OUTPUT_DIR+'oof_EN_b3_lgb.csv', index=False)    
