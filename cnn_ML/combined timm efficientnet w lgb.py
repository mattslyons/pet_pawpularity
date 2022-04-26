# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 12:56:14 2021

@author: matts
"""

# imports

import os
import sys
import math
import time
import random

import matplotlib.pyplot as plt
from IPython.display import display

import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import cv2
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.append('../input/pytorch-image-models/pytorch-image-models-master')
import timm
import lightgbm as lgb
import pickle

from torch.cuda.amp import autocast, GradScaler

import warnings
warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#%%

train = pd.read_csv('./train.csv')
# filtering out scores of 100 slightly improves results
train = train[train['Pawpularity'] != 100].reset_index()
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

# directory settings

OUTPUT_DIR = './lgb_train'
MODEL_DIR = './'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
    
#%%
    
# configuration for CNN

class CFG:
    apex=False
    debug=False
    print_freq=10
    num_workers=0
    size=512
    model_name='tf_efficientnet_b3_ns'
    scheduler='ReduceLROnPlateau'
    epochs=12
    factor=0.2
    patience=4
    eps=1e-6
    lr=1e-4
    min_lr=1e-6
    batch_size=16
    weight_decay=1e-6
    gradient_accumulation_steps=1
    max_grad_norm=1000
    seed=42
    target_size=1
    target_col='Pawpularity'
    n_fold=5
    trn_fold=[0, 1, 2, 3, 4]
    train=True
    grad_cam=False
    
if CFG.debug:
    CFG.epochs = 1
    train = train.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)

#%%

# utility functions

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

# splitting/binning

num_bins = int(np.floor(1 + np.log2(len(train))))
train["bins"] = pd.cut(train[CFG.target_col], bins=num_bins, labels=False)

Fold = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

for n, (train_index, val_index) in enumerate(Fold.split(train, train["bins"])):
    train.loc[val_index, 'fold'] = int(n)

train['fold'] = train['fold'].astype(int)
display(train.groupby(['fold', "bins"]).size())

#%%

# set up dataset classes to read in images

class TrainDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.file_names = df['file_path'].values
        self.labels = df[CFG.target_col].values
        self.transform = transform
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_names[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)['image']
        label = torch.tensor(self.labels[idx]).float()
        return image, label

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

# standard Pytorch image normalization

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

train_dataset = TrainDataset(train, transform=get_transforms(data='train'))

#%%

# set up TIMM model based on efficientnet b#

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

# define loss

class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, yhat, y):
        loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        
#%%

# Helper functions for averaging, time/estimation

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
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


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


# training function
def train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    if CFG.apex:
        scaler = GradScaler()
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        if CFG.apex:
            with autocast():
                y_preds = model(images)
                loss = criterion(y_preds.view(-1), labels)
        else:
            y_preds = model(images)
            loss = criterion(y_preds.view(-1), labels)
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        if CFG.apex:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            if CFG.apex:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=optimizer.param_groups[0]['lr']))
    return losses.avg


def valid_fn(valid_loader, model, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds = []
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds.view(-1), labels)
        losses.update(loss.item(), batch_size)
        # record accuracy
        preds.append(y_preds.to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    return losses.avg, predictions

def inference_single_lightgbm(test, features, model_path, fold):
    test[[f"img_{i}" for i in np.arange(1280)]] = IMG_FEATURES[fold]
    with open(model_path, 'rb') as fin:
        clf = pickle.load(fin)
    prediction = clf.predict(test[features], num_iteration=clf.best_iteration)
    return prediction

#%%

# train loop

def train_loop(folds, fold):

    # load up the current set
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_col].values

    train_dataset = TrainDataset(train_folds, transform=get_transforms(data='train'))
    valid_dataset = TrainDataset(valid_folds, transform=get_transforms(data='train'))

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size, 
                              shuffle=True, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, 
                              batch_size=CFG.batch_size * 2, 
                              shuffle=False, 
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
    
    # build/transfer model, set params
    model = CustomModel(CFG, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)

    # training
    criterion = RMSELoss()

    best_score = np.inf
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        
        start_time = time.time()
        
        # train
        avg_loss = train_fn(fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        
        scheduler.step(avg_val_loss)

        # scoring
        score = get_score(valid_labels, preds)
        
        elapsed = time.time() - start_time

        if score < best_score:
            best_score = score
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth')
    
    valid_folds['preds'] = torch.load(OUTPUT_DIR+f'{CFG.model_name}_fold{fold}_best.pth', 
                                      map_location=torch.device('cpu'))['preds']

    return valid_folds

#%%

# light GBM Training

def run_single_lightgbm(param, train, features, target, fold=0, categorical=[]):
    
    train[[f"img_{i}" for i in np.arange(1280)]] = IMG_FEATURES[fold]
    
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

#%%

# main

def main():

    def get_result(result_df):
        preds = result_df['preds'].values
        labels = result_df[CFG.target_col].values
        score = get_score(labels, preds)
    
    if CFG.train:
        # train 
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(train, fold)
                oof_df = pd.concat([oof_df, _oof_df])
                get_result(_oof_df)
        # CV result
        get_result(oof_df)
        # save result
        oof_df.to_csv(OUTPUT_DIR+'oof_df.csv', index=False)
        
#%%

if __name__ == '__main__':
    main()