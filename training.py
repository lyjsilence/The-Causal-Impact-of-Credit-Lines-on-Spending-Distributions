import os
import time
import math
import torch
import numpy as np
import copy
import torch.nn as nn
import torch.optim as optim


def MAE(y_pred, target):
    """
    Mean average absolute error.

    Defined as ``1/N * |y_pred - target|``
    """
    mae = torch.mean(torch.abs(y_pred - target))
    return mae


def train_regression(args, model, dl_train, dl_test, dl_est_test_list, device, verbose=True):
    best_loss = float('inf')
    best_model = None

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=False, min_lr=1e-4)

    mae = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tic = time.time()

        loss_train_list = []

        for i, train_data in enumerate(dl_train):
            X_train, y_train = train_data[0], train_data[1]
            X_train = X_train.to(torch.float32).to(device)
            y_train = y_train.to(torch.float32).to(device)

            pred_train = model(X_train)
            if args.reg_met == 'rep_MLP':
                treat = X_train[:, -1] # treat = treat.to(torch.float32).to(device)
                loss_train = model.loss_cal(pred_train, y_train, treat, args.n_treat, args.loss, args.reweight_sample)

            else:
                loss_train = mae(pred_train, y_train)

            loss_train_list.append(loss_train.cpu().detach().numpy())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if verbose:
                print('\rEpoch [{}/{}], Batch [{}/{}], Loss = {:.4f}, time elapsed = {:.2f}, '
                      .format(epoch, args.epochs, i + 1, len(dl_train), np.sum(loss_train_list) / len(loss_train_list),
                              time.time() - tic), end='')

        with torch.no_grad():
            model.eval()

            for test_data in dl_test:
                X_test, y_test = test_data[0], test_data[1]
                X_test, y_test = X_test.to(torch.float32).to(device), y_test.to(torch.float32).to(device)
                pred_y_test = model(X_test)
                if args.reg_met == 'rep_MLP':
                    treat_test = X_test[:, -1] # treat_test = treat_test.to(torch.float32).to(device)
                    loss_test = model.loss_cal(pred_y_test, y_test, treat_test,
                                               args.n_treat, args.loss, args.reweight_sample)
                else:
                    loss_test = mae(pred_y_test, y_test)


            pred_y_test = pred_y_test.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()
            if verbose:
                print('Test: MAE={:.4f}'.format(loss_test.cpu().detach().numpy()))

            if loss_test.item() < best_loss:
                best_loss = loss_test.item()
                best_model = copy.deepcopy(model)

        scheduler.step(loss_test)

    # get the estimator for a=0 and a=1
    with torch.no_grad():
        model = copy.deepcopy(best_model)
        model.eval()
        pred_y_test_list = []

        for dl_est_test in dl_est_test_list:
            for test_data in dl_est_test:

                X_test, y_test = test_data[0], test_data[1]
                X_test = X_test.to(torch.float32).to(device)
                y_test = y_test.to(torch.float32).to(device)

                pred_y_test = model(X_test)
                pred_y_test_list.append(pred_y_test.detach().cpu().numpy())
    return pred_y_test_list

def train_classical(args, model, dl_train, dl_test, dl_est_test_list, device, verbose=True):
    best_loss = float('inf')
    best_model = None

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=False, min_lr=1e-4)

    mae = nn.L1Loss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tic = time.time()

        loss_train_list = []

        for i, train_data in enumerate(dl_train):
            X_train, y_train = train_data[0], train_data[1]
            X_train = X_train.to(torch.float32).to(device)
            y_train = y_train.to(torch.float32).to(device)

            pred_train = model(X_train)
            loss_train = mae(pred_train, y_train)

            loss_train_list.append(loss_train.cpu().detach().numpy())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if verbose:
                print('\rEpoch [{}/{}], Batch [{}/{}], Loss = {:.4f}, time elapsed = {:.2f}, '
                      .format(epoch, args.epochs, i + 1, len(dl_train), np.sum(loss_train_list) / len(loss_train_list),
                              time.time() - tic), end='')

        with torch.no_grad():
            model.eval()

            for test_data in dl_test:
                X_test, y_test = test_data[0], test_data[1]
                X_test, y_test = X_test.to(torch.float32).to(device), y_test.to(torch.float32).to(device)
                pred_y_test = model(X_test)
                loss_test = mae(pred_y_test, y_test)

            if verbose:
                print('Test: MAE={:.4f}'.format(loss_test.cpu().detach().numpy()))

            if loss_test.item() < best_loss:
                best_loss = loss_test.item()
                best_model = copy.deepcopy(model)

        scheduler.step(loss_test)

    # get the estimator for a=0 and a=1
    with torch.no_grad():
        model = copy.deepcopy(best_model)
        model.eval()
        pred_y_test_list = []

        for dl_est_test in dl_est_test_list:
            for test_data in dl_est_test:

                X_test, y_test = test_data[0], test_data[1]
                X_test = X_test.to(torch.float32).to(device)
                y_test = y_test.to(torch.float32).to(device)

                pred_y_test = model(X_test)
                pred_y_test_list.append(pred_y_test.detach().cpu().numpy())
    return pred_y_test_list


def train_classification(args, model, dl_train, dl_val, dl_test, device, verbose=True):
    best_loss = float('inf')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=False, min_lr=1e-4)

    for epoch in range(1, args.epochs + 1):
        model.train()
        tic = time.time()
        loss_train_list, acc_train_list = [], []
        criterion = torch.nn.CrossEntropyLoss()

        '''Training'''
        for i, train_data in enumerate(dl_train):
            optimizer.zero_grad()
            X_train = (train_data[:, 0:-1]).to(torch.float32).to(device)
            D_train = (train_data[:, -1]).to(torch.int64).to(device)
            # predict the probability
            D_train_pred = model(X_train)
            loss = criterion(D_train_pred, torch.squeeze(D_train))
            loss.backward()
            optimizer.step()

            loss_train_list.append(loss.item())
            acc_train_list.append(((torch.argmax(D_train_pred, dim=1) == D_train) * 1.0).cpu().detach().numpy())

            if verbose:
                print('\rEpoch [{}/{}], Batch [{}/{}], Loss = {:.4f}, Acc = {:.4f}%, time elapsed = {:.2f}, '
                      .format(epoch, args.epochs, i + 1, len(dl_train), np.mean(loss_train_list),
                              np.mean(acc_train_list) * 100, time.time() - tic), end='')

        '''Validation'''
        with torch.no_grad():
            model.eval()
            acc_val_list = []
            for i, val_data in enumerate(dl_val):
                X_val = (val_data[:, 0:-1]).to(torch.float32).to(device)
                D_val = (val_data[:, -1]).to(torch.int64).to(device)
                # predict the probability
                D_val_pred = model(X_val)
                acc_val_list.append(((torch.argmax(D_val_pred, dim=1) == D_val) * 1.0).cpu().detach().numpy())

            if verbose:
                print('Validation: Acc={:.4f}%'.format(np.mean(acc_val_list) * 100))

            if np.mean(acc_val_list) < best_loss:
                best_loss = np.mean(acc_val_list)
                torch.save(model, 'class_models.pt')

        scheduler.step(np.mean(acc_val_list))

    '''Test'''
    with torch.no_grad():
        model.eval()
        model = torch.load('class_models.pt')
        for i, test_data in enumerate(dl_test):
            X_test = test_data.to(torch.float32).to(device)
            # predict the probability
            D_test_pred = model(X_test)
            softmax = nn.Softmax()
            predict_prob = softmax(D_test_pred)

    return predict_prob.cpu().detach().numpy()