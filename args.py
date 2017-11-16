#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import argparse


def get_parser():

    parser=argparse.ArgumentParser(description='pytorch siamese network in face similarity training.')
    ################################# Data Loader #############################################
    parser.add_argument('--training_dir',type=str,default='/media/Dataset/facedata/training/',help='the directory of training dataset ')
    parser.add_argument('--testing_dir',type=str,default='/media/Dataset/facedata/testing/',help='the directory of testing dataset')
    parser.add_argument('--validate_dir',type=str,default='/home/lzy/Work/re_identify/facedata/validate/',help='the directory of validate dataset')
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--training_batch_size',type=int,default=64,help='batch size of training set')
    parser.add_argument('--test_batch_size',type=int,default=1,help='batch size of testing set')
    parser.add_argument('--validate_batch_size',type=int ,default=16,help='batch size of testing set')

    ################################# Model #############################################
    parser.add_argument('--cuda',type=bool,default=False,help='if the GPU is available')
    parser.add_argument('--seed',type=int,default=1,help='manual seed ')

    ################################# Optimizer #############################################
    parser.add_argument('--start_epoch',type=int,default=100,help='the number of checkpoint start epoch')
    parser.add_argument('--resume',type=str,default='')
    parser.add_argument('--train_number_epochs',type=int,default=200)
    parser.add_argument('--lr',type=int,default=0.0005)
    parser.add_argument('--bit_length',type=int,default=64)
    parser.add_argument('--lambda',type=int,default=0.5)#[0.2,0.5,1,2]
    parser.add_argument('--beta',type=int,default=0.1)#[0.1,0.5,1,2,4]
    parser.add_argument('--train_same_class_num',type=int,default=10)
    parser.add_argument('--test_same_class_num',type=int,default=10)

    return parser
