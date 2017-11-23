#! /usr/bin/env python
#-*- coding:UTF-8 -*-

import argparse


def get_parser():

    parser=argparse.ArgumentParser(description='pytorch deidentify&siamese network in face similarity training.')
    ################################# Data Loader #############################################
    parser.add_argument('--training_dir',type=str,default='/media/Dataset/ORLface/training/',help='the directory of training dataset ')
    parser.add_argument('--testing_dir',type=str,default='/media/Dataset/ORLface/testing/',help='the directory of testing dataset')
    parser.add_argument('--validate_dir',type=str,default='/media/Dataset/ORLface/validating/',help='the directory of validate dataset')
    parser.add_argument('--num_workers',type=int,default=8)
    parser.add_argument('--Siamese_training_batch_size',type=int,default=64,help='batch size of siamesenetwork training set')
    parser.add_argument('--Siamese_test_batch_size',type=int,default=1,help='batch size of siamesenetwork testing set')
    parser.add_argument('--Siamese_validate_batch_size',type=int ,default=16,help='batch size of siamesenetwork validate set')
    parser.add_argument('--Reidentify_training_batch_size',type=int,default=64,help='batch size of reidentifynetwork training set')
    parser.add_argument('--Reidentify_test_batch_size',type=int,default=1,help='batch size of reidentifynetwork testing set')
    parser.add_argument('--Reidentify_validate_batch_size',type=int ,default=16,help='batch size of reidentifynetwork validate set')

    ################################# Model #############################################
    parser.add_argument('--cuda',type=bool,default=False,help='if the GPU is available')
    parser.add_argument('--seed',type=int,default=1,help='manual seed ')

    ################################# Train & Validate #############################################
    parser.add_argument('--valfre',type=int,default=10,help='frequency of validate')
    parser.add_argument('--patience',type=int,default=1,help='max number of validate')

    ################################ Deidentification #############################################
    parser.add_argument('--n',type=int,default=10,help='arnold transform n')
    parser.add_argument('--a',type=int,default=3,help='arnold transform a')
    parser.add_argument('--b',type=int,default=5,help='arnold transform b')

    ################################# Optimizer #############################################
    parser.add_argument('--Siamese_Start_epoch',type=int,default=0,help='the number of siamesenetwork checkpoint start epoch')
    parser.add_argument('--Reidentify_Start_epoch',type=int,default=0,help='the number of reidentify checkpoint start epoch')
    parser.add_argument('--resume',type=str,default='./')
    parser.add_argument('--Siamese_train_number_epochs',type=int,default=200)
    parser.add_argument('--Reidentify_train_number_epochs',type=int,default=500)
    parser.add_argument('--siamese_lr',type=int,default=0.0005)
    parser.add_argument('--reidentify_lr',type=int,default=0.0005)
    parser.add_argument('--bit_length',type=int,default=64)
    parser.add_argument('--lambda',type=int,default=0.5)#[0.2,0.5,1,2]
    parser.add_argument('--beta',type=int,default=0.1)#[0.1,0.5,1,2,4]
    parser.add_argument('--train_same_class_num',type=int,default=10)
    parser.add_argument('--test_same_class_num',type=int,default=10)

    return parser
