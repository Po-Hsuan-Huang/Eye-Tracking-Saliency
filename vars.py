#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:30:15 2019

@author: pohsuanh

Variables for Alma's CHLA and BCH data struct.

"""

# eyetracking_data_file_path

SubjData =  '/media/pohsuanh/Data/Toxic Stress/eye_tracking_data/'

PPD  = 38.7  # pixel per degree

Actual_Res = {'height': 720, 'width': 1280}

Saliency_Map_Res = {'height': 45, 'width': 80}

Saliency_Map_BaseDir = '/media/pohsuanh/Data/Toxic Stress/feature_maps/Smooth_Feature_Maps'

Map_Types = ['C','F', 'I', 'M', 'O']

Saliency_Map_Receptive_Field = {'height': 16, 'width': 16}  #GCD(45,80), to reduce the number of parameters for now


kernel_h = PPD *2 * Saliency_Map_Res['height'] / Actual_Res['height'] 

kernel_w = PPD *2 * Saliency_Map_Res['width']/ Actual_Res['width'] 

kernel_h, kernel_w = round( kernel_h), round(kernel_w)

fig_dir = '/media/pohsuanh/Data/Toxic Stress/feature_maps/Gaze_data/'



