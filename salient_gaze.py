#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 12:30:35 2019

@author: pohsuanh

Salient_gaze()  Retrieve saliency values at gaze position for CHLA and BCH dataset of Toxic Stress Project 2019

"""
import numpy as np
import math
import os
import scipy as sp
from scipy.io import loadmat
from scipy.signal import convolve2d
import h5py 
import vars
import matplotlib.pyplot as plt
import subprocess
import pickle
import re
from glob import glob
################## Functions #####################
    
def preprocess_gaze_points(gaze_start, gaze_points_x, gaze_points_y, gaze_duration, clip_time) : 
    # Align X,Y from image coordinates to saliency map coordinates:
    # Input : 
    # t : list of time stamps
    # x : list of x position
    # y : list of y position
    # return :
    # tuple of gaze_position in saliency coordinates
    
    # remove invalid gaze duration fixation points
    
    exceps = np.isnan(gaze_duration)
        
    np.delete( gaze_points_x, exceps )
    
    np.delete( gaze_points_y, exceps )
    
    np.delete( gaze_start, exceps )
    
    # find the first time index in fixation that is larger than the time stamp of 1st clip frame.
    
    Gx = gaze_points_x[gaze_start > clip_time[0]]
    
    Gy = gaze_points_y[gaze_start > clip_time[0]]
    
    gaze_start = gaze_start[gaze_start > clip_time[0]]
    
    #Gx=  (Gx * vars.PPD)
    
    #Gy = (Gy * vars.PPD)
    
    # transform x,y from frame coordinate to saliency map coordinate
    
    newOrigin = np.subtract( [1920/2, 1200/2], [vars.Actual_Res['width'] / 2, vars.Actual_Res['height'] / 2])
    
    Gx = ( Gx - newOrigin[0] )*vars.Saliency_Map_Res['width']/vars.Actual_Res['width'] 
    
    Gy = ( Gy - newOrigin[1] )*vars.Saliency_Map_Res['height']/vars.Actual_Res['height'] 
    
    #  move points beyond boundaries to boundaries ( why not just discard or extrapolate ?)
    
    outlier = np.zeros(1)
    
    outlier = np.concatenate( (outlier, np.where(Gx >= vars.Saliency_Map_Res['width'])[0] )  )  #vars.Saliency_Map_Res['width'] - 1
    
    outlier = np.concatenate( (outlier, np.where(Gx < 0)[0]) ) 
    
    outlier = np.concatenate( (outlier, np.where( Gy >= vars.Saliency_Map_Res['height'])[0]) ) 
    
    outlier = np.concatenate( (outlier, np.where( Gy < 0 )[0]) )  
    
    outlier = np.unique(outlier)
    
    Gx = np.delete( Gx, outlier )
            
    Gy = np.delete( Gy, outlier )
            
    gaze_start = np.delete( gaze_start, outlier )
    
    return (Gx,Gy, gaze_start)


def get_sal_maps(file_path):
    # Return :
    # resturn saliency maps of dimension(width, height, time_step)
    #
    # Note : Can't use loadmat because source file is saved as .mat v7.3

    f =  h5py.File( file_path, 'r')
    
    sal_maps  = np.asanyarray(f.get('feats'))
    
    w = int( np.asanyarray( f.get('w'))[0][0] )
    
    h = int( np.asanyarray( f.get('h'))[0][0] )
    
    t = sal_maps.shape[1]
    
    sal_maps = sal_maps.reshape([w,h,t]).T
    
    return sal_maps

def find_ClipDisplay_time(clip_info):
    # clip_time register time steps of ClipDisplay, Warning, Play_Audio_Sound
    # The function returns the indices of clip_time that registers ClipDisplay, which is then used to retrieve corr. sal_map.
   return  [  i   for i, info in enumerate(clip_info) if 'ClipDisplay' in info[0] or 'display_blank' in info[0]]

def gaze_on_sal_map(gaze_time_steps, clip_time_steps, clip_display_time_idx) :
    #  return frame index of adjusted saliency map 
 
    clip_frame_idx = [];
        
    initial_K = 0 ;
        
    for t in gaze_time_steps :
                  
        for k  in range( initial_K, len(clip_display_time_idx)) :
            
            # if ClipDisplay
            if k < len(clip_display_time_idx) - 1 : 
  
                T0 = clip_time_steps[ clip_display_time_idx[k] ]
                
                T1 = clip_time_steps[ clip_display_time_idx[k+1] ]
            
                if t >= T0 and t < T1 :
                    
                    # Eyelink displays frames at varied interval,
                    
                    # corresponding frame index of saliency map decoded @30Hz
                    
                    clip_frame_idx.append(k)
                    
                    initial_K = k
                    
                    break
            
#            else :
#                
#                raise ValueError('Cannnot find %d in the original time_steps' % t)
#            
#                return # this is the statement that exit the function due to the error message
            
            # if display blank, we  should not take that fixation into account.
            elif k == len(clip_display_time_idx) - 1 : 
                
                T0 = clip_time_steps[ clip_display_time_idx[k] ]
                
                if t >= T0 :
                    
                    clip_frame_idx.append(-1)
                
                
                
    return clip_frame_idx

def n_th_gaze_on_sal_map(gaze_time_steps, clip_time_steps, bdry_time_id, n) :
    #  return frame index of of nth gaze and corresponding frame after clippet transition 
 
    gaze_idx = [] # idx of gaze that are n_th for each scene
    
    bdy_idx = [] # idx of scenes that contains the n_th gaze
        
    initial_K = 0 ;
    
    count = 0
        
    for i, t in enumerate(gaze_time_steps) :
                  
        for k  in range( initial_K, len(bdry_time_id)) :
                                    
            # if ClipDisplay
            if k < len(bdry_time_id) - 1 : 
  
                T0 = clip_time_steps[ bdry_time_id[k] ]
                
                T1 = clip_time_steps[ bdry_time_id[k+1]]
            
                if t >= T0 and t < T1 :
                                        
                    count += 1
                    # Eyelink displays frames at varied interval,
                    
                    # corresponding frame index of saliency map decoded @30Hz
                    
                    initial_K = k
                    
                    if count == n : 
                        
                        gaze_idx.append(i)
                        
                        bdy_idx.append(k)
                                            
                        initial_K = k + 1
                        
                        count = 0
                    
                        break
                    
            
#            else :
#                
#                raise ValueError('Cannnot find %d in the original time_steps' % t)
#            
#                return # this is the statement that exit the function due to the error message
            
            # if display blank, we  should not take that fixation into account.
            elif k == len(bdry_time_id) - 1 : 
                
                T0 = clip_time_steps[ bdry_time_id[k] ]
                
                if t >= T0 :
                    
                    gaze_idx.append(-1)
                
                
                
    return gaze_idx, bdy_idx

def compute_convolved_sal_maps(sal_maps):
    # compute the convolved saliency map by doing padded 2d convolution with a kernel of 2 degree * PDD
    
    kernel_map = np.ones([ vars.kernel_w, vars.kernel_h])
    
    # convolve the whole map first
    
    conv_maps = np.empty(sal_maps.shape)
    
    for i, sal_map in enumerate( sal_maps ) : 
        
        conv_maps[i,:,:] = convolve2d(sal_map,kernel_map, mode= 'same')
    
    return conv_maps

def compute_sal_values_at_gaze(Gx,Gy, sal_maps, clip_frame_idx, n = None):
    """
    Comptue the saliency values of the fixation points, if the clip frame index is not -1
    
    Inputs: 
      Gx, Gy  : gaze point (x,y)
      sal_map : saliency map (W,H)
      clip_frame_idx :  The frame index at the fixation.
      n : (optional) only return n_th fixation for each scene.
      
    Return : saliency value  
    """
    gaze_values = []
    
    vx, vy =  range(vars.Saliency_Map_Res['width']), range(vars.Saliency_Map_Res['height'])
    
    
    idx = 0
    
    for i, gaze in enumerate( zip( Gx, Gy) ) :
        
        gx,gy = gaze

        # if the corresponding index of clip_frame differs from the previous gaze point,
        # recompute the interpolation function of sal_map
    
        
        if idx != clip_frame_idx[i] : 
            
            idx = clip_frame_idx[i] 
            
            if idx != -1 : # if the sal frame is not display blank 
        
                sal_map = sal_maps[ idx ] 
                
                # bilinear interpoloation for gaze saliency value
    
                f = sp.interpolate.interp2d( vx, vy, sal_map, kind='linear', copy=True, bounds_error=False)
            
        # have to normalize
        
        gaze_values.append(f(gx, gy))
        
        
    return gaze_values

def compute_sal_values_at_nth_gaze(Gx, Gy, bdy_sal_maps, gaze_ind, bdy_ind) :
    
    gaze_values = []

    vx, vy =  range(vars.Saliency_Map_Res['width']), range(vars.Saliency_Map_Res['height'])
    
    for i, k in zip(gaze_ind, bdy_ind) : 
             
        gx, gy = Gx[i], Gy[i]  
        
        sal_map = bdy_sal_maps[k]
        
        f = sp.interpolate.interp2d( vx, vy, sal_map, kind='linear', copy=True, bounds_error=False)
        
        gaze_values.append(f(gx, gy))
        
    return gaze_values
    
def figsave_n_gaze_overlay( Gx, Gy, bdy_sal_maps, gaze_ind, bdy_ind ,dir_name) :
    
    sal_maps = bdy_sal_maps
    

    for i, k in zip(gaze_ind, bdy_ind) : 
             
        gx, gy = Gx[i], Gy[i]    
            
            
        fig, ax = plt.subplots()
         
        img = ax.imshow(sal_maps[k])
         
        # recenters origin from bottom left to top left
                 
        ax.add_patch(plt.Circle(( gx, gy), 1, color= 'red', clip_on = True) ) # gaze point
         
        ax.add_patch(plt.Circle( (gx, gy ), vars.kernel_h, color='blue', clip_on = True, alpha = 0.5) ) # gaze disk condiering DPP 
         
        plt.xticks([]) # remove the tick marks by setting to an empty list
    
        plt.yticks([])
         
        plt.tight_layout()
        
          
        if os.path.isdir(dir_name): 
            
            pass    
         
        else : 
            
           os.makedirs(dir_name)
            
           print("Path is created %s" % dir_name)
           
        filename = 'n_th_fixation_%03d.png' % i
        
        filename = os.path.join(dir_name, filename)
         
        plt.savefig( filename, transparent = True)
        
        plt.clf()
        
        plt.close()

def figsave_gaze_overlay( Gx, Gy, sal_maps, clip_frame_idx ,dir_name):
    ## SAVE FIGURES THAT SUPERPOSE GAZE LOCATION ON TOP OF SALIENCY MAPS
    
    idx = 0

    
    for i, gaze in enumerate( zip( Gx, Gy) ) :
        
        gx,gy = gaze
    
        # if the corresponding index of clip_frame differs from the previous gaze point,
        # recompute the interpolation function of sal_map
    
        
        if idx != clip_frame_idx[i] : 
            
            idx = clip_frame_idx[i] 
        
            sal_map = sal_maps[ idx ] 
        
        fig, ax = plt.subplots()
         
        img = ax.imshow(sal_map)
         
        # recenters origin from bottom left to top left
         
        ax.add_patch(plt.Circle((gx,gy), 1, color= 'red', clip_on = True) ) # gaze point
         
        ax.add_patch(plt.Circle( (gx, gy ), vars.kernel_h, color='blue', clip_on = True, alpha = 0.5) ) # gaze disk condiering DPP 
         
        plt.xticks([]) # remove the tick marks by setting to an empty list
    
        plt.yticks([])
         
        plt.tight_layout()
    
         
        if os.path.isdir(dir_name): 
            
            pass    
         
        else : 
            
           os.makedirs(dir_name)
            
           print("Path is created %s" % dir_name)
           
        filename = 'fixation_%03d.png' % i
        
        filename = os.path.join(dir_name, filename)
         
        plt.savefig( filename, transparent = True)
        
        plt.clf()
        
        plt.close()
        
#    subprocess.call("ffmpeg -f image2 -framerate 5 -s 288*432 -i {}/fixation_%".format(dir_name) +"03d.png" + " {}/out.gif".format(dir_name))

def save_file( obj, filename ):
     
        if os.path.isdir(os.path.dirname(filename)): 
            
            pass    
         
        else : 
            
           os.makedirs(os.path.dirname(filename))
            
           print("Path is created %s" % os.path.dirname(filename))
        
        with open(filename, 'wb') as f:
        
            pickle.dump(obj, f)
            
def find_BoundaryDisplay_time(clip_info, clip_key) :
    
    clip_boundaries_dict = ['blockA_clip_boundaries.mat', 'blockB_clip_boundaries.mat', 'blockC_clip_boundaries.mat', 'blockD_clip_boundaries.mat', 'blockE_clip_boundaries.mat', 'blockF_clip_boundaries.mat']

    bdy_dir ='/media/pohsuanh/Data/Toxic Stress/feature_maps/ClipBoundary'
    
    bdy_name = clip_boundaries_dict[clip_key]
        
    clippet_bdy_file = loadmat(os.path.join(bdy_dir, bdy_name))
    
    clip_dict = [ 'blockA', 'blockB', 'blockC', 'blockD', 'blockE', 'blockF']
    
    clip_name = clip_dict[clip_key]
        
    bdy_frames = clippet_bdy_file[clip_name]
        
    clip_dpt_id = [  i  for i, info in enumerate(clip_info) if 'ClipDisplay' in info[0] or 'display_blank' in info[0]]
        
    bdry_time_id = [ clip_dpt_id[i-1] for i in bdy_frames[0]]  
    
    return bdry_time_id        
    
#%%   
def run( p ) :
    # The main function to process saliency at fixation and saliency images
    # input :
    # p : .mat file
    # load the gaze position from .mat file 
    
    subj_folder = os.path.dirname(p).split('/')[-1]
    
    data_set_name = os.path.dirname(p).split('/')[-2]
    
    subj_data = loadmat(p)
    
    clip_keys = subj_data['subjData']['clipNums'][0][0][0] - 1
    
    clip_dict = [ 'blockA', 'blockB', 'blockC', 'blockD', 'blockE', 'blockF']

    # Load saliency maps of a clip from .mat file
 
    sal_map_dir = '/media/pohsuanh/Data/Toxic Stress/feature_maps/Smooth_Feature_Maps/'
    
    #clip_name = clip_dict[clip_key]
    
    channel_dict = ['C','F','I','M','O']
    
    # check if the file has been generated

    filename = '{}_salient_gaze_conv.p'.format( subj_folder )
                
    output = os.path.join( vars.fig_dir, data_set_name, subj_folder, filename)
                    
    if os.path.exists(output) :
        
        return
    
    # define output data structure
    
    struct = {}
    
    for i in clip_dict :
        struct[i] = {}
        for j in channel_dict :
            struct[i][j] = {}
    
    struct_conv = struct.copy()
    
    # start computing saliency values
         
    for i, clip_key in enumerate( clip_keys ) :  
        
        percentValidData = subj_data['subjData']['percentValidData'][0][0][0][i]
        
        clip_name = clip_dict[clip_key]
                
        if percentValidData < 80 :
            
            print('%s has percentVlidData %f' % (clip_name, percentValidData) )
            
            continue
            
        else :
        
            clip_time = subj_data['subjData']['messages'][0][0]['real_time'][0][i][0]
            
            clip_info = subj_data['subjData']['messages'][0][0]['info'][0][i][0]
            
            clip_display_time_idx = find_ClipDisplay_time(clip_info)
            
            bdry_time_id = find_BoundaryDisplay_time(clip_info, clip_key)
    
            gaze_start = subj_data['subjData']['fixs'][0][0]['start'][0][i][0]
            
            gaze_duration = subj_data['subjData']['fixs'][0][0]['duration'][0][i][0]
            
            gaze_points_x = subj_data['subjData']['fixs'][0][0]['posX'][0][i][0]
            
            gaze_points_y = subj_data['subjData']['fixs'][0][0]['posY'][0][i][0]
            
            gaze_pupil_size = subj_data['subjData']['fixs'][0][0]['pupilSize'][0][i][0]
            
            try : 
            
            	Gx, Gy, gaze_start = preprocess_gaze_points(gaze_start,gaze_points_x, gaze_points_y, gaze_duration, clip_time)
        
            	assert( Gx.size > 0  and Gy.size > 0) , "No valid gaze points."
            
            except AssertionError:
            
            	continue
        
            for chl in channel_dict :
                
                sal_file_name = 'feat%s.mat' % chl
                
                sal_file_path = os.path.join(sal_map_dir, clip_name , sal_file_name)
                
                sal_maps = get_sal_maps(sal_file_path)

                conv_maps = compute_convolved_sal_maps(sal_maps) 
                
                bdy_sal_maps = conv_maps[bdry_time_id]
                
                clip_frame_idx = gaze_on_sal_map( gaze_start, clip_time, clip_display_time_idx)
                
                gaze_ind, bdy_ind = n_th_gaze_on_sal_map( gaze_start, clip_time, bdry_time_id, 1)
                
                
                if False :
                    # for sparse feature 
             
                    gaze_values  = compute_sal_values_at_gaze( Gx, Gy, sal_maps, clip_frame_idx)
                    
                    struct[clip_dict[clip_key]][chl]['gaze_values'] =  gaze_values 
                    
                    filename = '%s_salient_gaze.p' % subj_folder
        
                    output = os.path.join( vars.fig_dir, data_set_name, subj_folder, filename)
        
                    save_file(struct, output)
                
                if True :

                    # for smooth feature
                    
                    gaze_values_convolve = compute_sal_values_at_gaze( Gx, Gy, conv_maps, clip_frame_idx) 
                    
                    struct_conv[clip_dict[clip_key]][chl]['gaze_values_convolved'] = gaze_values_convolve
                    
                    filename = '%s_salient_gaze_conv.p' % subj_folder
                
                    output = os.path.join( vars.fig_dir, data_set_name, subj_folder, filename)
                
                    save_file(struct_conv, output)
                
                if False : 

                    # for n_th gaze feature
                    
                    nth_gaze_values_convolve = compute_sal_values_at_nth_gaze( Gx, Gy, bdy_sal_maps, gaze_ind, bdy_ind)
                    
                    struct_conv[clip_dict[clip_key]][chl]['gaze_values_convolved'] = nth_gaze_values_convolve
                    
                    filename = '1st_gaze_{}_salient_gaze_conv.p'.format( subj_folder )
                
                    output = os.path.join( vars.fig_dir, data_set_name, subj_folder, filename)
                
                    save_file(struct_conv, output)
                
                
                ######### Make Gaze Overlay Saliency Map and Clip ####################
                 
                dir_name = os.path.join(vars.fig_dir,data_set_name, subj_folder , clip_name, chl) 
                
#                figsave_gaze_overlay( Gx, Gy, sal_maps, clip_frame_idx ,dir_name) 
                
#                figsave_n_gaze_overlay( Gx, Gy, bdy_sal_maps, gaze_ind, bdy_ind ,dir_name)  # compute the n-th gaze afterclippet transition.
                
