#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 16:18:27 2019

@author: pohsuanh

This code runs salinet_gaze.py in multiprocess
"""

import time
from multiprocessing import Process, cpu_count, JoinableQueue
import salient_gaze, vars
import os, glob, sys
import numpy as np

class Worker(Process):          
    
      def __init__(self, queue):
          Process.__init__(self)
          self.queue=queue
          
      def run(self):
          np.random.seed()
          
          while True:
              job=self.queue.get()
              if not job:
                  print('Exiting...', self.name)
                  print('Job, ', job)
                  self.queue.task_done()
                  break
              else :
                  print('working... ',job)
                  salient_gaze.run(job)
                  self.queue.task_done()              
          
          
if __name__ == '__main__':
    
    global start
        
    Overwrite = False
    
    ''' Check if the folders exist && empty. If not, create a new folder.'''
     
    gaze_dir = '/media/pohsuanh/Data/Toxic Stress/eye_tracking_data/'
    
#    gaze_dir = '/media/pohsuanh/Data/ONDRI/eye_tracking_data/'

    data_set_name = 'CHLA'

    fp = sorted( glob.glob(os.path.join(gaze_dir, data_set_name, '*', '*Clips.mat')))

    print('check overwrite')
    
    keep = []
        
    for p in fp :
    
        subj_folder = os.path.dirname(p).split('/')[-1]
       
        filename = '{}_salient_gaze_conv.p'.format( subj_folder )
                            
        output = os.path.join( vars.fig_dir, data_set_name, subj_folder, filename)
                      
        if os.path.exists(output) : # do not overwrite existing files
                        
            keep.append(p)
    
    for x in keep :
        
        fp.remove(x)
    
    print( 'Multiprocess...')
    
    start = time.time()
    
    job_queue = JoinableQueue()
    
    for f in fp : 
        
        job_queue.put( f )
        
    process_list = []

    ''' Generate data'''
        
    for p in range(2*cpu_count()-5): #range(2*cpu_count()-1): # PROCESS_NUM
         job_queue.put(None)
         process = Worker(job_queue)
         process_list.append(process)  
         process.start()
    
    for p in process_list : p.join()
    
    job_queue.join()
    end = time.time()   

    print( 'time: ', end-start  )
    
      