import numpy as np
import cv2 
from random import randint
import copy
import math
import epipolar as ep

def hransac(src_pts,dst_pts,threshold,N_max, inlper):
  length = len(src_pts)
  inlsrc_m = []
  inldst_m = []
  N = 1
  inlper_c = 0
  
  while((N < N_max) and (inlper_c < inlper)): 
    index = np.array([-1,-1,-1,-1])
    inlsrc_c = []
    inldst_c = []
    for i in range(4):
      a = randint(0,length-1)
      while(True):
        if (a not in index):
          index[i] = a
          break
        else:
          a = randint(0, length-1) 
    #print(index)
    randpt_src = [src_pts[i] for i in index]  
    randpt_dst = [dst_pts[i] for i in index]
    randpt_src = np.array(randpt_src)
    randpt_dst = np.array(randpt_dst)
    H,_ = cv2.findHomography(randpt_src,randpt_dst) 
    if(H[2][2]!= 0):
      for i in range(0,length): 
       src = list(src_pts[i].astype('float'))
       dst = list(dst_pts[i].astype('float'))
       src.append(1)
       dst.append(1)
       dst_est = np.matmul(H,src)
       #print(dst_est)
       dst_est[0] = dst_est[0] / dst_est[2];
       dst_est[1] = dst_est[1] / dst_est[2];
       if (math.isnan(dst_est[0]) or math.isnan(dst_est[0])):
        continue
       else:
         dst_est[0] = round(dst_est[0])
         dst_est[1] = round(dst_est[1]) 
         error = math.sqrt((dst[0]-dst_est[0])**2 + (dst[1] - dst_est[1])**2)
         if error < threshold:
          inlsrc_c.append(src_pts[i])
          inldst_c.append(dst_pts[i])
      
      if len(inlsrc_c) > len(inlsrc_m):
        inlsrc_m = copy.deepcopy(inlsrc_c)
        inldst_m = copy.deepcopy(inldst_c)
      inlper_c = (float(len(inlsrc_m)) / float(len(src_pts))) * 100.0   
      N = N + 1
  inlsrc_m = np.array(inlsrc_m)
  inldst_m = np.array(inldst_m)
  return inlsrc_m,inldst_m       
  
def fransac(src_pts, dst_pts, threshold, N_max, inlper):
  length = len(src_pts)
  inlsrc_m = []
  inldst_m = []
  N = 1
  inlper_c = 0
  while((N < N_max) and (inlper_c < inlper)): 
    index = np.array([-1,-1,-1,-1,-1,-1,-1,-1])
    inlsrc_c = []
    inldst_c = []
    for i in range(8):
      a = randint(0,length-1)
      while(True):
        if (a not in index):
          index[i] = a
          break
        else:
          a = randint(0, length-1) 
    randpt_src = [src_pts[i] for i in index]  
    randpt_dst = [dst_pts[i] for i in index]
    F = ep.compute_Fmatrix(randpt_src,randpt_dst) 
    for i in range(0,length): 
     src = np.array(src_pts[i], dtype = np.dtype(float))
     dst = np.array(dst_pts[i], dtype = np.dtype(float))
     src = np.append(src,1)
     dst = np.append(dst,1)
     error = abs(np.matmul(np.matmul(src,F), np.transpose(dst)))
     if error < threshold:
      inlsrc_c.append(src_pts[i])
      inldst_c.append(dst_pts[i])
    
    if len(inlsrc_c) > len(inlsrc_m):
      inlsrc_m = []
      inldst_m = []
      inlsrc_m = copy.deepcopy(inlsrc_c)
      inldst_m = copy.deepcopy(inldst_c)
    inlper_c = (float(len(inlsrc_m)) / float(len(src_pts))) * 100.0   
    N = N + 1
  inlsrc_m = np.array(inlsrc_m)
  inldst_m = np.array(inldst_m)
  return inlsrc_m,inldst_m  
 
def PnPransac(inl,world_pts,threshold,N_max,inlper,K):
  inl_worldm = []
  inl_m = []
  P = np.empty((3,4))
  Pc = np.empty((3,4))
  N = 1
  inlper_c = 0
  while((N < N_max) and (inlper_c < inlper)): 
    inl_worldc = []
    inl_c = []
    while(True):
      flag = 0
      index_arr = np.random.randint(len(inl), size = (6))
      sliced_world = [] 
      sliced_inl = []
      for i in range(6):
        sliced_world.append(list(world_pts[index_arr[i]]))
        sliced_inl.append(list(inl[index_arr[i]]))
      for j in range(6):
        count = sliced_inl.count(sliced_inl[j])
        if count > 1:
          flag = 1
          break
      if flag == 0:
        break 
    P = ep.linearPnP(sliced_world, sliced_inl,K)
    for i in range(0,len(inl)): 
      u1 = inl[i][0]
      v1 = inl[i][1]
      a1 = np.divide(np.matmul(P[0],world_pts[i]), np.matmul(P[2],world_pts[i]))
      a2 = np.divide(np.matmul(P[1],world_pts[i]), np.matmul(P[2],world_pts[i]))
      error = (u1 - a1)**2 + (v1 - a2)**2
      if error < threshold:
        inl_worldc.append(world_pts[i])
        inl_c.append(inl[i])
      
    if len(inl_worldc) > len(inl_worldm):
      inl_worldm = copy.deepcopy(inl_worldc)
      inl_m = copy.deepcopy(inl_c)
      Pc = copy.deepcopy(P)
    inlper_c = (float(len(inl_worldm)) / float(len(inl))) * 100.0   
    N = N + 1
  inl_worldm = np.array(inl_worldm)
  inl_m = np.array(inl_m)
  Pc = np.array(Pc)
  return Pc


