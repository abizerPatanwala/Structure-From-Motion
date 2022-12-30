#!/usr/bin/env python3

import cv2
import copy 
import sys
import epipolar as ep
import numpy as np
import ransac 
import data_proc as dp
import time



def main():
  worldpts_global = []
  P_all = []
  R_all = []
  T_all = []
  np.set_printoptions(threshold=sys.maxsize)
  K = np.array([[531.12215, 0, 407.19255],[0, 531.5417, 313.3087],[0, 0, 1]]) 
  origin = np.array([[1.0,0,0,0],[0,1.0,0,0],[0,0,1.0,0]])
  P1 = np.matmul(K,origin)
  R1 = np.array([[1.0 , 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
  T1 = np.array([[0.0], [0.0], [0.0]])  
  R_all.append(R1)
  T_all.append(T1)
  P_all.append(P1)
  
  img1 = cv2.imread("P3Data/1.png")
  img2 = cv2.imread("P3Data/2.png")
  img1_pts_12,img2_pts_12 = dp.extract_matches("P3Data/matching1.txt",2)
 
  inl1_12,inl2_12 = ransac.hransac(img1_pts_12,img2_pts_12,5,1000,95)

  dp.write_matches(img1,inl1_12,img2,inl2_12,800,"img1_img2_matches")
  
  F = ep.compute_Fmatrix(inl1_12,inl2_12)
  E = ep.E_from_F(F,K)
  [R,t] = ep.ExtractCameraPose(E)
  [P2, worldpts_12]= ep.disam_cam_pose(P1, R,t,K,inl1_12,inl2_12)
  P_all.append(P2)
  dp.write_reproj(img2,P2,worldpts_12,inl2_12,'reproj2_linear')
  dp.write_reproj(img1,P1,worldpts_12,inl1_12,'reproj1_linear')

  worldpts_12 = ep.nonlinear_triangulate(P1,P2,inl1_12,inl2_12,worldpts_12) 
  worldpts_global = dp.addworldpts_to_global(worldpts_global,worldpts_12)

  dp.write_reproj(img2,P2,worldpts_12,inl2_12,'reproj2_nonlinear')
  dp.write_reproj(img1,P1,worldpts_12,inl1_12,'reproj1_nonlinear')
  
  inl1_1p = copy.deepcopy(inl1_12)
  worldpts_1p = copy.deepcopy(worldpts_12) 
  #Pnp portion
  for i in range(3,6,1):
    imgc = cv2.imread("P3Data/" + str(i) + ".png")
    img1_pts_1c,imgc_pts_1c = dp.extract_matches("P3Data/matching1.txt",i)
    inl1_1c,inlc_1c = ransac.hransac(img1_pts_1c, imgc_pts_1c,5,1000,95)
    dp.write_matches(img1,inl1_1c,imgc,inlc_1c,800,"img1_img" + str(i) + "_matches")
    
    inlc_1c_common, inlc_1c_new, inl1_1c_common, inl1_1c_new, worldpts_1c_common = dp.extract_comnew_inl(inlc_1c, inl1_1c, inl1_1p,  worldpts_1p)
    Pc = ransac.PnPransac(inlc_1c_common, worldpts_1c_common,10,500,90,K)
    Pc_opt = ep.nonlinear_Pnp(inlc_1c_common, worldpts_1c_common,Pc,K)
    P_all.append(Pc_opt)
    worldpts_1c_new = ep.linear_triangulate(P1, Pc_opt, inl1_1c_new, inlc_1c_new)
    worldpts_1c_new_opt = ep.nonlinear_triangulate(P1,Pc_opt,inl1_1c_new,inlc_1c_new,worldpts_1c_new)
    
    worldpts_global = dp.addworldpts_to_global(worldpts_global, worldpts_1c_new_opt)
    
    dp.write_reproj(imgc,Pc,worldpts_1c_common,inlc_1c_common,'reproj' +  str(i) + '_linearPnp_common')
    dp.write_reproj(imgc,Pc_opt,worldpts_1c_common, inlc_1c_common, 'reproj' + str(i) + '_non_linearPnp_common')
    dp.write_reproj(imgc,Pc_opt,worldpts_1c_new,inlc_1c_new, 'reproj' + str(i) + '_lineartriang_new')
    dp.write_reproj(imgc,Pc_opt,worldpts_1c_new_opt,inlc_1c_new, 'reproj' + str(i) + '_nonlineartriang_new')
    
    inl1_1p = []
    worldpts_1p = []
    for j in range(len(inlc_1c_common)):
      inl1_1p.append(inl1_1c_common[j])
      worldpts_1p.append(worldpts_1c_common[j])
    for j in range(len(inlc_1c_new)):
      inl1_1p.append(inl1_1c_new[j])
      worldpts_1p.append(worldpts_1c_new_opt[j])  

  f = open("world_pts.txt",'w')
  for i in range(len(worldpts_global)):
    # f.write(f"{world_pts[i][0],world_pts[i][1],world_pts[i][2]}\n")
      f.write(f'{worldpts_global[i][0]}, ')
      f.write(f'{worldpts_global[i][1]}, ')
      f.write(f'{worldpts_global[i][2]}, ')
      f.write(f'{worldpts_global[i][3]}\n')

  





if __name__ == "__main__":
    main()
