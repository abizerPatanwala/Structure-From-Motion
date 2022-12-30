import numpy as np
import copy 
import cv2

def extract_matches(filename,dest_img):
  f = open(filename,'r')
  f_lines = []
  for count,line in enumerate(f):
    f_lines.append(line.strip().split(' '))
  imgsrc_points = []
  imgdst_points = []
  for i in range(1, len(f_lines)):
    if (int(f_lines[i][0]) == 2):
      if (int(f_lines[i][6]) == dest_img):
        imgsrc_points.append([float(f_lines[i][4]), float(f_lines[i][5])])
        imgdst_points.append([float(f_lines[i][7]), float(f_lines[i][8])])
    elif (int(f_lines[i][0]) == 3):
      if (int(f_lines[i][6]) == dest_img):
        imgsrc_points.append([float(f_lines[i][4]), float(f_lines[i][5])])
        imgdst_points.append([float(f_lines[i][7]), float(f_lines[i][8])])
      if (int(f_lines[i][9]) == dest_img):
        imgsrc_points.append([float(f_lines[i][4]), float(f_lines[i][5])])
        imgdst_points.append([float(f_lines[i][10]), float(f_lines[i][11])])
    elif (int(f_lines[i][0]) == 4):
      if (int(f_lines[i][6]) == dest_img):
        imgsrc_points.append([float(f_lines[i][4]), float(f_lines[i][5])])
        imgdst_points.append([float(f_lines[i][7]), float(f_lines[i][8])])
      if (int(f_lines[i][9]) == dest_img):
        imgsrc_points.append([float(f_lines[i][4]), float(f_lines[i][5])])
        imgdst_points.append([float(f_lines[i][10]), float(f_lines[i][11])])
      if (int(f_lines[i][12]) == dest_img):
        imgsrc_points.append([float(f_lines[i][4]), float(f_lines[i][5])])
        imgdst_points.append([float(f_lines[i][13]), float(f_lines[i][14])])
    elif (int(f_lines[i][0]) == 5):
      if (int(f_lines[i][6]) == dest_img):
        imgsrc_points.append([float(f_lines[i][4]), float(f_lines[i][5])])
        imgdst_points.append([float(f_lines[i][7]), float(f_lines[i][8])])
      if (int(f_lines[i][9]) == dest_img):
        imgsrc_points.append([float(f_lines[i][4]), float(f_lines[i][5])])
        imgdst_points.append([float(f_lines[i][10]), float(f_lines[i][11])])
      if (int(f_lines[i][12]) == dest_img):
        imgsrc_points.append([float(f_lines[i][4]), float(f_lines[i][5])])
        imgdst_points.append([float(f_lines[i][13]), float(f_lines[i][14])])
      if (int(f_lines[i][15]) == dest_img):
        imgsrc_points.append([float(f_lines[i][4]), float(f_lines[i][5])])
        imgdst_points.append([float(f_lines[i][16]), float(f_lines[i][17])])
  imgsrc_points = np.array(imgsrc_points)
  imgdst_points = np.array(imgdst_points)
  return imgsrc_points,imgdst_points

def clean(temp1,temp2):
  img1_points = []
  img2_points = []  
  for i in range(len(temp1)):
    if (temp1[i] not in img1_points) and (temp2[i] not in img2_points):
      img1_points.append(temp1[i])
      img2_points.append(temp2[i])
    elif (temp1[i] in img1_points):
      ind = img1_points.index(temp1[i])
      if(temp2[i] != img2_points[ind]):
        img1_points.remove(temp1[i])
        img2_points.remove(img2_points[ind])
    elif (temp2[i] in img2_points):
      ind = img2_points.index(temp2[i])
      if(temp1[i] != img1_points[ind]):
        img2_points.remove(temp2[i])
        img1_points.remove(img1_points[ind])
  
  return img1_points, img2_points  

def write_matches(img1,img1_points,img2,img2_points,width,string):
  img1t = copy.deepcopy(img1)
  img2t = copy.deepcopy(img2)
  for i in range(len(img1_points)):
    cv2.circle(img1t,(int(round(img1_points[i][0])),int(round(img1_points[i][1]))),2,(255,0,0),-1)
  for i in range(len(img2_points)):
    cv2.circle(img2t,(int(round(img2_points[i][0])),int(round(img2_points[i][1]))),2,(255,0,0),-1)
  
  img_con = cv2.hconcat([img1t, img2t])
  for i in range(len(img1_points)):
    img_con = cv2.line(img_con,(int(round(img1_points[i][0])),int(round(img1_points[i][1]))),(int(round(img2_points[i][0])) + width ,int(round(img2_points[i][1]))),(0,0,255),1)
  cv2.imwrite(string + ".png", img_con)

def write_reproj(img,P,wpts,inl, string):
  reproj_img = copy.deepcopy(img)
  for i in range(len(wpts)):
    pt = wpts[i]
    reproj = np.matmul(P,pt.T)
    reproj = reproj / reproj[2]
    cv2.circle(reproj_img,(int(round(reproj[0])),int(round(reproj[1]))),2,(0,0,255), -2)
    cv2.circle(reproj_img,(int(round(inl[i][0])),int(round(inl[i][1]))),2,(255,0,0), 1)
  cv2.imwrite(string + ".png", reproj_img) 
  return None 

def write_reproj_ba(img1,img2, img3, inl_global, worldpts_globalf, P_allf,V):
  reproj_img1 = copy.deepcopy(img1)
  reproj_img2 = copy.deepcopy(img2)
  reproj_img3 = copy.deepcopy(img3)
  for i in range(len(worldpts_globalf)):
    if (V[0][i] == 1 and V[1][i] == 1 and V[2][i] == 0):
      reproj = np.matmul(P_allf[1],worldpts_globalf[i])
      reproj = reproj / reproj[2]
      cv2.circle(reproj_img2,(int(reproj[0]),int(reproj[1])),2,(0,0,255), -2)
      cv2.circle(reproj_img2,(int(inl_global[1][i][0]),int(inl_global[1][i][1])),2,(255,0,0), 1)
    if (V[0][i] == 1 and V[1][i] == 0 and V[2][i] == 1):
      reproj = np.matmul(P_allf[2],worldpts_globalf[i])
      reproj = reproj / reproj[2]
      cv2.circle(reproj_img3,(int(reproj[0]),int(reproj[1])),2,(0,0,255), -2)
      cv2.circle(reproj_img3,(int(inl_global[2][i][0]),int(inl_global[2][i][1])),2,(255,0,0), 1) 
  cv2.imwrite("reproj2_BA.png", reproj_img2)
  cv2.imwrite("reproj3_BA.png", reproj_img3)  
  return None   

def extract_comnew_inl(inl,inlcm,inlref,world_pt):
  X = copy.deepcopy(world_pt)
  inl_common = []
  inl_new = []
  inlcm_new = []
  inlcm_common = []
  world_pt_common= []
  for i in range(len(inl)):
    flag = 0
    for j in range(len(inlref)):
      if (inlcm[i][0] == inlref[j][0] and inlcm[i][1] == inlref[j][1]):
        flag = 1
        index = j
        break
    if flag == 1:
      world_pt_common.append(X[index])
      inl_common.append(inl[i])
      inlcm_common.append(inlcm[i])
    if flag == 0:
      inl_new.append(inl[i])
      inlcm_new.append(inlcm[i])
  inl_common = np.array(inl_common)
  inl_new = np.array(inl_new)
  inlcm_new = np.array(inlcm_new)
  world_pt_common = np.array(world_pt_common) 
  inlcm_common = np.array(inlcm_common)   
  return inl_common, inl_new,  inlcm_common, inlcm_new, world_pt_common

def addworldpts_to_global(pts_global,pts_current):
  for i in range(len(pts_current)):
    pts_global.append(pts_current[i])
  return pts_global 

def addinl_to_global(inlpts_global, inlpts_current, worldpts_global, worldpts_current, img):
  if len(inlpts_global) > img:
    for i in range(len(worldpts_global)):
      for j in range(len(worldpts_current)):
         if ((worldpts_global[i][0] == worldpts_current[j][0]) and (worldpts_global[i][1] == worldpts_current[j][1])):
          inlpts_global[img-1][i][0] = inlpts_current[j][0]
          inlpts_global[img-1][i][1] = inlpts_current[j][1]
  
  if len(inlpts_global) < img:
    tmp = np.zeros((len(worldpts_global), 2))
    for i in range(len(worldpts_global)):
      for j in range(len(worldpts_current)):
         if ((worldpts_global[i][0] == worldpts_current[j][0]) and (worldpts_global[i][1] == worldpts_current[j][1])):
          tmp[i][0] = inlpts_current[j][0]
          tmp[i][1] = inlpts_current[j][1]
    tmp = list(tmp) 
    inlpts_global.append(tmp)    
  return inlpts_global  


def append_camera_indices(camera_indices,worldpts_global, worldpts, match, new_or_common):
  if (match == '12'):
    for i in range(len(worldpts_global)):
      camera_indices.append([1, 2, -1,-1,-1])
  else:
    if (new_or_common == 'new'):
      for i in range(len(worldpts)):
        tmp = [-1,-1,-1,-1,-1]
        tmp[int(match[0]) - 1] = int(match[0])
        tmp[int(match[1]) - 1] = int(match[1])
        camera_indices.append(tmp)
    elif (new_or_common == 'common'):
      for i in range(len(worldpts)):
        index = []
        for j in range(len(worldpts_global)):
          if ((worldpts[i][0] == worldpts_global[j][0]) and (worldpts[i][1] == worldpts_global[j][1])):
            index.append(j) 
          for k in range(len(index)):
            camera_indices[index[k]][int(match[1])-1] = int(match[1])    
  return camera_indices  

def visibility_matrix(camera_indices,worldpts_length,num_cameras):
  V = np.zeros((num_cameras,worldpts_length))
  for i in range(num_cameras):
    for j in range(worldpts_length):
      if i+1 in camera_indices[j]:
        V[i][j] = 1 
  return V
