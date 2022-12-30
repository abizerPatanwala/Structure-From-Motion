import numpy as np
import copy
import math
from scipy.optimize import least_squares

def compute_Fmatrix(img1_points,img2_points):
  p1 = copy.deepcopy(img1_points)
  p2 = copy.deepcopy(img2_points)
  A = np.empty((len(img1_points),9))
  F = np.zeros((3,3))
  for i in range(len(img1_points)):
    x1 = p1[i][0]
    y1 = p1[i][1]
    x1d = p2[i][0]
    y1d = p2[i][1] 
    A[i] = [x1*x1d, x1*y1d, x1, y1*x1d, y1*y1d, y1, x1d, y1d,1]  
  u,s,vh = np.linalg.svd(A)
  sol = vh[-1]
  F = [sol[0:3],sol[3:6],sol[6:9]]
  F = np.array(F)
  F = np.transpose(F)
  u,s,vh = np.linalg.svd(F)
  s[-1] = 0
  temp = np.matmul(u, np.diag(s))
  F = np.matmul(temp, vh)
  return F 

def E_from_F(F,K):
  K = np.array(K)
  E = np.matmul(np.matmul(np.transpose(K),F),K)
  u,s,vh = np.linalg.svd(E)
  s = [1,1,0]
  temp = np.matmul(u, np.diag(s))
  E = np.matmul(temp, vh)
  return E

def ExtractCameraPose(E):
  w = [[0,-1,0],[1,0,0],[0,0,1]]
  u,s,vh = np.linalg.svd(E)
  t = np.zeros((4,3,1))
  R = np.zeros((4,3,3))
  wt = np.transpose(w)
  R[0] = np.matmul(np.matmul(u,w),vh) 
  R[1] = np.matmul(np.matmul(u,w),vh) 
  R[2] = np.matmul(np.matmul(u,wt),vh) 
  R[3] = np.matmul(np.matmul(u,wt),vh)
  t[0] = np.reshape(u[:,2],(3,1))
  t[1] = np.reshape(-u[:,2], (3,1))
  t[2] = np.reshape(u[:,2], (3,1))
  t[3] = np.reshape(-u[:,2], (3,1))
  if (np.linalg.det(R[0]) < 0):
    R[0] = -1 * R[0]
    R[1] = -1 * R[1]
    R[2] = -1 * R[2]
    R[3] = -1 * R[3]
    t[0] = -1 * t[0]
    t[1] = -1 * t[1]
    t[2] = -1 * t[2]
    t[3] = -1 * t[3]
  return R,t  
 
  
def linear_triangulate(P1,P2,inl1,inl2):
  A = np.zeros([4,4])
  world_pts = np.zeros((len(inl1),4))
  for i in range(len(inl1)):
    x = inl1[i][0]
    y = inl1[i][1]
    xd = inl2[i][0]
    yd = inl2[i][1]
    A[0] = x*P1[2] - P1[0]
    A[1] = y*P1[2] - P1[1]
    A[2] = xd*P2[2] - P2[0]
    A[3] = yd*P2[2] - P2[1]
    u,s,vh = np.linalg.svd(A)
    sol = vh[-1]
    sol = sol/sol[3]
    world_pts[i] = sol
  return world_pts
  
def disam_cam_pose(P1, R,t,K,inl1,inl2):
  inlm = 0
  world_pts = []
  Rf = []
  Cf = []
  for i in range(4):
    inlc = 0
    RT = np.hstack((R[i],t[i]))
    P2 = np.matmul(K,RT)
    world_pts_temp = linear_triangulate(P1,P2,inl1,inl2)
    C = -1 * np.matmul(np.linalg.inv(R[i]),t[i])
    for j in range(len(world_pts_temp)):
      pt = np.reshape(world_pts_temp[j],(4,1))[0:3]
      c1 = np.matmul([0.0,0.0,1.0],pt)
      c2 = np.matmul(R[i][:,2],pt - C )
      if c1 > 0.0 and c2 > 0.0:
        inlc = inlc + 1
    if inlc > inlm:
      inlm = inlc
      Pc = copy.deepcopy(P2)
      world_pts = copy.deepcopy(world_pts_temp) 
  return Pc,world_pts

def error_func_triang(X,inl1,inl2,P1,P2):
  error = []
  u1 = inl1[0]
  v1 = inl1[1]
  u2 = inl2[0]
  v2 = inl2[1]
  a1 = np.matmul(P1[0],X) / np.matmul(P1[2],X)
  a2 = np.matmul(P1[1],X) / np.matmul(P1[2],X)
  b1 = np.matmul(P2[0],X) / np.matmul(P2[2],X)
  b2 = np.matmul(P2[1],X) / np.matmul(P2[2],X)
  error.append((a1 - u1))
  error.append((a2 - v1))
  error.append((b1 - u2))
  error.append((b2 - v2))
  return error
  
def nonlinear_triangulate(P1,P2,inl1,inl2,world_pts):
  X0 = copy.deepcopy(world_pts)
  world_pts_opt = []
  for i in range(len(world_pts)):
    tmp = least_squares(error_func_triang, X0[i], args = [inl1[i], inl2[i], P1, P2])
    pt = tmp.x
    pt = pt / pt[3]
    world_pts_opt.append(pt)
  world_pts_opt = np.array(world_pts_opt)
  return world_pts_opt 

def linearPnP(world_pts,inl,K):
  P = np.empty((3,4))
  A = np.empty((len(inl*2),12))
  for i in range(len(inl)):
    X = world_pts[i][0]
    Y = world_pts[i][1]
    Z = world_pts[i][2]
    x = inl[i][0]
    y = inl[i][1]
    A[2*i] = [X ,Y, Z, 1, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x]
    A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -y*X, -y*Y, -y*Z, -y]
  u,s,vh = np.linalg.svd(A)
  sol = np.transpose(vh)[:,-1]
  P = [sol[0:4],sol[4:8],sol[8:12]]
  P = np.array(P)
  '''
  gama_R = np.matmul(np.linalg.inv(K), P[:, 0:3])
  u,s,vh = np.linalg.svd(gama_R)
  gamma = s[0]
  R = np.matmul(u,vh)
  t = np.reshape(P[:,3],(3,1))
  t = np.matmul(np.linalg.inv(K),t) / gamma
  RT = np.hstack((R,t))
  P = np.matmul(K,RT)
  '''
  return P

def error_func_PnP(P,inl,world_pts,shape):
  P = np.reshape(P, (3,4))
  error = []
  for i in range(0,len(inl)): 
      u1 = inl[i][0]
      v1 = inl[i][1]
      a1 = np.divide(np.matmul(P[0],world_pts[i]), np.matmul(P[2],world_pts[i]))
      a2 = np.divide(np.matmul(P[1],world_pts[i]), np.matmul(P[2],world_pts[i]))
      error.append((u1 - a1)**2 + (v1 - a2)**2)
  error_sum = 0
  for j in range(len(inl)):
    error_sum = error_sum + error[j]
  return error

def nonlinear_Pnp(inl,world_pts,P0,K):
  shape = P0.shape
  P0 = np.reshape(P0,shape[0]*shape[1])
  P_opt = np.array((12,1))
  optim = least_squares(error_func_PnP,P0,args = (inl,world_pts,shape))   
  P_opt = optim.x
  P_opt = np.reshape(P_opt,shape)
  
  '''
  gama_R = np.matmul(np.linalg.inv(K), P_opt[:, 0:3])
  u,s,vh = np.linalg.svd(gama_R)
  gamma = s[0]
  R_opt = np.matmul(u,vh)
  t_opt = np.reshape(P_opt[:,3],(3,1))
  t_opt = np.matmul(np.linalg.inv(K),t_opt) / gamma
  RT = np.hstack((R_opt,t_opt))
  P_opt = np.matmul(K,RT)
  '''
  return P_opt

def error_func_ba(X,V,inl_global,camera_length,worldpt_length):
  P = []
  error = []
  for i in range(camera_length):
    tmp = [[0]*4 for l in range(3)]
    for j in range(3):
      for k in range(4):
        tmp[j][k] = X[12*i +4*j +k]
    P.append(tmp)  
  world_pts = []
  tmp = []
  for i in range(12*camera_length , len(X)-3, 4):
    tmp = [X[i], X[i+1], X[i+2], X[i+3]]
    world_pts.append(tmp)
  for i in range(camera_length):
    for j in range(worldpt_length):
      reproj =  np.matmul(P[i], world_pts[j])
      X = reproj[0] / reproj[2]
      Y = reproj[1] / reproj[2]
      u = inl_global[i][j][0]
      v = inl_global[i][j][1]
      #error = error +V[i][j] * ((u-X)**2 + (v-Y)**2)
      error.append(V[i][j] * (X-u))
      error.append(V[i][j] * (Y-v))
  return error   

def bundle_adjustment(P_all, V, worldpts_global,inl_global):
  camera_length = len(P_all)
  worldpt_length = len(worldpts_global)
  X0 = []
  X1 = []
  P_allf = []
  worldpts_globalf = []
  for i in range(camera_length):
    for j in range(3):
      for k in range(4):
        X0.append(P_all[i][j][k])
  for i in range(worldpt_length):
    X0.append(worldpts_global[i][0])
    X0.append(worldpts_global[i][1])
    X0.append(worldpts_global[i][2])
    X0.append(worldpts_global[i][3])
  sol = least_squares(error_func_ba, X0, args= [V,inl_global,camera_length,worldpt_length])
  print(sol.success)
  print(sol.message)
  X1 = sol.x
  for i in range(camera_length):
    tmp = [[0]*4 for l in range(3)]
    for j in range(3):
      for k in range(4):
        tmp[j][k] = X1[12*i +4*j +k]
    P_allf.append(tmp)  
  for i in range(12*camera_length , len(X1)-3, 4):
    tmp = [X1[i], X1[i+1], X1[i+2], X1[i+3]]
    worldpts_globalf.append(tmp)
  return P_allf,worldpts_globalf  

