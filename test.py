import cv2 
import numpy as np
import open3d as o3d

left_img_path = '/home/maaitrayo/Autonomous Vehicle/data_odometry_gray/dataset/sequences/00/image_0/000000.png'
right_img_path = '/home/maaitrayo/Autonomous Vehicle/data_odometry_gray/dataset/sequences/00/image_1/000000.png'

calibFile = open('/home/maaitrayo/Autonomous Vehicle/data_odometry_gray/dataset/sequences/00/calib.txt', 'r').readlines()

P1Vals = calibFile[0].split()

Proj1 = np.zeros((3,4))
for row in range(3):
    for column in range(4):
        Proj1[row, column] = float(P1Vals[row*4 + column + 1])

P2Vals = calibFile[1].split()
Proj2 = np.zeros((3,4))
for row in range(3):
    for column in range(4):
        Proj2[row, column] = float(P2Vals[row*4 + column + 1])

print(Proj1,"\n\n")
print(Proj2,"\n\n")

img_left = cv2.imread(left_img_path)
img_right = cv2.imread(right_img_path)

#cv2.imshow('left image', img_left)
#cv2.imshow('right image', img_right)

block = 15
P1 = block * block * 8
P2 = block * block * 32
disparityEngine = cv2.StereoSGBM_create(0, 96, 9, 8 * 9 * 9, 32 * 9 * 9, 1, 63, 10, 100, 32)
disparity = disparityEngine.compute(img_left, img_right).astype(np.float32)
ImT1_disparityA = np.divide(disparity, 255.0)
cv2.imshow('disparity', ImT1_disparityA)

rows_left = img_left.shape[0]
columns_left = img_left.shape[1]

rows_right = img_right.shape[0]
columns_right = img_right.shape[1]

total_points_left = rows_left * columns_left
total_points_right = rows_right * columns_right
z_left=0
z_right=0

left_img_points = np.zeros([total_points_left,2])
right_img_points = np.zeros([total_points_left,2])

for r in range(rows_left):
	for c in range(columns_left):
		left_img_points[z_left][0] = c
		left_img_points[z_left][1] = r
		z_left=z_left+1

for r in range(rows_right):
	for c in range(columns_right):
		right_img_points[z_right][0] = c
		right_img_points[z_right][1] = r
		z_right=z_right+1

print(left_img_points)
print(len(right_img_points))

trackPoints1_KLT_L = left_img_points
trackPoints2_KLT_L = right_img_points

trackPoints1_KLT_R = np.copy(trackPoints1_KLT_L)
trackPoints2_KLT_R = np.copy(trackPoints2_KLT_L)
selectedPointMap = np.zeros(trackPoints1_KLT_L.shape[0])
#print(len(selectedPointMap))

disparityMinThres = 0.0
disparityMaxThres = 100.0
for i in range(trackPoints1_KLT_L.shape[0]):
    T1Disparity = ImT1_disparityA[int(trackPoints1_KLT_L[i,1]), int(trackPoints1_KLT_L[i,0])]
    #T2Disparity = ImT2_disparityA[int(trackPoints2_KLT_L[i,1]), int(trackPoints2_KLT_L[i,0])]
    
    if (T1Disparity > disparityMinThres and T1Disparity < disparityMaxThres): 
        #and T2Disparity > disparityMinThres and T2Disparity < disparityMaxThres):
        trackPoints1_KLT_R[i, 0] = trackPoints1_KLT_L[i, 0] - T1Disparity
        #trackPoints2_KLT_R[i, 0] = trackPoints2_KLT_L[i, 0] - T2Disparity
        selectedPointMap[i] = 1
        
selectedPointMap = selectedPointMap.astype(bool)
trackPoints1_KLT_L_3d = trackPoints1_KLT_L[selectedPointMap, ...]
trackPoints1_KLT_R_3d = trackPoints1_KLT_R[selectedPointMap, ...]

print("left",trackPoints1_KLT_L_3d)
print("right",trackPoints1_KLT_R_3d)

numpoints = len(trackPoints1_KLT_L_3d)

points3D = np.zeros((numpoints,3))
print(points3D)

for i in range(numpoints):
    pLeft = trackPoints1_KLT_L_3d[i,:]
    pRight = trackPoints1_KLT_R_3d[i,:]
    
    X = np.zeros((4,4))
    X[0,:] = pLeft[0] * Proj1[2,:] - Proj1[0,:]
    X[1,:] = pLeft[1] * Proj1[2,:] - Proj1[1,:]
    X[2,:] = pRight[0] * Proj2[2,:] - Proj2[0,:]
    X[3,:] = pRight[1] * Proj2[2,:] - Proj2[1,:]
    
    [u,s,v] = np.linalg.svd(X)
    v = v.transpose()
    vSmall = v[:,-1]
    vSmall /= vSmall[-1]

    points3D[i, :] = vSmall[0:-1]

print(points3D)
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points3D)
o3d.visualization.draw_geometries([pcd])

cv2.waitKey(0)