import cv2 
import numpy as np
import open3d as o3d

left_img_path = '/home/maaitrayo/Autonomous Vehicle/data_odometry_gray/dataset/sequences/21/image_0/000000.png'
right_img_path = '/home/maaitrayo/Autonomous Vehicle/data_odometry_gray/dataset/sequences/21/image_1/000000.png'

img_left = cv2.imread(left_img_path)
img_right = cv2.imread(right_img_path)

#cv2.imshow('left image', img_left)
#cv2.imshow('right image', img_right)

block = 15
P1 = block * block * 8
P2 = block * block * 32
disparityEngine = cv2.StereoSGBM_create(minDisparity=0,numDisparities=16, blockSize=block, P1=P1, P2=P2)
disparity = disparityEngine.compute(img_left, img_right).astype(np.float32)
#print(disparity)
#cv2.imshow('disparity', disparity/255)

rows = img_right.shape[0]
columns = img_right.shape[1]

total_points =rows*columns
point_cloud = np.zeros([111294,3])
point_cloud_world = np.zeros([111294,3])
project_ini = np.ones([4,1])
project_final = np.ones([4,1])
#print(len(point_cloud))

#print(total_points)
#print(rows*columns)
z=0
a=0
fx = 718.856 
fy = 718.856 
cx = 607.1928 
cy = 185.2157
b = 0.573
Q = np.zeros([4,4])
Q[0][0] = 1
Q[1][1] = 1
Q[0][3] = -cx
Q[1][3] = -cy
Q[2][3] = -fx
Q[3][2] = -1/b
print(Q)

Q2 = np.float32([[1,0,0,0],
				[0,-1,0,0],
				[0,0,fx*0.05,0], #Focal length multiplication obtained experimentally. 
				[0,0,0,1]])


for r in range(rows):
	for c in range(columns):
		if(disparity[r][c] <= 0.0 or disparity[r][c] >= 96.0  ):
			continue

		x = (c - cx) / fx
		y = (r - cy) / fy
		depth = (fx*b) / disparity[r][c]
		point_cloud[z][0] = x * depth
		point_cloud[z][1] = y * depth
		point_cloud[z][2] = depth

		project_ini[0][0] = point_cloud[z][0]
		project_ini[1][0] = point_cloud[z][1]
		project_ini[2][0] = point_cloud[z][2]

		project_final = np.dot(Q,project_ini)

		point_cloud_world[z][0] = project_final[0][0]
		point_cloud_world[z][1] = project_final[1][0]
		point_cloud_world[z][2] = project_final[2][0]
		
		z = z+1

print(point_cloud)
print(point_cloud_world)
print(project_final)
print('points3D')
points3D = cv2.reprojectImageTo3D(disparity, Q2)
print(points3D)
pcd = o3d.geometry.PointCloud()
#pcd.points = o3d.utility.Vector3dVector(point_cloud_world)
pcd.points = o3d.utility.Vector3dVector(points3D)
o3d.visualization.draw_geometries([pcd])



cv2.waitKey(0)