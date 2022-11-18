import numpy as np 
import random 
import matplotlib.pyplot as plt
from enum import Enum

class ObstacleField:
	# initalization of map with a given size 
	# and a goal, so we dont' make the goal an obstacle 
	def __init__(self, x, y, z, goal_x, goal_y, goal_z):
		self.goal_y = goal_y
		self.goal_x = goal_x
		self.goal_z = goal_z
		self.x=x
		self.y=y
		self.z=z 
		self.obstacle_map_list = []
		self.obstacle_map=np.zeros((self.x,self.y, self.z))

	def get_map(self):
		return self.obstacle_map

	def get_list(self):
		return self.obstacle_map_list

    # populates the map the given percent of map spaces with an obstacle 
	def populate(self, percent):
		spots_to_fill = int(((self.x * self.y) * (percent/100)))
 
		spots_filled = self.get_spots_filled()
		# While the percentage of the map filled is less than the requested amount
		while spots_filled < spots_to_fill:
			# add a random tile 
			self.add_object()
			spots_filled = self.get_spots_filled()

		return 

	# returns true if the space at the x,y location does not have an obstacle 
	def is_empty(x, y, z):
		return self.obstacle_map[x][y][z] == 0

	# adds a random object to a random location in the map 
	def add_object(self):
		# get random x/y/theta 
		x_place = random.randint(0,self.x-1)
		y_place = random.randint(0,self.y-1)
		z_place =  random.randint(0, self.z-1) if self.z > 1  else  0

		# don't allow the goal to be an obstacle 
		if x_place == self.goal_x and y_place == self.goal_y:
			return 


		self.obstacle_map_list.append((x_place, y_place))

		theta = random.randint(0,3) # 0=0 1=90 2=180 3=270
		mirror = random.randint(0,1) 
		if mirror == 0:
			mirror = -1 
		
		# pick random shape
		tetrominoe_type=random.randint(0,3)

		#print(x_place, y_place, theta)
		# put it in the map 
		# straight line 
		if tetrominoe_type == 0:
			if theta == 0:
				for i in range(0,4):
					if x_place + i < self.x:
						self.obstacle_map[x_place + i][y_place][z_place] = 1
			elif theta == 1:
				for i in range(0,4):
					if y_place - i >= 0:
						self.obstacle_map[x_place][y_place - i][z_place] = 1
			elif theta == 2:
				for i in range(0,4):
					if x_place - i >= 0:
						self.obstacle_map[x_place - i][y_place][z_place] = 1
			elif theta == 3:
				for i in range(0,4):
					if y_place + i < self.y:
						self.obstacle_map[x_place][y_place + i][z_place] = 1
			else:
				print("invalid theta")
		# L shape 
		elif tetrominoe_type == 1:
			if theta == 0:
				for i in range(0,3):
					if x_place + i < self.x:
						self.obstacle_map[x_place + i][y_place][z_place] = 1
				if  0 <= y_place + (1 * mirror) < self.y:
						self.obstacle_map[x_place][y_place + (1*mirror)][z_place] = 1
			elif theta == 1:
				for i in range(0,3):
					if y_place - i >= 0:
						self.obstacle_map[x_place][y_place - i][z_place] = 1
				if 0 <= x_place + (1 * mirror) < self.x:
						self.obstacle_map[x_place + (1*mirror)][y_place][z_place] = 1
			elif theta == 2:
				for i in range(0,3):
					if x_place - i >= 0:
						self.obstacle_map[x_place - i][y_place] = 1
				if  0 <= y_place + (1 * mirror) < self.y:
						self.obstacle_map[x_place][y_place + (1*mirror)][z_place] = 1
			elif theta == 3:
				for i in range(0,3):
					if y_place + i < self.y:
						self.obstacle_map[x_place][y_place + i] = 1
				if 0 <= x_place + (1 * mirror) < self.x:
					self.obstacle_map[x_place + (1*mirror)][y_place][z_place] = 1
			else:
				print("invalid theta")
		# Z shape 
		elif tetrominoe_type == 2:
			if theta == 0:
				for i in range(0,2):
					if x_place + i < self.x:
						self.obstacle_map[x_place + i][y_place][z_place] = 1
				x_place=x_place+1
				y_place=y_place+1
				for i in range(0,2):
					if 0 <= x_place + i < self.x and 0 <= y_place < self.y:
						self.obstacle_map[x_place + i][y_place][z_place] = 1
			elif theta == 1:
				for i in range(0,2):
					if y_place - i >= 0:
						self.obstacle_map[x_place][y_place - i][z_place] = 1
				x_place=x_place+1
				y_place=y_place+1
				for i in range(0,2):
					if 0 <=  y_place - i < self.y and 0 <= x_place < self.x:
						self.obstacle_map[x_place][y_place - i][z_place] = 1
			elif theta == 2:
				for i in range(0,2):
					if x_place - i >= 0:
						self.obstacle_map[x_place - i][y_place][z_place] = 1
				x_place=x_place+1
				y_place=y_place+1
				for i in range(0,2):
					if 0 <=  x_place - i < self.x  and 0 <= y_place < self.y:
						self.obstacle_map[x_place-i][y_place ][z_place] = 1
			elif theta == 3:
				for i in range(0,2):
					if y_place + i < self.y:
						self.obstacle_map[x_place][y_place + i][z_place]= 1
				x_place=x_place+1
				y_place=y_place+1
				for i in range(0,2):
					if 0 <=  y_place + i < self.y and 0 <= x_place < self.x:
						self.obstacle_map[x_place][y_place + i][z_place]= 1
			else:
				pritn("invalid theta")
		# T shape 
		elif tetrominoe_type == 3:
			if theta == 0:
				for i in range(0,3):
					if x_place + i < self.x:
						self.obstacle_map[x_place + i][y_place][z_place] = 1
				if  0 <= y_place + (1 * mirror) < self.y and  0 <= x_place + 1 < self.x:
						self.obstacle_map[x_place + 1][y_place + (1*mirror)][z_place] = 1
			elif theta == 1:
				for i in range(0,3):
					if y_place - i >= 0:
						self.obstacle_map[x_place][y_place - i][z_place] = 1
				if  0 <= y_place - 1 < self.y and  0 <= x_place + (1 * mirror) < self.x:
						self.obstacle_map[x_place + (1*mirror)][y_place -1 ][z_place] = 1
			elif theta == 2:
				for i in range(0,3):
					if x_place - i >= 0:
						self.obstacle_map[x_place - i][y_place][z_place] = 1
				if  0 <= y_place + (1 * mirror) < self.y and  0 <= x_place - 1  < self.x:
						self.obstacle_map[x_place - 1][y_place + (1*mirror)][z_place]= 1
			elif theta == 3:
				for i in range(0,3):
					if y_place + i < self.y:
						self.obstacle_map[x_place][y_place + i][z_place] = 1
				if  0 <= y_place + 1  < self.y and  0 <= x_place + (1 * mirror) < self.x:
						self.obstacle_map[x_place + (1*mirror)][y_place + 1][z_place] = 1
			else:
				print("invalid theta")

		else:
			print("invalid type")

	# returns the number of places in the map that contain an obstacle 		
	def get_spots_filled(self):
		return np.sum(self.obstacle_map)

    # prints the map as a matplotlib image 
	def print_map(self, fig, row, column, loc):
		fig.add_subplot(row, column, loc)
		plt.imshow(self.obstacle_map, interpolation='none')
