import sys
import math
import time

from obstacle_map import ObstacleField
from bot import *
import numpy as np
import matplotlib.animation
import matplotlib.pyplot as plt


class Map:
    def __init__(self):
        self.q1_minlimit = 0
        self.q2_minlimit = 0
        self.q3_minlimit = 0
        self.q4_minlimit = 0

        self.q1_maxlimit = 360
        self.q2_maxlimit = 180
        self.q3_maxlimit = 20
        self.q4_maxlimit = 180
        # self.grid = np.random.randint(2,
        #                            size =(self.q1_maxlimit - self.q1_minlimit,
        #                             self.q2_maxlimit - self.q2_minlimit,
        #                             self.q3_maxlimit - self.q3_minlimit,
        #                             self.q4_maxlimit - self.q4_minlimit
        #                            ))
        self.grid = np.zeros( shape =(self.q1_maxlimit - self.q1_minlimit,
                                    self.q2_maxlimit - self.q2_minlimit,
                                    self.q3_maxlimit - self.q3_minlimit,
                                    self.q4_maxlimit - self.q4_minlimit)
                            )
        

class Node:
    def __init__(self, q1, q2, q3, q4):
            self.q1 = q1
            self.q2 = q2
            self.q3 = q3
            self.q4 = q4
            self.path_q1 = []
            self.path_q2 = []
            self.path_q3 = []
            self.path_q4 = []
            self.parent = None

    def __repr__(self):
        return str([self.q1, self.q2, self.q3, self.q4])

    def __eq__(self, other):
        return self.q1 == other.q1 and self.q2 == other.q2 and self.q3 == other.q3 and self.q4 == other.q4

    def __hash__(self):
        return hash(str(self.q1)) + hash(str(self.q2)) +hash(str(self.q3)) +hash(str(self.q4)) 

class RRT:
    def __init__(self,
                 map_: Map,
                 robot,
                 step=5,
                 goal_sample_rate=25,
                 max_iter=1000000,
                 path_resolution=1):
        
        self.map = map_
        self.robot = robot
        
        self.step = step
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []
        self.path_resolution = path_resolution
        self.delta = 5

    def planning(self,start_pose,goal_pose):
        start = self.robot.InvKin2(start_pose)
        goal = self.robot.InvKin2(goal_pose)
        self.start = Node(start[0],start[1],start[2]*100,start[3])
        self.goal = Node(goal[0],goal[1],goal[2]*100,goal[3])

        self.node_list = [self.start]
        print("start", self.start, "end", self.goal)
        time.sleep(2.0)
        
        for i in range(self.max_iter):
            rnd_node = self.get_rndNode()
            print("rnd", rnd_node)
            nearest_node = self.node_list[self.get_nearNodeIdx(rnd_node)]
            print("nearest", nearest_node)

            new_node =  self.steer(nearest_node, rnd_node, self.delta, self.step)
            print("new", new_node)
            if new_node in self.node_list:
                print("already visited", new_node)
                continue
            
            if not self.check_if_outside_play_area(new_node):
                if self.check_collision(new_node):
                    self.node_list.append(new_node)
            print("nodes", len(self.node_list))
            if self.calc_dist_to_goal(self.node_list[-1]) <= self.step:
                final_node = self.steer(self.node_list[-1], self.goal, self.delta, self.step)
                if not self.check_if_outside_play_area(final_node):
                    if self.check_collision(final_node):
                        return self.generate_final_course(len(self.node_list) - 1) 
            #time.sleep(2.0)
    
    def get_rndNode(self):
        if np.random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(
                np.random.randint(self.map.q1_minlimit, self.map.q1_maxlimit),
                np.random.randint(self.map.q2_minlimit, self.map.q2_maxlimit),
                np.random.randint(self.map.q3_minlimit, self.map.q3_maxlimit),
                np.random.randint(self.map.q4_minlimit, self.map.q4_maxlimit))
        else:  # goal point sampling
            rnd = Node(
                self.goal.q1,
                self.goal.q2,
                self.goal.q3,
                self.goal.q4)
        
        return rnd        

    def get_nearNodeIdx(self, rnd_node:Node):
        dlist = []
        for idx, node in enumerate(self.node_list):
            q1_diff_1 = (node.q1 - rnd_node.q1)
            q2_diff_1 = (node.q2 - rnd_node.q2)
            q3_diff_1 = (node.q3 - rnd_node.q3)
            q4_diff_1 = (node.q4 - rnd_node.q4)
            
            q1_diff_2 = (node.q1 - (rnd_node.q1-360))
            q2_diff_2 = (node.q2 - (rnd_node.q2-360))
            q3_diff_2 = (node.q3 - (rnd_node.q3-360))
            q4_diff_2 = (node.q4 - (rnd_node.q4-360))

            q1_diff = min(abs(q1_diff_1), abs(q1_diff_2))
            q2_diff = min(abs(q2_diff_1), abs(q2_diff_2))
            q3_diff = min(abs(q3_diff_1), abs(q3_diff_2))
            q4_diff = min(abs(q4_diff_1), abs(q4_diff_2))

            dlist.append((q1_diff)**2 + (q2_diff)**2 + (q3_diff)**2 + (q4_diff)**2)
        min_value = min(dlist)
        print(min_value)                          
        idx = dlist.index(min_value)
        return idx

    def steer(self, from_node:Node, to_node:Node, delta, step=float("inf")):

        q1_diff = to_node.q1 -from_node.q1 
        q2_diff = to_node.q2 - from_node.q2 
        q3_diff = to_node.q3 -from_node.q3
        q4_diff = to_node.q4 - from_node.q4

        #print("diffs", [q1_diff, q2_diff, q3_diff, q4_diff])
        q1_diff = q1_diff if q1_diff < delta else delta
        q2_diff = q2_diff if q2_diff < delta else delta
        q3_diff = q3_diff if q3_diff < delta else delta
        q4_diff = q4_diff if q4_diff < delta else delta

        q1_dt = q1_diff / step
        q2_dt = q2_diff / step
        q3_dt = q3_diff / step
        q4_dt = q4_diff / step
        
        new_node = Node(from_node.q1,
                        from_node.q2,
                        from_node.q3,
                        from_node.q4)
                             
        new_node.path_q1 = [new_node.q1]
        new_node.path_q2 = [new_node.q2]
        new_node.path_q3 = [new_node.q3]
        new_node.path_q4 = [new_node.q4]

        for _ in range(step):
            new_node.q1 = int(new_node.q1 + q1_dt)
            new_node.q2 = int(new_node.q2 + q2_dt)
            new_node.q3 = int(new_node.q3 + q3_dt)
            new_node.q4 = int(new_node.q4 + q4_dt)
            new_node.path_q1.append(new_node.q1)
            new_node.path_q2.append(new_node.q2)
            new_node.path_q3.append(new_node.q3)
            new_node.path_q4.append(new_node.q4)

            #print("path i", _, new_node)
        
        new_node.parent = from_node

        return new_node
        



    def calc_euclideanDistance(self, from_node:Node, to_node:Node):
        dq1 = to_node.q1 - from_node.q1
        dq2 = to_node.q2 - from_node.q2
        dq3 = to_node.q3 - from_node.q3
        dq4 = to_node.q4 - from_node.q4
        
        d = math.sqrt(dq1**2+dq2**2+dq3**2+dq4**2)
        return d


    def calc_dist_to_goal(self, node:Node):
        dq1 = node.q1 - self.goal.q1
        dq2 = node.q2 - self.goal.q2
        dq3 = node.q3 - self.goal.q3
        dq4 = node.q4 - self.goal.q4
        return math.sqrt(dq1**2+dq2**2+dq3**2+dq4**2)

    def check_collision(self,node:Node):
        if node is None:
            return False
        
        for i in range(len(node.path_q1)):
            collisions = self.robot.check_collision_at_pose([node.path_q1[i], node.path_q2[i], node.path_q3[i]/100, node.path_q4[i]])
            if collisions:
                return False
        else:
            return True

    def check_if_outside_play_area(self,node:Node):

        if node.q1 < 0 or node.q1 > (self.map.q1_maxlimit - self.map.q1_minlimit)-1 or \
           node.q2 < 0 or node.q2 > (self.map.q2_maxlimit - self.map.q2_minlimit)-1 or \
           node.q3 < 0 or node.q3 > (self.map.q3_maxlimit - self.map.q3_minlimit)-1 or \
           node.q4 < 0 or node.q4 > (self.map.q4_maxlimit - self.map.q4_minlimit)-1:

            return True  # outside - bad
        else:
            return False  # inside - ok

    def generate_final_course(self, goal_ind):
        path = [[self.goal.q1, self.goal.q2, self.goal.q3, self.goal.q4]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.q1, node.q2, node.q3, node.q4])
            node = node.parent
        path.append([node.q1, node.q2, node.q3, node.q4])

        return path        




def main():

    obstacles = ObstacleField(int(75*2)+1, int(75*2)+1, int(75*2)+1, 0, 0, 0) # robot is centered at 0,0 which is len/2 +1, len/2 +1 
    obstacles.populate(20)
    #fig = plt.figure()
    #obstacles.print_map(fig,2,3, 1)
    #plt.show()

    JointLimits = [[0, 259],[0, 259],[0.1,0.2],[0, 259]]
    arm_length = [0.30, 0.25]
    
    inittheta = [0.0,            0.0,           0.0,  0.0]
    r       =   [0.0,            arm_length[0], 0.10, arm_length[1]]
    d       =   [0.0,            0.0,           0.0,  0.0]
    alpha   =   [math.pi/2.0,    0.0,           0.0,  0.0        ]
    types   =   [JointType.REVOLUTE, JointType.REVOLUTE, JointType.PRISMATIC, JointType.REVOLUTE]

    RRBOT = Robot("RRRobot",DHParam(inittheta,r,d,alpha,types),JointLimits, obstacles.get_map())
    map_ = Map()
    planner = RRT(map_ = map_, robot =RRBOT)

    start = [0.75*np.cos(np.radians(45)), 0.75*np.sin(np.radians(45)), 0.0]
    end =   [0,0,0.65]

    path = planner.planning(start,end)

    RRBOT.plot_configuration(RRBOT.InvKin2(start))

    ani = matplotlib.animation.FuncAnimation(RRBOT.figure, RRBOT.update, fargs=(path,), frames=len(path),  )
    writervideo = animation.FFMpegWriter(fps=60)

    ani.save('arm_plan.gif', writer=writervideo)
    plt.show()

if __name__ == '__main__':
    main()
