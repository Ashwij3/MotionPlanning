import sys
import math
import time

import numpy as np



class Map:
    def __init__(self):
        self.q1_minlimit = 0
        self.q2_minlimit = 0
        self.q3_minlimit = 0
        self.q4_minlimit = 0

        self.q1_maxlimit = 5
        self.q2_maxlimit = 5
        self.q3_maxlimit = 5
        self.q4_maxlimit = 5
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


class RRT:

    def __init__(self,
                 map: Map,
                 step=5,
                 goal_sample_rate=5,
                 max_iter=1000000,
                 path_resolution=1):
        
        self.map = map

        self.step = step
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = []
        self.path_resolution = path_resolution

    def planning(self,start,goal):
        self.start = Node(start[0],start[1],start[2],start[3])
        self.goal = Node(goal[0],goal[1],goal[2],goal[3])

        self.node_list = [self.start]
        
        for i in range(self.max_iter):
            rnd_node = self.get_rndNode()
            nearest_node = self.node_list[self.get_nearNodeIdx(rnd_node)]
            
            new_node =  self.steer(nearest_node, rnd_node, self.step)

            if not self.check_if_outside_play_area(new_node):
                if self.check_collision(new_node):
                    self.node_list.append(new_node)
            
            if self.calc_dist_to_goal(self.node_list[-1]) <= self.step:
                final_node = self.steer(self.node_list[-1], self.goal)
                if not self.check_if_outside_play_area(final_node):
                    if self.check_collision(final_node):
                        return self.generate_final_course(len(self.node_list) - 1) 
    
    def get_rndNode(self):
        if np.random.randint(0, 100) > self.goal_sample_rate:
            rnd = Node(
                np.random.uniform(self.map.q1_minlimit, self.map.q1_maxlimit),
                np.random.uniform(self.map.q2_minlimit, self.map.q2_maxlimit),
                np.random.uniform(self.map.q3_minlimit, self.map.q3_maxlimit),
                np.random.uniform(self.map.q4_minlimit, self.map.q4_maxlimit))
        else:  # goal point sampling
            rnd = Node(
                self.goal.q1,
                self.goal.q2,
                self.goal.q2,
                self.goal.q3)
        
        return rnd        

    def get_nearNodeIdx(self, rnd_node:Node):  ## NEEDS change for wrapped space
        dlist = [(node.q1 - rnd_node.q2)**2 + (node.q2 - rnd_node.q2)**2 + (node.q3 - rnd_node.q3)**2 + (node.q4 - rnd_node.q4)**2
                 for node in self.node_list]
        idx = dlist.index(min(dlist))
        return idx
        

        return self.node_list[idx]

    def steer(self, from_node:Node, to_node:Node, extend_length=float("inf")):

        new_node = Node(from_node.q1,
                        from_node.q2,
                        from_node.q3,
                        from_node.q4)
                             
        new_node.path_q1 = [new_node.q1]
        new_node.path_q2 = [new_node.q2]
        new_node.path_q3 = [new_node.q3]
        new_node.path_q4 = [new_node.q4]

        d = self.calc_euclideanDistance(new_node, to_node)

        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)

        for _ in range(n_expand):
            new_node.q1 += (to_node.q1 - from_node.q1) / d * extend_length*self.path_resolution
            new_node.q2 += (to_node.q2 - from_node.q2) / d * extend_length*self.path_resolution
            new_node.q3 += (to_node.q3 - from_node.q3) / d * extend_length*self.path_resolution
            new_node.q4 += (to_node.q4 - from_node.q4) / d * extend_length*self.path_resolution
            new_node.path_q1.append(new_node.q1)
            new_node.path_q2.append(new_node.q2)
            new_node.path_q3.append(new_node.q3)
            new_node.path_q4.append(new_node.q4)
        
        new_node.parent = from_node

        return new_node
        



    def calc_euclideanDistance(self, from_node:Node, to_node:Node):
        dq1 = to_node.q1 - from_node.q1
        dq2 = to_node.q2 - from_node.q2
        dq3 = to_node.q3 - from_node.q3
        dq4 = to_node.q4 - from_node.q4
        
        d = math.hypot(dq1, dq2, dq3, dq4)
        return d


    def calc_dist_to_goal(self, node:Node):
        dq1 = node.q1 - self.goal.q1
        dq2 = node.q2 - self.goal.q2
        dq3 = node.q3 - self.goal.q3
        dq4 = node.q4 - self.goal.q4
        return math.hypot(dq1, dq2, dq3, dq4)

    def check_collision(self,node:Node):
        if node is None:
            return False
        
        for i in range(len(node.path_q1)):
            if (self.map.grid[node.path_q1[i],node.path_q2[i],node.path_q3[i],node.path_q4[i]]):
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
    map = Map()
    planner = RRT(map = map)

    start = [1,0,0,0]
    end = [8,0,0,0]
    # print(planner.map.grid[0,0,0,100])
    print(planner.planning(start,end))


if __name__ == '__main__':
    main()

