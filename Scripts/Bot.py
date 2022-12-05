import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from obstacle_map import ObstacleField
from decimal import * 
from enum import Enum
import fcl
from concurrent.futures import ThreadPoolExecutor

class JointType(Enum):
    REVOLUTE = 1,
    PRISMATIC = 2

class DHParam(object):
    def __init__(self, theta, r, d, alpha, types):
        self.theta_ = theta  # Unit [radian]
        self.r_ = r          # Unit [metres]
        self.d_ = d          # Unit [metres]
        self.alpha_ = alpha  # Unit [radian]
        self.types = types    # Unit [revolute or prismatic]

    def joints(self):
        return len(self.theta_)

class Robot(object):
    def __init__(self,RobotName:str, DHparam:DHParam, JointLimits, ObstacleMap):
        self.RobotName_ = RobotName
        self.DHparam_ = DHparam
        self.JointLimits_ = JointLimits 
        self.T_ = np.matrix(np.identity(4))
        self.EEPosition_ = np.zeros(3)
        self.theta_ = np.zeros(DHparam.joints())
        self.defaultcfg = 0
        self.trajectory_ = [[], [], []]
        self.task_space_obstacles = ObstacleMap
        self.config_space_obstacles = np.zeros((360,360,10,360))

        obstacle_cos = []
        # add obstacles to task space plot 
        for k in range(0, ObstacleMap.shape[2]):
            for j in range(0, ObstacleMap.shape[1]):
                for i in range(0, ObstacleMap.shape[0]):
                    if ObstacleMap[i][j][k] > 0:
                        box = fcl.Box(1,1,1)
                        box_transform = fcl.Transform(np.identity(3), [i, j, k])
                        obj = fcl.CollisionObject(box, box_transform)
                        obstacle_cos.append(obj)
        
        self.obstacle_manager = fcl.DynamicAABBTreeCollisionManager()
        self.obstacle_manager.registerObjects(obstacle_cos)
        self.obstacle_manager.setup()
        
        link_1_transform = fcl.Transform(np.identity(3), [0,0,0]) 
        link_2_transform = fcl.Transform(np.identity(3), [0,0,0])
        self.link_1 = fcl.Cylinder(0.05, 0.3) 
        self.link_2 = fcl.Cylinder(0.05, 0.25)
        self.link_1_co = fcl.CollisionObject(self.link_1, link_1_transform)
        self.link_2_co = fcl.CollisionObject(self.link_2, link_2_transform)
        arm = [self.link_1_co, self.link_2_co]

        self.arm_manager = fcl.DynamicAABBTreeCollisionManager()
        self.arm_manager.registerObjects(arm)
        self.arm_manager.setup()

        self.__theta = np.zeros((1,3), dtype=np.float)
    
    def cal_DH2Fk(self,JointIndex:int):
        Ti = np.matrix(np.identity(4))
        
        Ti[0,0] = np.cos(self.DHparam_.theta_[JointIndex])
        Ti[0,1] = -np.sin(self.DHparam_.theta_[JointIndex]) * np.cos(self.DHparam_.alpha_[JointIndex])
        Ti[0,2] = np.sin(self.DHparam_.theta_[JointIndex]) * np.sin(self.DHparam_.alpha_[JointIndex])
        Ti[0,3] = self.DHparam_.r_[JointIndex] * np.cos(self.DHparam_.theta_[JointIndex])

        Ti[1, 0] = np.sin(self.DHparam_.theta_[JointIndex])
        Ti[1, 1] = np.cos(self.DHparam_.theta_[JointIndex]) * np.cos(self.DHparam_.alpha_[JointIndex])
        Ti[1, 2] = -np.cos(self.DHparam_.theta_[JointIndex]) * np.sin(self.DHparam_.alpha_[JointIndex])
        Ti[1, 3] = self.DHparam_.r_[JointIndex] * np.sin(self.DHparam_.theta_[JointIndex])

        Ti[2, 0] = 0
        Ti[2, 1] = np.sin(self.DHparam_.alpha_[JointIndex])
        Ti[2, 2] = np.cos(self.DHparam_.alpha_[JointIndex])
        Ti[2, 3] = self.DHparam_.d_[JointIndex]

        Ti[3, 0] = 0
        Ti[3, 1] = 0
        Ti[3, 2] = 0
        Ti[3, 3] = 1

        #print(Ti)
        return Ti

    def extractTranslation(self):
        self.EEPosition_[0] = self.T_[0, 3]
        self.EEPosition_[1] = self.T_[1, 3]
        self.EEPosition_[2] = self.T_[2, 3]
        
    def FwdKin(self,JointValues):
        self.DHparam_.theta_ = JointValues
        for i in range(len(self.DHparam_.theta_)):
                if self.DHparam_.types[i] == JointType.PRISMATIC:
                    self.DHparam_.r_[i] = self.DHparam_.theta_[i]
                    self.DHparam_.theta_[i] = 0
                t = self.cal_DH2Fk(i)
                self.T_ =  self.T_ * t
        
        self.extractTranslation()
        self.T_ = np.matrix(np.identity(4))

    def InvKin2(self, TargetLoc, clf = 0):
        print(TargetLoc)
        theta_ = np.zeros(self.theta_.shape)
        theta_[0] = np.arctan2(TargetLoc[2], TargetLoc[0]);

        COS_beta_num = round(self.DHparam_.r_[1]**2 - self.DHparam_.r_[2]**2 + TargetLoc[0]**2 + TargetLoc[1]**2, 4) 
        COS_beta_den = round(2 * self.DHparam_.r_[1] * np.sqrt(TargetLoc[0]**2 + TargetLoc[1]**2),4)
        

        if clf == 0:
                 theta_[1] = np.arctan2(TargetLoc[1], TargetLoc[0]) - np.arccos(COS_beta_num/COS_beta_den)
        elif clf == 1:
                 theta_[1] = np.arctan2(TargetLoc[1], TargetLoc[0]) + np.arccos(COS_beta_num/COS_beta_den)

        COS_alpha_num = round(self.DHparam_.r_[1]**2 + self.DHparam_.r_[2]**2 - TargetLoc[0]**2 - TargetLoc[1]**2, 4)
        COS_alpha_den = round(2 * self.DHparam_.r_[1] * self.DHparam_.r_[2], 4)

        if clf == 0:
                 theta_[2] = np.pi - np.arccos(COS_alpha_num/COS_alpha_den)
        elif clf == 1:
                 theta_[2] = np.arccos(COS_alpha_num/COS_alpha_den) - np.pi
      
        return theta_

    def InvKin(self,TargetLoc,clf=0):
        self.EETarget_ = TargetLoc

        self.theta_ = self.InvKin2(TargetLoc,clf)

        self.FwdKin(self.theta_)

    def check_collision_at_pose(self, thetas):
        #print(thetas)
        self.DHparam_.theta_ = [np.radians(thetas[0]), np.radians(thetas[1]), (10 + thetas[2])/100, np.radians(thetas[3])]
        for m in range(len(self.DHparam_.theta_)-1):
            if self.DHparam_.types[m] == JointType.PRISMATIC:
                self.DHparam_.r_[m] = self.DHparam_.theta_[m]
                self.DHparam_.theta_[m] = 0
            t = self.cal_DH2Fk(m)
            self.T_ =  self.T_ * t
        J1Pose = [0, 0, 0]
        J1Pose[0] = self.T_[0, 3]
        J1Pose[1] = self.T_[1, 3]
        J1Pose[2] = self.T_[2, 3]
        link_1_R = self.T_[0:3,0:3]
        self.T_ = np.matrix(np.identity(4))

        self.FwdKin([np.radians(thetas[0]), np.radians(thetas[1]), (10 + thetas[2])/100, np.radians(thetas[3])])
        link_2_R = self.T_[0:3,0:3]
        self.link_1_co.setRotation(link_1_R)
        self.link_2_co.setTranslation(J1Pose)
        self.link_2_co.setRotation(link_2_R)
        req = fcl.CollisionRequest(num_max_contacts = 100, enable_contact =True)
        collisions  = fcl.CollisionData(request = req)
        
        self.arm_manager.collide(self.obstacle_manager, collisions, fcl.defaultCollisionCallback)
        if collisions.result.contacts:
            return True
        else:
            return False

def main():
    obstacles = ObstacleField(int(76*2), int(76*2), int(76*2), 0, 0, 0) # robot is centered at 0,0 which is len/2 +1, len/2 +1 
    obstacles.populate(1)
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

    RRBOT.defaultcfg = 1 # 0/1 for changing the configuration to reach a particular point
    TrajectoryType = "linear"
    StartPoint = [0.50, 0.0, 0.0]
    TargetPoint = [0.10, 0.30, 0.0]
    steps = 25

    start = [1,0,0,0]
    end = [10,0,0,0]

    print(RRBOT.planner.planning(start,end))



if __name__=="__main__":
    main()
