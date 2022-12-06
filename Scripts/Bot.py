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
        
        theta_ = np.zeros(self.theta_.shape)
        theta_[0] = np.arctan2(TargetLoc[1], TargetLoc[0])

        rot = np.matrix([[np.cos(-theta_[0]), -np.sin(-theta_[0]), 0],[np.sin(-theta_[0]), np.cos(-theta_[0]), 0],[0,0,1]])
        print(rot)
        rotated_ee = np.transpose(TargetLoc*np.transpose(rot))
        print("ee", TargetLoc, "rot_ee", rotated_ee)
        rho =  0 if rotated_ee[0] >= 0 else math.pi
        if not rotated_ee[0] == 0:
            link_3_pos = [rotated_ee[0] - self.DHparam_.r_[3]*np.cos(rho), 0, rotated_ee[2] - self.DHparam_.r_[3]*np.sin(rho)]
        else:
            link_3_pos = [rotated_ee[0] - self.DHparam_.r_[3]*np.sin(rho), 0, rotated_ee[2] - self.DHparam_.r_[3]*np.cos(rho)]
        print("l3", link_3_pos)
        theta_[2] = math.sqrt(link_3_pos[0]**2 + link_3_pos[2]**2 ) -  self.DHparam_.r_[1]

        theta_[1] = np.arctan2(link_3_pos[2], link_3_pos[0])

        theta_[3] = rho  - theta_[1] if not rotated_ee[0]==0 else theta_[1] - rho - math.pi/2
        print(TargetLoc, theta_)
        return [np.degrees(theta_[0]), np.degrees(theta_[1]), theta_[2], np.degrees(theta_[3])]

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

    def update(self, i, data):
        thetas = data[i]
        self.FwdKin([np.radians(thetas[0]), np.radians(thetas[1]), (.1 + thetas[2]/100), np.radians(thetas[3])])
        jp = np.zeros((4,3), dtype=np.float64)
        for i in range(len(self.DHparam_.theta_)):
            t = self.cal_DH2Fk(i)
            self.T_ =  self.T_ * t
            if i > 0:
                jp[i-1][0] = float(self.T_[0, 3])
                jp[i-1][1] = float(self.T_[1, 3])
                jp[i-1][2] = float(self.T_[2, 3])

        self.line1.set_data_3d([0.0, jp[0][0]], [0.0, jp[0][1]], [0.0, jp[0][2]])
        self.line2.set_data_3d([jp[0][0], jp[1][0]], [jp[0][1], jp[1][1]],[jp[0][2], jp[1][2]])
        self.line3.set_data_3d([jp[1][0], jp[2][0]], [jp[1][1], jp[2][1]],[jp[1][2], jp[2][2]])

        self.line4.set_data_3d(jp[0][0], jp[0][1],jp[0][2])
        self.line5.set_data_3d(jp[1][0], jp[1][1],jp[1][2])
        self.line6.set_data_3d(jp[2][0], jp[2][1],jp[2][2])

        self.theta_plot.set_data_3d(thetas[0], thetas[1], thetas[3])
        
        return [self.line1,self. line2, self.line3, self.line4, self.line5, self.line6, self.theta_plot]

    def plot_configuration(self, thetas):

        print("theta", thetas)
        self.FwdKin([np.radians(thetas[0]), np.radians(thetas[1]), (.1 + thetas[2]), np.radians(thetas[3])])
        self.figure = plt.figure()
        ax1 = self.figure.add_subplot(1,2,1,projection='3d')
        ax2 = self.figure.add_subplot(1,2,2,projection='3d')
        self.figure.set_figheight(50)
        self.figure.set_figwidth(35)    
        self.figure.set_dpi(100)

        self.task_plot = ax1
        self.config_plot = ax2

        self.task_plot.plot(0.0, 0.0,0.0, marker = 'o', ms = 25, mfc = [0,0,0], markeredgecolor = [0,0,0], mew = 5) #joint 1
        self.line1, = ax1.plot([],[],[],'k-',linewidth=10) #link 1
        self.line2, = ax1.plot([],[],[],'k-',linewidth=10) #link 2
        self.line3, = ax1.plot([],[],[],'k-',linewidth=10) #link 3

        self.line4, = ax1.plot([],[],[], marker = 'o', ms = 15, mfc = [0.7, 0.0, 1, 1], markeredgecolor = [0,0,0], mew = 5) #joint 2
        self.line5, = ax1.plot([],[],[], marker = 'o', ms = 15, mfc = [0.0, 1, 0.75, 1], markeredgecolor = [0,0,0], mew = 5) #joint 3
        self.line6, = ax1.plot([],[],[], marker = 'o', ms = 15, mfc = [0,0.75,1, 1], markeredgecolor = [0,0,0], mew = 5) #End Effector

        ax1.axes.set_xlim3d(left=-0.60, right=0.75) 
        ax1.axes.set_ylim3d(bottom=-0.6, top=0.75) 
        ax1.axes.set_zlim3d(bottom=-0.6, top=0.75) 

        ax2.axes.set_xlim3d(left=0, right=360) 
        ax2.axes.set_ylim3d(bottom=0, top=360) 
        ax2.axes.set_zlim3d(bottom=0, top=360) 

        self.theta_plot, = self.config_plot.plot([],[],marker = 'o', ms = 15, mfc = [0,0.75,1, 1], markeredgecolor = [0,0,0], mew = 5) # theta

        ob_x =[]
        ob_y = []
        ob_z = []
        x,y,z = self.task_space_obstacles.shape
        for k in range(0, z):
            for j in range(0,y):
                for i in range(0,x):
                    if self.task_space_obstacles[i][j][k] > 0:
                        ob_x.append((i-(x/2))/100)
                        ob_y.append((j-(y/2))/100)
                        ob_z.append((k-(z/2))/100)
        print("obs", len(ob_x))
        self.task_plot.scatter(ob_x, ob_y, ob_z,  marker="s",)

        jp = np.zeros((4,3), dtype=np.float64)
        for i in range(len(self.DHparam_.theta_)):
            t = self.cal_DH2Fk(i)
            self.T_ =  self.T_ * t
            if i > 0:
                jp[i-1][0] = float(self.T_[0, 3])
                jp[i-1][1] = float(self.T_[1, 3])
                jp[i-1][2] = float(self.T_[2, 3])
        
        print(jp)
        print(self.EEPosition_)

        self.line1.set_data_3d([0.0, jp[0][0]], [0.0, jp[0][1]], [0.0, jp[0][2]])
        self.line2.set_data_3d([jp[0][0], jp[1][0]], [jp[0][1], jp[1][1]],[jp[0][2], jp[1][2]])
        self.line3.set_data_3d([jp[1][0], jp[2][0]], [jp[1][1], jp[2][1]],[jp[1][2], jp[2][2]])
        
        self.line4.set_data_3d(jp[0][0], jp[0][1],jp[0][2])
        self.line5.set_data_3d(jp[1][0], jp[1][1],jp[1][2])
        self.line6.set_data_3d(jp[2][0], jp[2][1],jp[2][2])

        #self.__theta[0] = thetas
        self.theta_plot.set_data_3d(thetas[0], thetas[1], thetas[3])

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
