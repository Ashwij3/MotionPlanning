import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from obstacle_map import ObstacleField

class DHParam(object):
    def __init__(self, theta, r, d, alpha):
        self.theta_ = theta  # Unit [radian]
        self.r_ = r          # Unit [metres]
        self.d_ = d          # Unit [metres]
        self.alpha_ = alpha  # Unit [radian]

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
        self.config_space_obstacles = np.zeros((360,360))


        self.figure, (ax1, ax2) = plt.subplots(2)
        self.figure.set_figheight(50)
        self.figure.set_figwidth(35)    
        self.figure.set_dpi(100)

        self.task_plot = ax1
        self.config_plot = ax2

        self.task_plot.plot(0.0, 0.0, marker = 'o', ms = 25, mfc = [0,0,0], markeredgecolor = [0,0,0], mew = 5) #joint 1
        line1, = ax1.plot([],[],'k-',linewidth=10) #link 1
        line2, = ax1.plot([],[],'k-',linewidth=10) #link 2     
        line3, = ax1.plot([],[], marker = 'o', ms = 15, mfc = [0.7, 0.0, 1, 1], markeredgecolor = [0,0,0], mew = 5) #joint 2
        line4, = ax1.plot([],[], marker = 'o', ms = 15, mfc = [0,0.75,1, 1], markeredgecolor = [0,0,0], mew = 5) #End Effector

        theta_plot, = self.config_plot.plot([],[],marker = 'o', ms = 15, mfc = [0,0.75,1, 1], markeredgecolor = [0,0,0], mew = 5) # theta

        self.line_ = [line1, line2, line3, line4, theta_plot]

        self.__animation_dMat = np.zeros((1, 4), dtype=np.float)
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

        return Ti

    def extractTranslation(self):
        self.EEPosition_[0] = self.T_[0, 3]
        self.EEPosition_[1] = self.T_[1, 3]
        self.EEPosition_[2] = self.T_[2, 3]
        
    
    def FwdKin(self,JointValues):
        self.DHparam_.theta_ = JointValues

        for i in range(len(self.DHparam_.theta_)):
                self.T_ =  self.T_ * self.cal_DH2Fk(i)
        
        self.extractTranslation()
        self.T_ = np.matrix(np.identity(4))

    def InvKin2(self, TargetLoc, clf = 0):
        theta_ = np.zeros(self.theta_.shape)
        theta_[0] = np.arctan2(TargetLoc[2], TargetLoc[0]);

        COS_beta_num = self.DHparam_.r_[1]**2 - self.DHparam_.r_[2]**2 + TargetLoc[0]**2 + TargetLoc[1]**2 
        COS_beta_den = 2 * self.DHparam_.r_[1] * np.sqrt(TargetLoc[0]**2 + TargetLoc[1]**2)

        if clf == 0:
                 theta_[1] = np.arctan2(TargetLoc[1], TargetLoc[0]) - np.arccos(COS_beta_num/COS_beta_den)
        elif clf == 1:
                 theta_[1] = np.arctan2(TargetLoc[1], TargetLoc[0]) + np.arccos(COS_beta_num/COS_beta_den)

        COS_alpha_num = self.DHparam_.r_[1]**2 + self.DHparam_.r_[2]**2 - TargetLoc[0]**2 - TargetLoc[1]**2 
        COS_alpha_den = 2 * self.DHparam_.r_[1] * self.DHparam_.r_[2]

        if clf == 0:
                 theta_[2] = np.pi - np.arccos(COS_alpha_num/COS_alpha_den)
        elif clf == 1:
                 theta_[2] = np.arccos(COS_alpha_num/COS_alpha_den) - np.pi

        return theta_

    def InvKin(self,TargetLoc,clf=0):
        self.EETarget_ = TargetLoc

        self.theta_ = self.InvKin2(TargetLoc,clf)

        self.FwdKin(self.theta_)


    def TrajectoryGen(self,start,Target,TrajectoryType ='linear', steps = 25, obstacles = []):
        if TrajectoryType == 'joint':
            x = []
            y = []
            z = []
            self.InvKin(start,self.defaultcfg)
            start_theta  = self.DHparam_.theta_

            self.InvKin(Target,self.defaultcfg)
            target_theta  = self.DHparam_.theta_

            theta0_dt = np.linspace(start_theta[0], target_theta[0],steps)
            theta1_dt = np.linspace(start_theta[1], target_theta[1],steps)
            theta2_dt = np.linspace(start_theta[2], target_theta[2],steps)

            for i in range(len(theta0_dt)):
                    self.FwdKin([theta0_dt[i], theta1_dt[i], theta2_dt[i]])
                    x.append(self.EEPosition_[0])
                    y.append(self.EEPosition_[1])
                    z.append(self.EEPosition_[2])
        if TrajectoryType == 'linear':
            time = np.linspace(0.0, 1.0, steps)
            x = (1 - time) * start[0] + time * Target[0]
            y = (1 - time) * start[1] + time * Target[1] 
            z = (1 - time) * start[2] + time * Target[2] 
        
        return [x, y, z]

    def CheckTrajectory(self):
        pass
    
    def Animation_Data_Generation(self):
        self.__animation_dMat = np.zeros((len(self.trajectory_[0]), 4), dtype=np.float)
        self.__theta = np.zeros((len(self.trajectory_[0]), len(self.DHparam_.theta_)), dtype=np.float) 
        
        for i in range(len(self.trajectory_[0])):
            self.InvKin([self.trajectory_[0][i], self.trajectory_[1][i], self.trajectory_[2][i]] , self.defaultcfg)
            self.__animation_dMat[i][0] = self.DHparam_.r_[1]*np.cos(self.DHparam_.theta_[1])
            self.__animation_dMat[i][1] = self.DHparam_.r_[1]*np.sin(self.DHparam_.theta_[1])
            self.__animation_dMat[i][2] = self.EEPosition_[0]
            self.__animation_dMat[i][3] = self.EEPosition_[1]
            self.__theta[i][0] = self.DHparam_.theta_[0]
            self.__theta[i][1] = self.DHparam_.theta_[1]
            self.__theta[i][2] = self.DHparam_.theta_[2]

    def InitAnimation(self):
        self.Animation_Data_Generation()
        self.line_[0].set_data([0.0, self.__animation_dMat[0][0]], [0.0, self.__animation_dMat[0][1]])
        self.line_[1].set_data([self.__animation_dMat[0][0], self.__animation_dMat[0][2]], [self.__animation_dMat[0][1], self.__animation_dMat[0][3]])
        self.line_[2].set_data(self.__animation_dMat[0][0], self.__animation_dMat[0][1])
        self.line_[3].set_data(self.__animation_dMat[0][2], self.__animation_dMat[0][3])

        self.line_[4].set_xdata(self.__theta[0][1])
        self.line_[4].set_ydata(self.__theta[0][2])

        self.config_plot.plot(self.__theta[0][1], self.__theta[0][2], label=r'Initial Position: $p_{(x, y)}$', marker = 'o', ms = 30, mfc = [0,0.5,1], markeredgecolor = [0,0,0], mew = 5)
        self.config_plot.plot(self.__theta[len(self.trajectory_[0]) - 1][1], self.__theta[len(self.trajectory_[1]) - 1][2], label=r'Target Position: $p_{(x, y)}$', marker = 'o', ms = 30, mfc = [0,1,0], markeredgecolor = [0,0,0], mew = 5)


        return self.line_        
    def start_animation(self, i):
        self.line_[0].set_data([0.0, self.__animation_dMat[i][0]], [0.0, self.__animation_dMat[i][1]])
        self.line_[1].set_data([self.__animation_dMat[i][0], self.__animation_dMat[i][2]], [self.__animation_dMat[i][1], self.__animation_dMat[i][3]])
        self.line_[2].set_data(self.__animation_dMat[i][0], self.__animation_dMat[i][1])
        self.line_[3].set_data(self.__animation_dMat[i][2], self.__animation_dMat[i][3])
  
        self.line_[4].set_xdata(self.__theta[i][1])
        self.line_[4].set_ydata(self.__theta[i][2])

        return self.line_


    
    def display_environment(self, obstacles, work_envelope = [False, 0]):
        if len(self.trajectory_[0]) > 0:
            
            self.task_plot.plot(self.trajectory_[0][0], self.trajectory_[1][0], label=r'Initial Position: $p_{(x, y)}$', marker = 'o', ms = 30, mfc = [0,0.5,1], markeredgecolor = [0,0,0], mew = 5)
            self.task_plot.plot(self.trajectory_[0][len(self.trajectory_[0]) - 1], self.trajectory_[1][len(self.trajectory_[1]) - 1], label=r'Target Position: $p_{(x, y)}$', marker = 'o', ms = 30, mfc = [0,1,0], markeredgecolor = [0,0,0], mew = 5)

            self.p = [self.trajectory_[0][len(self.trajectory_[0]) - 1], self.trajectory_[1][len(self.trajectory_[1]) - 1]]
            # self.InvKin(self.p,self.defaultcfg)
       
        # Get all of the static obstacles and plot them 
        ob_x = []
        ob_y = []
        t1 = []
        t2 = []
        # add obstacles to task space plot 
        for j in range(0, obstacles.shape[1]):
            for i in range(0, obstacles.shape[0]):
                if obstacles[i][j] > 0:
                    ob_x.append(i/100)
                    ob_y.append(j/100)

        # for every point in configuration space
        for j in range(0, 360):
            for i in range(0, 360):
                # find link/ee positions
                self.FwdKin([0, np.radians(i), np.radians(j)])
                joint_2_pose_x = self.DHparam_.r_[1]*np.cos(self.DHparam_.theta_[1])
                joint_2_pose_y = self.DHparam_.r_[1]*np.sin(self.DHparam_.theta_[1])
                link1x=np.linspace(0.0, joint_2_pose_x, 10)
                link1y=np.linspace(0.0, joint_2_pose_y, 10)

                link2x=np.linspace(joint_2_pose_x, self.EEPosition_[0], 10)
                link2y=np.linspace(joint_2_pose_y, self.EEPosition_[1], 10)

                # if any reference point in the links is an obstacle in task space
                # then this is an obstacle in config space 
                #print(i, j, joint_2_pose_x, joint_2_pose_y, link1x, link1y)
                for k in range(0, len(link1x)):
                    x,y = int(link1x[k]*100) , int(link1y[k]*100)
                    if obstacles[x][y] > 0:
                        self.config_space_obstacles[i][j] = 1
                        t1.append(i)
                        t2.append(j)
                        break; 
                for k in range(0, len(link2x)):
                    x,y = int(link2x[k]*100) , int(link2y[k]*100)
                    if obstacles[x][y] > 0:
                        self.config_space_obstacles[i][j] = 1
                        t1.append(i)
                        t2.append(j)
                        break; 

        self.static_x = ob_x
        self.static_y = ob_y
        self.static_t1 = t1
        self.static_t2 = t2

        #print(len(ob_x),len(ob_y))

        self.task_plot.plot(ob_x, ob_y, 's', marker = 'x', ms = 10, mfc = [0,0,0], markeredgecolor = [0,0,0], mew = 5)
        self.config_plot.plot(t1,t2,'s', marker = 'x', ms = 10, mfc = [0,0,0], markeredgecolor = [0,0,0], mew = 5)
        
        self.task_plot.axis([(-1)*(self.DHparam_.r_[1] + self.DHparam_.r_[2]) - 0.2, (1)*(self.DHparam_.r_[1] + self.DHparam_.r_[2]) + 0.2, (-1)*(self.DHparam_.r_[1] + self.DHparam_.r_[2]) - 0.2, (1)*(self.DHparam_.r_[1] + self.DHparam_.r_[2]) + 0.2])
        self.task_plot.grid()
        
        self.task_plot.set_xlabel('x position [m]', fontsize = 20, fontweight ='normal')
        self.task_plot.set_ylabel('y position [m]', fontsize = 20, fontweight ='normal')
        self.task_plot.set_title(self.RobotName_, fontsize = 50, fontweight ='normal')
        self.task_plot.legend(loc=0,fontsize=20)

        self.config_plot.axis([self.JointLimits_[0][0], self.JointLimits_[0][1], self.JointLimits_[1][0], self.JointLimits_[1][1]])
        self.config_plot.grid()
        self.config_plot.set_xlabel('Theta 1 [rad]', fontsize = 20, fontweight ='normal')
        self.config_plot.set_ylabel('Theta 2 [rad]', fontsize = 20, fontweight ='normal')
        self.config_plot.set_title(self.RobotName_, fontsize = 50, fontweight ='normal')
        


def main():
    obstacles = ObstacleField(int(56*2), int(56*2), 1, 0, 0, 0) # robot is centered at 0,0 which is len/2 +1, len/2 +1 
    obstacles.populate(1)
    #fig = plt.figure()
    #obstacles.print_map(fig,2,3, 1)
    #plt.show()

    JointLimits = [[0, 360],[0, 360]]
    arm_length = [0.30, 0.25]
    
    inittheta = [0.0, 0.0, 0.0]
    r       = [0, arm_length[0], arm_length[1]]
    d       = [0.0, 0.0, 0.0]
    alpha   = [0.0, 0.0, 0.0]

    RRBOT = Robot("RRRobot",DHParam(inittheta,r,d,alpha),JointLimits, obstacles)

    RRBOT.defaultcfg = 1 # 0/1 for changing the configuration to reach a particular point
    TrajectoryType = "linear"
    StartPoint = [0.50, 0.0, 0.0]
    TargetPoint = [0.10, 0.30, 0.0]
    steps = 25

    x,y, z = RRBOT.TrajectoryGen(StartPoint,TargetPoint,TrajectoryType,steps, obstacles.get_map())
    for j in range(steps):
        RRBOT.trajectory_[0].append(x[j])
        RRBOT.trajectory_[1].append(y[j])
        RRBOT.trajectory_[2].append(z[j])
    RRBOT.display_environment(obstacles.get_map(), [True, 1], )
    animator = animation.FuncAnimation(RRBOT.figure, RRBOT.start_animation, init_func=RRBOT.InitAnimation, frames=len(RRBOT.trajectory_[0]), interval=2, blit=True, repeat=False)
    animator.save('test.mp4', fps=30, bitrate=1000, dpi=100)



if __name__=="__main__":
    main()
