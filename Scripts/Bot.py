import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

class DHParam(object):
    def __init__(self, theta, r, d, alpha):
        self.theta_ = theta  # Unit [radian]
        self.r_ = r          # Unit [metres]
        self.d_ = d          # Unit [metres]
        self.alpha_ = alpha  # Unit [radian]

    def joints(self):
        return len(self.theta_)

class Robot(object):
    def __init__(self,RobotName:str, DHparam:DHParam, JointLimits):
        self.RobotName_ = RobotName
        self.DHparam_ = DHparam
        self.JointLimits_ = JointLimits 
        self.T_ = np.matrix(np.identity(4))
        self.EEPosition_ = np.zeros(3)
        self.theta_ = np.zeros(DHparam.joints())
        self.defaultcfg = 0
        self.trajectory_ = [[], [], []]
        self.figure = plt.figure(num=None, figsize=(25, 17.5), dpi=80, facecolor='w', edgecolor='k')

        plt.plot(0.0, 0.0, marker = 'o', ms = 25, mfc = [0,0,0], markeredgecolor = [0,0,0], mew = 5) #joint 1
        line1, = plt.plot([],[],'k-',linewidth=10) #link 1
        line2, = plt.plot([],[],'k-',linewidth=10) #link 2     
        line3, = plt.plot([],[], marker = 'o', ms = 15, mfc = [0.7, 0.0, 1, 1], markeredgecolor = [0,0,0], mew = 5) #joint 2
        line4, = plt.plot([],[], marker = 'o', ms = 15, mfc = [0,0.75,1, 1], markeredgecolor = [0,0,0], mew = 5) #End Effector
        self.line_ = [line1, line2, line3, line4]
    
        self.__animation_dMat = np.zeros((1, 4), dtype=np.float)

    
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

    
    def InvKin(self,TargetLoc,clf=0):
        self.EETarget_ = TargetLoc

        # initial twist is based on x/z 
        self.theta_[0] = np.arctan2(self.EETarget_[0], self.EETarget_[2]);

        COS_beta_num = self.DHparam_.r_[1]**2 - self.DHparam_.r_[2]**2 + self.EETarget_[0]**2 + self.EETarget_[1]**2 
        COS_beta_den = 2 * self.DHparam_.r_[1] * np.sqrt(self.EETarget_[0]**2 + self.EETarget_[1]**2)
        
        if clf == 0:
                 self.theta_[1] = np.arctan2(self.EETarget_[1], self.EETarget_[0]) - np.arccos(COS_beta_num/COS_beta_den)
        elif clf == 1:
                 self.theta_[1] = np.arctan2(self.EETarget_[1], self.EETarget_[0]) + np.arccos(COS_beta_num/COS_beta_den)
        
        COS_alpha_num = self.DHparam_.r_[1]**2 + self.DHparam_.r_[2]**2 - self.EETarget_[0]**2 - self.EETarget_[1]**2 
        COS_alpha_den = 2 * self.DHparam_.r_[1] * self.DHparam_.r_[2]
        
        if clf == 0:
                 self.theta_[2] = np.pi - np.arccos(COS_alpha_num/COS_alpha_den)
        elif clf == 1:
                 self.theta_[2] = np.arccos(COS_alpha_num/COS_alpha_den) - np.pi
        
        self.FwdKin(self.theta_)


    def TrajectoryGen(self,start,Target,TrajectoryType ='linear', steps = 25):
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
        
        for i in range(len(self.trajectory_[0])):
            self.InvKin([self.trajectory_[0][i], self.trajectory_[1][i], self.trajectory_[2][i]] ,self.defaultcfg)
            self.__animation_dMat[i][0] = self.DHparam_.r_[1]*np.cos(self.DHparam_.theta_[1])
            self.__animation_dMat[i][1] = self.DHparam_.r_[1]*np.sin(self.DHparam_.theta_[1])
            self.__animation_dMat[i][2] = self.EEPosition_[0]
            self.__animation_dMat[i][3] = self.EEPosition_[1]

    def InitAnimation(self):
       
        self.Animation_Data_Generation()
        self.line_[0].set_data([0.0, self.__animation_dMat[0][0]], [0.0, self.__animation_dMat[0][1]])
        self.line_[1].set_data([self.__animation_dMat[0][0], self.__animation_dMat[0][2]], [self.__animation_dMat[0][1], self.__animation_dMat[0][3]])
        self.line_[2].set_data(self.__animation_dMat[0][0], self.__animation_dMat[0][1])
        self.line_[3].set_data(self.__animation_dMat[0][2], self.__animation_dMat[0][3])
        
        return self.line_        
    def start_animation(self, i):

        self.line_[0].set_data([0.0, self.__animation_dMat[i][0]], [0.0, self.__animation_dMat[i][1]])
        self.line_[1].set_data([self.__animation_dMat[i][0], self.__animation_dMat[i][2]], [self.__animation_dMat[i][1], self.__animation_dMat[i][3]])
        self.line_[2].set_data(self.__animation_dMat[i][0], self.__animation_dMat[i][1])
        self.line_[3].set_data(self.__animation_dMat[i][2], self.__animation_dMat[i][3])

        return self.line_

    
    def display_environment(self, work_envelope = [False, 0]):
        if len(self.trajectory_[0]) > 0:
            
            plt.plot(self.trajectory_[0][0], self.trajectory_[1][0], label=r'Initial Position: $p_{(x, y)}$', marker = 'o', ms = 30, mfc = [0,0.5,1], markeredgecolor = [0,0,0], mew = 5)
            plt.plot(self.trajectory_[0][len(self.trajectory_[0]) - 1], self.trajectory_[1][len(self.trajectory_[1]) - 1], label=r'Target Position: $p_{(x, y)}$', marker = 'o', ms = 30, mfc = [0,1,0], markeredgecolor = [0,0,0], mew = 5)

            self.p = [self.trajectory_[0][len(self.trajectory_[0]) - 1], self.trajectory_[1][len(self.trajectory_[1]) - 1]]
            # self.InvKin(self.p,self.defaultcfg)
       
    
        plt.axis([(-1)*(self.DHparam_.r_[1] + self.DHparam_.r_[2]) - 0.2, (1)*(self.DHparam_.r_[1] + self.DHparam_.r_[2]) + 0.2, (-1)*(self.DHparam_.r_[1] + self.DHparam_.r_[2]) - 0.2, (1)*(self.DHparam_.r_[1] + self.DHparam_.r_[2]) + 0.2])
        plt.grid()
        plt.xlabel('x position [m]', fontsize = 20, fontweight ='normal')
        plt.ylabel('y position [m]', fontsize = 20, fontweight ='normal')
        plt.title(self.RobotName_, fontsize = 50, fontweight ='normal')
        plt.legend(loc=0,fontsize=20)


def main():
    JointLimits = [[-2.44346, 2.44346],[-2.61799, 2.61799]]
    arm_length = [0.3, 0.25]
    
    inittheta = [0.0, 0.0, 0.0]
    r       = [0, arm_length[0], arm_length[1]]
    d       = [0.0, 0.0, 0.0]
    alpha   = [0.0, 0.0, 0.0]

    RRBOT = Robot("RRRobot",DHParam(inittheta,r,d,alpha),JointLimits)

    RRBOT.defaultcfg = 1 # 0/1 for changing the configuration to reach a particular point
    TrajectoryType = "linear"
    StartPoint = [0.50, 0.0, 0.0]
    TargetPoint = [0.10, 0.30, 0.0]
    steps = 25

    x,y, z = RRBOT.TrajectoryGen(StartPoint,TargetPoint,TrajectoryType,steps)
    for j in range(steps):
        RRBOT.trajectory_[0].append(x[j])
        RRBOT.trajectory_[1].append(y[j])
        RRBOT.trajectory_[2].append(z[j])
    RRBOT.display_environment([True, 1])
    animator = animation.FuncAnimation(RRBOT.figure, RRBOT.start_animation, init_func=RRBOT.InitAnimation, frames=len(RRBOT.trajectory_[0]), interval=2, blit=True, repeat=False)
    animator.save('test.mp4', fps=30, bitrate=1000)


if __name__=="__main__":
    main()
