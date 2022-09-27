import numpy as np
import matplotlib.pyplot as plt

class DHParam:
    def __init__(self, theta, r, d, alpha):
        self.theta_ = theta  # Unit [radian]
        self.r_ = r          # Unit [metres]
        self.d_ = d          # Unit [metres]
        self.alpha_ = alpha  # Unit [radian]

class Robot(object):
    def __init__(self,RobotName:str, DHparam:DHParam, JointLimits):
        self.RobotName_ = RobotName
        self.DHparam_ = DHparam
        self.JointLimits_ = JointLimits 
        self.T_ = np.matrix(np.identity(4))
        self.EEPosition_ = np.zeros(2)
        self.theta = np.zeros(2)

        plt.plot(0.0, 0.0, 
            label=r'Joint 1: $\theta_1 ('+ str(self.ax_wr[0] * (180/np.pi)) +','+ str(self.ax_wr[1] * (180/np.pi)) +')$', 
            marker = 'o', ms = 25, mfc = [0,0,0], markeredgecolor = [0,0,0], mew = 5
        )


    
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
        self.EEPosition_[0] = self.T[0, 3]
        self.EEPosition_[1] = self.T[1, 3]
        

    
    def FwdKin(self,JointValues):
        self.thetaTarget_ = np.zeros(2)
        self.thetaTarget_[0] = JointValues[0]
        self.thetaTarget_[1] = JointValues[1]

        self.DHparam_.theta_ = self.thetaTarget_

        for i in range(len(self.DHparam_.theta_)):
                self.T_ =  self.T_ * self.cal_DH2Fk(i)
        
        self.extractTranslation()
        self.T_ = np.matrix(np.identity(4))

    
    def InvKin(self,TargetLoc,cfg):
        self.EETarget_ = np.zeros(2)
        self.EETarget_[0] = TargetLoc[0]
        self.EETarget_[1] = TargetLoc[1]
        

        COS_beta_num = self.DHparam_.r_[0]**2 - self.DHparam_.r_[1]**2 + self.EETarget_[0]**2 + self.EETarget_[1]**2 
        COS_beta_den = 2 * self.DHparam_.r_[0] * np.sqrt(self.EETarget_[0]**2 + self.EETarget_[1]**2)
        
        if cfg == 0:
                 self.theta[0] = np.arctan2(self.EETarget_[1], self.EETarget_[0]) - np.arccos(COS_beta_num/COS_beta_den)
        elif cfg == 1:
                 self.theta[0] = np.arctan2(self.EETarget_[1], self.EETarget_[0]) + np.arccos(COS_beta_num/COS_beta_den)
        
        COS_alpha_num = self.DHparam_.r_[0]**2 + self.DHparam_.r_[1]**2 - self.EETarget_[0]**2 - self.EETarget_[1]**2 
        COS_alpha_den = 2 * self.DHparam_.r_[0] * self.DHparam_.r_[1]
        
        if cfg == 0:
                 self.theta[1] = np.pi - np.arccos(COS_alpha_num/COS_alpha_den)
        elif cfg == 1:
                 self.theta[1] = np.arccos(COS_alpha_num/COS_alpha_den) - np.pi
        
        self.FwdKin(self.theta)


    def TrajectoryGen(self,start,Target,steps = 25):
        # x = []
        # y = []
        # self.InvKin(start)
        # start_theta  = self.DHparam_.theta_

        # self.InvKin(Target)
        # target_theta  = self.DHparam_.theta_

        # theta1_dt = np.linspace(start_theta[0], target_theta[0],steps)
        # theta2_dt = np.linspace(start_theta[1], target_theta[1],steps)

        # for i in range(len(theta1_dt)):
        #         self.FwdKin([theta1_dt[i], theta2_dt[i]])
        #         x.append(self.EEPosition_[0])
        #         y.append(self.EEPosition_[1])
       
        time = np.linspace(0.0, 1.0, steps)
        x = (1 - time) * start[0] + time * Target[0]
        y = (1 - time) * start[1] + time * Target[1]
        
        return [x, y]





def main():
    JointLimits = [[-2.44346, 2.44346],[-2.61799, 2.61799]]
    arm_length = [0.3, 0.25]
    
    inittheta = [0.0,0.0]
    r       = [arm_length[0], arm_length[1]]
    d       = [0.0, 0.0]
    alpha   = [0.0, 0.0]

    RRBOT = Robot("RRRobot",DHParam(inittheta,r,d,alpha),JointLimits)

    StartPoint = []
    TargetPoint = []

    RRBOT.TrajectoryGen(StartPoint,TargetPoint)











