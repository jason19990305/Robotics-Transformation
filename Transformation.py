import cv2
import numpy as np
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from IPython.display import display, clear_output
import time


class Homogeneous():
    def __init__(self) -> None:
        
        self.H = np.identity(4)
        self.alpha = 0
        self.beta = 0
        self.gamma = 0
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0
    def set_r_x(self,theta):
        self.alpha = theta / 180 * math.pi
    def set_r_y(self,theta):
        self.beta = theta / 180 * math.pi
    def set_r_z(self,theta):
        self.gamma = theta / 180 * math.pi
    def set_pos_x(self,x):
        self.pos_x = x
    def set_pos_y(self,y):
        self.pos_y = y
    def set_pos_z(self,z):
        self.pos_z = z
    def update(self):  
        cos_alpha = math.cos(self.alpha)
        sin_alpha = math.sin(self.alpha)
        cos_beta = math.cos(self.beta)
        sin_beta = math.sin(self.beta)
        cos_gamma = math.cos(self.gamma)
        sin_gamma = math.sin(self.gamma)
        
        self.H[0][0] = cos_beta * cos_gamma
        self.H[0][1] = sin_alpha * sin_beta * cos_gamma - cos_alpha * sin_gamma
        self.H[0][2] = cos_alpha * sin_beta * cos_gamma + sin_alpha * sin_gamma
        self.H[1][0] = cos_beta * sin_gamma
        self.H[1][1] = sin_alpha * sin_beta * sin_gamma + cos_alpha * cos_gamma
        self.H[1][2] = cos_alpha * sin_beta * sin_gamma - sin_alpha * cos_gamma
        self.H[2][0] = -sin_beta
        self.H[2][1] = sin_alpha * cos_beta
        self.H[2][2] = cos_alpha * cos_beta

        self.H[0][3] = self.pos_x
        self.H[1][3] = self.pos_y
        self.H[2][3] = self.pos_z




class Transformation():
    def __init__(self) -> None:

        self.T1 = Homogeneous()
        self.T2 = Homogeneous()

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self.init_plot()

        img = np.zeros([480,400,1])

        cv2.imshow('transition',img)
        # A
        cv2.createTrackbar('A R x', 'transition', 0, 180, self.A_r_x_bar)
        cv2.createTrackbar('A R y', 'transition', 0, 180, self.A_r_y_bar)
        cv2.createTrackbar('A R z', 'transition', 0, 180, self.A_r_z_bar)

        cv2.createTrackbar('A p x', 'transition', 0, 100, self.A_p_x_bar)
        cv2.createTrackbar('A p y', 'transition', 0, 100, self.A_p_y_bar)
        cv2.createTrackbar('A p z', 'transition', 0, 100, self.A_p_z_bar)

        cv2.setTrackbarPos('A R x', 'transition', 0)
        cv2.setTrackbarPos('A R y', 'transition', 0)
        cv2.setTrackbarPos('A R z', 'transition', 0)

        cv2.setTrackbarPos('A p x', 'transition', 0)
        cv2.setTrackbarPos('A p y', 'transition', 0)
        cv2.setTrackbarPos('A p z', 'transition', 0)

        # B
        cv2.createTrackbar('B R x', 'transition', 0, 180, self.B_r_x_bar)
        cv2.createTrackbar('B R y', 'transition', 0, 180, self.B_r_y_bar)
        cv2.createTrackbar('B R z', 'transition', 0, 180, self.B_r_z_bar)

        cv2.createTrackbar('B p x', 'transition', 0, 100, self.B_p_x_bar)
        cv2.createTrackbar('B p y', 'transition', 0, 100, self.B_p_y_bar)
        cv2.createTrackbar('B p z', 'transition', 0, 100, self.B_p_z_bar)

        cv2.setTrackbarPos('B R x', 'transition', 0)
        cv2.setTrackbarPos('B R y', 'transition', 0)
        cv2.setTrackbarPos('B R z', 'transition', 0)

        cv2.setTrackbarPos('B p x', 'transition', 0)
        cv2.setTrackbarPos('B p y', 'transition', 0)
        cv2.setTrackbarPos('B p z', 'transition', 0)
        while True:
            self.plot()
            if cv2.waitKey(1) == ord('q'):
                break
            self.T1.update()
            self.T2.update()

        cv2.destroyAllWindows()
    def A_r_x_bar(self,val):
        self.T1.set_r_x(val)
    def A_r_y_bar(self,val):
        self.T1.set_r_y(val)
    def A_r_z_bar(self,val):
        self.T1.set_r_z(val)
    def A_p_x_bar(self,val):
        self.T1.set_pos_x(val * 0.1)
    def A_p_y_bar(self,val):
        self.T1.set_pos_y(val * 0.1)
    def A_p_z_bar(self,val):
        self.T1.set_pos_z(val * 0.1)

    def B_r_x_bar(self,val):
        self.T2.set_r_x(val)
    def B_r_y_bar(self,val):
        self.T2.set_r_y(val)
    def B_r_z_bar(self,val):
        self.T2.set_r_z(val)
    def B_p_x_bar(self,val):
        self.T2.set_pos_x(val * 0.1)
    def B_p_y_bar(self,val):
        self.T2.set_pos_y(val * 0.1)
    def B_p_z_bar(self,val):
        self.T2.set_pos_z(val * 0.1)
        
    def init_plot(self):
        origin = np.array([0,0,0]).reshape(1,3)
        T1_x = self.T1.H[0:3 , 0].reshape(1,3)
        T1_y = self.T1.H[0:3 , 1].reshape(1,3)
        T1_z = self.T1.H[0:3 , 2].reshape(1,3)
        T1_p = self.T1.H[0:3 , 3].reshape(1,3)
        T1_x = np.concatenate((T1_p , T1_p + T1_x),axis=0)
        T1_y = np.concatenate((T1_p , T1_p + T1_y),axis=0)
        T1_z = np.concatenate((T1_p , T1_p + T1_z),axis=0)
        T1_p = np.concatenate((origin , T1_p),axis=0)

        T2_x = self.T1.H[0:3 , 0].reshape(1,3)
        T2_y = self.T1.H[0:3 , 1].reshape(1,3)
        T2_z = self.T1.H[0:3 , 2].reshape(1,3)
        T2_p = self.T1.H[0:3 , 3].reshape(1,3)
        T2_x = np.concatenate((T2_p , T2_p + T2_x),axis=0)
        T2_y = np.concatenate((T2_p , T2_p + T2_y),axis=0)
        T2_z = np.concatenate((T2_p , T2_p + T2_z),axis=0)
        T2_p = np.concatenate((origin , T2_p),axis=0)

        self.line1, = self.ax.plot(T1_x[:,0],T1_x[:,1],T1_x[:,2],color='blue', label='A-X')
        self.line2, = self.ax.plot(T1_y[:,0],T1_y[:,1],T1_y[:,2],color='green', label='A-Y')
        self.line3, = self.ax.plot(T1_z[:,0],T1_z[:,1],T1_z[:,2],color='red', label='A-Z')
        self.line4, = self.ax.plot(T1_p[:,0],T1_p[:,1],T1_p[:,2],color='black', label='A-Z')


        self.line5, = self.ax.plot(T2_x[:,0],T2_x[:,1],T2_x[:,2],color='blue', label='B-X')
        self.line6, = self.ax.plot(T2_y[:,0],T2_y[:,1],T2_y[:,2],color='green', label='B-Y')
        self.line7, = self.ax.plot(T2_z[:,0],T2_z[:,1],T2_z[:,2],color='red', label='B-Z')
        self.line8, = self.ax.plot(T2_p[:,0],T2_p[:,1],T2_p[:,2],color='yellow', label='B-Z')


    def plot(self):
        
        self.T2.H = self.T1.H.dot(self.T2.H)

        origin = np.array([0,0,0]).reshape(1,3)
        T1_x = self.T1.H[0:3 , 0].reshape(1,3)
        T1_y = self.T1.H[0:3 , 1].reshape(1,3)
        T1_z = self.T1.H[0:3 , 2].reshape(1,3)
        T1_p = self.T1.H[0:3 , 3].reshape(1,3)
        A_p = T1_p

        T1_x = np.concatenate((T1_p , T1_p + T1_x),axis=0)
        T1_y = np.concatenate((T1_p , T1_p + T1_y),axis=0)
        T1_z = np.concatenate((T1_p , T1_p + T1_z),axis=0)
        T1_p = np.concatenate((origin , T1_p),axis=0)

        T2_x = self.T2.H[0:3 , 0].reshape(1,3)
        T2_y = self.T2.H[0:3 , 1].reshape(1,3)
        T2_z = self.T2.H[0:3 , 2].reshape(1,3)
        T2_p = self.T2.H[0:3 , 3].reshape(1,3)

        T2_x = np.concatenate((T2_p , T2_p + T2_x),axis=0)
        T2_y = np.concatenate((T2_p , T2_p + T2_y),axis=0)
        T2_z = np.concatenate((T2_p , T2_p + T2_z),axis=0)
        T2_p = np.concatenate((A_p , T2_p),axis=0)
        B_p = T2_p



        self.line1.set_xdata(T1_x[:,0])
        self.line1.set_ydata(T1_x[:,1])
        self.line1.set_3d_properties(T1_x[:,2])

        self.line2.set_xdata(T1_y[:,0])
        self.line2.set_ydata(T1_y[:,1])
        self.line2.set_3d_properties(T1_y[:,2])

        self.line3.set_xdata(T1_z[:,0])
        self.line3.set_ydata(T1_z[:,1])
        self.line3.set_3d_properties(T1_z[:,2])

        self.line4.set_xdata(T1_p[:,0])
        self.line4.set_ydata(T1_p[:,1])
        self.line4.set_3d_properties(T1_p[:,2])
        # B

        self.line5.set_xdata(T2_x[:,0])
        self.line5.set_ydata(T2_x[:,1])
        self.line5.set_3d_properties(T2_x[:,2])

        self.line6.set_xdata(T2_y[:,0])
        self.line6.set_ydata(T2_y[:,1])
        self.line6.set_3d_properties(T2_y[:,2])

        self.line7.set_xdata(T2_z[:,0])
        self.line7.set_ydata(T2_z[:,1])
        self.line7.set_3d_properties(T2_z[:,2])

        self.line8.set_xdata(T2_p[:,0])
        self.line8.set_ydata(T2_p[:,1])
        self.line8.set_3d_properties(T2_p[:,2])

        self.ax.legend()

        plt.draw()
        
        plt.pause(0.3)


if __name__ == '__main__':
    transition = Transformation()