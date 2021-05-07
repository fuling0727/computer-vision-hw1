import sys
import cv2
import numpy as np
from numpy import asarray
from PyQt5.QtGui import QPixmap, QIcon, QFont, QImage
from PyQt5.QtWidgets import QMainWindow, QApplication, QLabel, QPushButton, QWidget, QLineEdit
from PyQt5 import QtCore
from skimage.filters import median
from skimage.morphology import disk
import matplotlib.pyplot as plt
import imutils

class MainWindow(QMainWindow):
    def __init__(self):
        
        super(MainWindow, self).__init__()
        self.resize(1300,600) # smart phone size
        self.title = "2020 Opencvdl HW1"
        self.setWindowTitle(self.title)

        #第一題
        label1 = QLabel(self)
        label1.setText("1. Image Processing")
        label1.setFixedWidth(200)
        label1.setFixedHeight(35)
        label1.setStyleSheet("color:#3C3C3C	")
        myFont=QFont("Arial Font",11,QFont.Bold)
        label1.setFont(myFont)
        label1.move(50,50)

        button1_1 = QPushButton(self)
        button1_1.setText("1.1 Load Image")  # 建立名字
        button1_1.setStyleSheet("background-color:#FCFCFC; ")
        button1_1.setFixedHeight(35)
        button1_1.setFixedWidth(150)
        button1_1.move(60,120)  # 移動位置
        button1_1.clicked.connect(self.buttonClicked1_1) # 設置button啟動function

        button1_2 = QPushButton(self)
        button1_2.setText("1.2 Color Seperation")  # 建立名字
        button1_2.setStyleSheet("background-color:#FCFCFC; ")
        button1_2.setFixedHeight(35)
        button1_2.setFixedWidth(150)
        button1_2.move(60,200)  # 移動位置
        button1_2.clicked.connect(self.buttonClicked1_2) # 設置button啟動function

        button1_3 = QPushButton(self)
        button1_3.setText("1.3 Image Flipping")  # 建立名字
        button1_3.setStyleSheet("background-color:#FCFCFC; ")
        button1_3.setFixedHeight(35)
        button1_3.setFixedWidth(150)
        button1_3.move(60,280)  # 移動位置
        button1_3.clicked.connect(self.buttonClicked1_3) # 設置button啟動function

        button1_4 = QPushButton(self)
        button1_4.setText("1.4 Blending")  # 建立名字
        button1_4.setStyleSheet("background-color:#FCFCFC; ")
        button1_4.setFixedHeight(35)
        button1_4.setFixedWidth(150)
        button1_4.move(60,360)  # 移動位置
        button1_4.clicked.connect(self.buttonClicked1_4) # 設置button啟動function
    
        #第二題
        label2 = QLabel(self)
        label2.setText("2. Image Smoothing")
        label2.setFixedWidth(200)
        label2.setFixedHeight(35)
        label2.setStyleSheet("color:#3C3C3C	")
        myFont=QFont("Arial Font",11,QFont.Bold)
        label2.setFont(myFont)
        label2.move(350,50)

        button2_1 = QPushButton(self)
        button2_1.setText("2.1 Median Filter")  # 建立名字
        button2_1.setStyleSheet("background-color:#FCFCFC; ")
        button2_1.setFixedHeight(35)
        button2_1.setFixedWidth(150)
        button2_1.move(360,120)  # 移動位置
        button2_1.clicked.connect(self.buttonClicked2_1) # 設置button啟動function

        button2_2 = QPushButton(self)
        button2_2.setText("2.2 Gaussian Blur")  # 建立名字
        button2_2.setStyleSheet("background-color:#FCFCFC; ")
        button2_2.setFixedHeight(35)
        button2_2.setFixedWidth(150)
        button2_2.move(360,200)  # 移動位置
        button2_2.clicked.connect(self.buttonClicked2_2) # 設置button啟動function

        button3_3 = QPushButton(self)
        button3_3.setText("2.3 Bilateral Filter")  # 建立名字
        button3_3.setStyleSheet("background-color:#FCFCFC; ")
        button3_3.setFixedHeight(35)
        button3_3.setFixedWidth(150)
        button3_3.move(360,280)  # 移動位置
        button3_3.clicked.connect(self.buttonClicked2_3) # 設置button啟動function
        #第三題
        label3 = QLabel(self)
        label3.setText("3. Edge Detection")
        label3.setFixedWidth(200)
        label3.setFixedHeight(35)
        label3.setStyleSheet("color:#3C3C3C	")
        myFont=QFont("Arial Font",11,QFont.Bold)
        label3.setFont(myFont)
        label3.move(650,50)

        button3_1 = QPushButton(self)
        button3_1.setText("3.1 Gaussian Blur")  # 建立名字
        button3_1.setStyleSheet("background-color:#FCFCFC; ")
        button3_1.setFixedHeight(35)
        button3_1.setFixedWidth(150)
        button3_1.move(660,120)  # 移動位置
        button3_1.clicked.connect(self.buttonClicked3_1) # 設置button啟動function

        button3_2 = QPushButton(self)
        button3_2.setText("3.2 Sobel X")  # 建立名字
        button3_2.setStyleSheet("background-color:#FCFCFC; ")
        button3_2.setFixedHeight(35)
        button3_2.setFixedWidth(150)
        button3_2.move(660,200)  # 移動位置
        button3_2.clicked.connect(self.buttonClicked3_2) # 設置button啟動function

        button3_3 = QPushButton(self)
        button3_3.setText("3.3 Sobel Y")  # 建立名字
        button3_3.setStyleSheet("background-color:#FCFCFC; ")
        button3_3.setFixedHeight(35)
        button3_3.setFixedWidth(150)
        button3_3.move(660,280)  # 移動位置
        button3_3.clicked.connect(self.buttonClicked3_3) # 設置button啟動function
        
        button3_4 = QPushButton(self)
        button3_4.setText("3.4 Magnitude")  # 建立名字
        button3_4.setStyleSheet("background-color:#FCFCFC; ")
        button3_4.setFixedHeight(35)
        button3_4.setFixedWidth(150)
        button3_4.move(660,360)  # 移動位置
        button3_4.clicked.connect(self.buttonClicked3_4) # 設置button啟動function

        #第四題
        label4 = QLabel(self)
        label4.setText("4. Transformation")
        label4.setFixedWidth(200)
        label4.setFixedHeight(35)
        label4.setStyleSheet("color:#3C3C3C	")
        myFont=QFont("Arial Font",11,QFont.Bold)
        label4.setFont(myFont)
        label4.move(950,50)

        label4_1 = QLabel(self)
        label4_1.setText("Rotation:")
        label4_1.setFixedWidth(60)
        label4_1.setFixedHeight(35)
        label4_1.move(950,120)
        self.line1 = QLineEdit(self)
        self.line1.move(1020,120)
        self.line1.setFixedWidth(170)
        self.line1.setFixedHeight(35)
        label4_1_1 = QLabel(self)
        label4_1_1.setText("deg")
        label4_1_1.setFixedWidth(60)
        label4_1_1.setFixedHeight(35)
        label4_1_1.move(1200,120)
        
        label4_2 = QLabel(self)
        label4_2.setText("Scaling:")
        label4_2.setFixedWidth(60)
        label4_2.setFixedHeight(35)
        label4_2.move(950,200)
        self.line2 = QLineEdit(self)
        self.line2.move(1020,200)
        self.line2.setFixedWidth(170)
        self.line2.setFixedHeight(35)

        label4_3 = QLabel(self)
        label4_3.setText("Tx:")
        label4_3.setFixedWidth(60)
        label4_3.setFixedHeight(35)
        label4_3.move(950,280)
        self.line3 = QLineEdit(self)
        self.line3.move(1020,280)
        self.line3.setFixedWidth(170)
        self.line3.setFixedHeight(35)
        label4_3_1 = QLabel(self)
        label4_3_1.setText("pixel")
        label4_3_1.setFixedWidth(60)
        label4_3_1.setFixedHeight(35)
        label4_3_1.move(1200,280)

        label4_4 = QLabel(self)
        label4_4.setText("Ty:")
        label4_4.setFixedWidth(60)
        label4_4.setFixedHeight(35)
        label4_4.move(950,360)
        self.line4 = QLineEdit(self)
        self.line4.move(1020,360)
        self.line4.setFixedWidth(170)
        self.line4.setFixedHeight(35)
        label4_4_1 = QLabel(self)
        label4_4_1.setText("pixel")
        label4_4_1.setFixedWidth(60)
        label4_4_1.setFixedHeight(35)
        label4_4_1.move(1200,360)

        button4 = QPushButton(self)
        button4.setText("4. Transformation")  # 建立名字
        button4.setStyleSheet("background-color:#FCFCFC; ")
        button4.setFixedHeight(35)
        button4.setFixedWidth(170)
        button4.move(1020,440)  # 移動位置
        button4.clicked.connect(self.buttonClicked4) # 設置button啟動function

        

    def buttonClicked1_1(self): # 顯示圖片
        print('test1')
        img1_1 = cv2.imread("Uncle_Roger.jpg")
        cv2.imshow("1.1 Load Image",img1_1)
        height, width = img1_1.shape[:2]
        print("Height = ",height)
        print("Width = ",width)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    def buttonClicked1_2(self): #RGB 
        print('test2')
        img = cv2.imread("Flower.jpg")
        cv2.imshow("1.2 Color Seperation",img)
        zero = np.zeros(img.shape[:2], dtype='uint8') 
        (B0, G0, R0) = cv2.split(img)
        
        B = cv2.merge([B0,zero,zero])
        cv2.imshow("Blue",B)
        G = cv2.merge([zero,G0,zero])
        cv2.imshow("Green",G)
        R = cv2.merge([zero,zero,R0])
        cv2.imshow("Red",R)

    def buttonClicked1_3(self): #flipping
        print('test3')
        img = cv2.imread("Uncle_Roger.jpg")
        cv2.imshow("1.3 Image Flipping",img)

        img1 = cv2.flip(img,1)
        cv2.imshow("Result",img1)

    a = 1
    b = 0
    def ChangeWeight(x,y):
        
        global a,b
        img = cv2.imread("Uncle_Roger.jpg")
        img1 = cv2.flip(img,1)

        a = cv2.getTrackbarPos('weight','Result')
        print(a)
        a = a*0.01
        b = 1 - a
        print(b)
        result = cv2.addWeighted(img, a, img1, b, 0.0)
        cv2.imshow("Result",result)
        
    def buttonClicked1_4(self): # blending
        print('test4')
        #cv2.namedWindow('Result')
        img = cv2.imread("Uncle_Roger.jpg")
        cv2.imshow("Result",img)
        cv2.createTrackbar('weight', 'Result', 0, 100, self.ChangeWeight)
    
    def buttonClicked2_1(self): #Median
        print("2_1")
        img = cv2.imread("Cat.png")
        img1 = cv2.medianBlur(img, 7)
        #img2 = median(img, disk(3), mode='constant', cval=0.0)
        
        cv2.imshow("Median Filter",img1)
        #cv2.imshow("Median Filter2",img2)

    def buttonClicked2_2(self): #Guassian
        print("2_2")
        img = cv2.imread("Cat.png")
        img1 = cv2.GaussianBlur(img,(3,3),0)
        cv2.imshow("Guassian",img1)

    def buttonClicked2_3(self): #Bilateral
        print("2_3")
        img = cv2.imread("Cat.png")
        img1 = cv2.bilateralFilter(img, 9,90,90)
        cv2.imshow("Bilateral",img1)

    def buttonClicked3_1(self):
        print("3_1")
        img = cv2.imread("Chihiro.jpg")
        img_g = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #cv2.imshow("Gaussian",img_g)
        a = np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
        b = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
        
        g2 = np.exp(-(a**2+b**2))
        g2 = g2/g2.sum()
        print(g2)
        result = cv2.filter2D(img_g,-1,g2)
        print(result)
        cv2.imshow("Gaussian1",result)
        #cv2.imwrite("gaussian.jpg",result)

    def buttonClicked3_2(self):
        print("3_2")
        #img = cv2.imread("D:\\florrie\\four1\\vision\\Hw1\\Dataset_opencvdl\\Q3_Image\\Chihiro.jpg")
        gaussian = cv2.imread("gaussian.jpg")
        
        kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        sobelX = cv2.filter2D(gaussian,-1,kernel)
        #print(kernel)
        #cv2.imwrite("D:\\florrie\\four1\\vision\\Hw1\\SobelX.jpg",sobelX)
        cv2.imshow("Sobel X",sobelX)

    def buttonClicked3_3(self):
        print("3_3")
        #img = cv2.imread("D:\\florrie\\four1\\vision\\Hw1\\Dataset_opencvdl\\Q3_Image\\Chihiro.jpg")
        gaussian = cv2.imread("gaussian.jpg")

        kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        #kernel2 = (255*(kernel - np.min(kernel))/np.ptp(kernel)).astype(int)
        sobelY = cv2.filter2D(gaussian,-1,kernel)
        print(kernel)
        #cv2.imwrite("D:\\florrie\\four1\\vision\\Hw1\\SobelY.jpg",sobelY)
        cv2.imshow("Sobel Y",sobelY)

    def buttonClicked3_4(self):
        print("3_4")
        gaussian = cv2.imread("gaussian.jpg")
        sobelX = cv2.imread("SobelX.jpg")
        sobelY = cv2.imread("SobelY.jpg")
        height, width = gaussian.shape[:2]
         
        x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
        y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        
        kernel = np.sqrt((x**2 + y**2))
        
        for i in range(3):
            for j in range(2):
                if x[i][j] < 0 or y[i][j] < 0:
                    kernel[i][j] *= -1
        
        new = (sobelX ** 2 + sobelY ** 2)**0.5
        new_arr = ((new - new.min()) * (1/(new.max() - new.min()) * 255)).astype('uint8')
        #cv2.imshow("n",new_arr)
        magnitude = cv2.filter2D(gaussian,-1,kernel)
        print(new_arr)
        cv2.imshow("Magnitude",magnitude)


        '''abs_sobelX=cv2.convertScaleAbs(sobelX)
        abs_sobelY=cv2.convertScaleAbs(sobelY)
        mag = cv2.addWeighted(abs_sobelX,0.5,abs_sobelY,0.5,0)
        cv2.imshow("mag",mag)'''
        
        
    def buttonClicked4(self):
        img = cv2.imread("Parrot.png")
        cv2.namedWindow("Transform")
        #cv2.resizeWindow("Transform", 600, 900)
        #cv2.namedWindow("T")
        
        cv2.imshow("Original",img)
        row,col = img.shape[:2]
        R = float(self.line1.text())
        S = float(self.line2.text())
        Tx = float(self.line3.text())
        Ty = float(self.line4.text())
        
        #平移
        Move_pic = np.float32([[1,0,Tx],[0,1,Ty]])#100->line3 50->line4 左右移在前  
        res = cv2.warpAffine(img,Move_pic,(col,row))
        
        #print(self.line1.text(),self.line2.text(),self.line3.text(),self.line4.text())#i->rotate 2->scale 3->x 4->y
        #旋轉
        M = cv2.getRotationMatrix2D((84,160),R,1)
        rotate_img = cv2.warpAffine(res,M,(col,row))
        #cv2.imshow("T",rotate_img)
        #縮放
        scale_img = cv2.resize(rotate_img,(int(S*col),int(S*row)),interpolation=cv2.INTER_CUBIC)
        rows,cols = scale_img.shape[:2]
        
        #cv2.resizeWindow("T", 600, 900)
        cv2.imshow("Transform",scale_img)
        

    
app = QApplication(sys.argv)
w = MainWindow()
w.show()
sys.exit(app.exec_())