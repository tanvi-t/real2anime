import tkinter as tk #
from tkinter import *#
from tkinter import filedialog#
import tkinter.messagebox ## CLEANUP
from PIL import Image, ImageTk#
import os
import os.path
import numpy as np
import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
"""
Dataset 1: 3400 anime and 3400 | real 100 anime 100 real —> 10 epochs, 0.001 learning rate
Dataset 2: 3400 anime and 3400 | real 100 anime 100 real —> 10 epochs, 0.001 learning rate
Dataset 3: 1700 anime+1700 anime 2, 1700 real + 1700 real 2 | real 50, real 2 50 anime 50 anime 2 50 —> epochs, 0.001 learning rate
Model AI_project_cyclegan http://192.168.33.183/user/1003068/notebooks/AI_project_cyclegan%20(2)%20TT.ipynb 
"""
class ResidualBlock(nn.Module):
    """ The following model directly follows the 
    ResidualBlock, Generator and Discriminator 
    used for training and testing in AI_project_cyclegan.ipynb
    """
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            # Initial Block
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),

            # Downsampling
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(inplace=True),

            # Residual Blocks
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # Upsampling
            nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(inplace=True),

            # Output
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, padding=1),
            nn.InstanceNorm2d(512), 
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class Window(Frame):
    """
    The following instantiates the Tkinter GUI window with functions and placement 
    to upload local images to see results or see results from 3 preset examples
    """
    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.pos = []
        self.master.title("Cartoon")
        self.pack(fill=BOTH, expand=1)
        upload=Button(root,text="Upload Image",command= self.uploadImage)
        upload.configure(background='#58c38b', foreground='white',font=('calibri',12)) # 'bold' #https://www.colorhexa.com
        upload.place(relx=0.25,rely=0.5,anchor=CENTER)
        cartoon=Button(root,text="Anime-ate Image!",command= self.cartoon)
        cartoon.configure(background='#58c38b', foreground='white',font=('calibri',12))
        cartoon.place(relx=0.75,rely=0.5,anchor=CENTER)
        self.canvas = tk.Canvas(self)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.image = None
        self.image2 = None
        label1=Label(self,image=img)
        label1.image=img
        label1.place(relx=0.5, rely=0.8, anchor=CENTER) #### LOGO placement so it scales

        def example1(): # here bc is setup
            """
            3 preset examples are defined here. Although we could have used a function 
            to make this more scalable, we decided not to since it is not the focus of the project
            """
            # print("ratty")
            self.filename='./media/eg1.jpg'
            self.uploadImage()
            self.cartoon()
            self.filename=None # reset for upload#######
        eg1=Button(root,text="1st example",command= example1)
        eg1.configure(background='#90d7b2', foreground='white',font=('calibri',12))
        eg1.place(relx=0.25,rely=0.6,anchor=CENTER)
        def example2(): # here bc is setup
            # print("ratty")
            self.filename='./media/eg2.jpg'
            self.uploadImage()
            self.cartoon()
            self.filename=None # reset for upload#######
        eg2=Button(root,text="2nd example",command= example2)
        eg2.configure(background='#90d7b2', foreground='white',font=('calibri',12))
        eg2.place(relx=0.25,rely=0.68,anchor=CENTER)
        def example3(): # here bc is setup
            # print("ratty")
            self.filename='./media/eg3.jpg'
            self.uploadImage()
            self.cartoon()
            self.filename=None # reset for upload#######
        eg3=Button(root,text="3rd example",command= example3)
        eg3.configure(background='#90d7b2', foreground='white',font=('calibri',12))
        eg3.place(relx=0.25,rely=0.76,anchor=CENTER)

    def uploadImage(self):
        """
        Instantiates a popup to upload local images, if a preset example is not chosen                
        """
        try:
            if self.filename!='./media/eg1.jpg' and self.filename!='./media/eg2.jpg' and self.filename!='./media/eg3.jpg':
                self.filename = filedialog.askopenfilename(initialdir=os.getcwd()) ####C:/Users/Tanvi/Desktop/AIproject/cartoon-gan/GANTEST.jpg
                if not self.filename:
                    return
        except AttributeError:
            self.filename = filedialog.askopenfilename(initialdir=os.getcwd()) ####C:/Users/Tanvi/Desktop/AIproject/cartoon-gan/GANTEST.jpg
            if not self.filename:
                return

        load = Image.open(self.filename) 
        transf = transforms.Compose([
        transforms.Resize(128), 
        ])
        load = transf(load)

        if self.image is None:
            w, h = load.size
            width, height = root.winfo_width(), root.winfo_height()
            self.render = ImageTk.PhotoImage(load)
            self.image = self.canvas.create_image((w / 2, h / 2), image=self.render)
            self.cartoon
           
        else:
            self.canvas.delete(self.image3)
            self.canvas.delete(self.image4)
            self.canvas.delete(self.image5)
            w, h = load.size
            width, height = root.winfo_screenmmwidth(), root.winfo_screenheight()
            self.render2 = ImageTk.PhotoImage(load)
            self.image2 = self.canvas.create_image((w / 2, h / 2), image=self.render2)
            frame = cv.imread(self.filename)

    def cartoon(self):
        # Warnings for wrong image types are printed from here
        if(len(self.filename) < 2): 
            print("Usage: choose a proper image")
            exit(0)
        if not (os.path.isfile(self.filename)):
            print("{} is not a file".format(self.filename))
            exit(0)

        transformer = transforms.Compose([ 
        transforms.Resize(256),
        transforms.ToTensor()
        ])

        """
        Instantiates 3 sets of Weights for the same Generator Real2Anime, 
        but for 3 different Datasets
        """
        checkpoint1 = torch.load('./weights/weights1.pth', map_location='cpu') 
        G1 = Generator(3,3).to('cpu')
        G1.load_state_dict(checkpoint1['g_state_dict'])
        checkpoint2 = torch.load('./weights/weights2.pth', map_location='cpu') 
        G2 = Generator(3,3).to('cpu')
        G2.load_state_dict(checkpoint2['g_state_dict'])
        checkpoint3 = torch.load('./weights/weights3.pth', map_location='cpu') 
        G3 = Generator(3,3).to('cpu')
        G3.load_state_dict(checkpoint3['g_state_dict'])
        
        im_size = 128
        transform_real = transforms.Compose([
                transforms.Resize((im_size,im_size),Image.NEAREST),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
        # transform_real_64 = transforms.Compose([
        #     transforms.CenterCrop(160),
        #     transforms.Resize((64,64),Image.NEAREST),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        #     ])
 

        with Image.open(self.filename) as img:
            """
            Input image is given as a batch. 
            3 different results for 3 different R2A weights. 
            The results are also saved locally for easy access. 
            """            
            pseudo_batched_img = transform_real(img)
            pseudo_batched_img = pseudo_batched_img[None]
            result1 = G1(pseudo_batched_img)
            result1 = transforms.ToPILImage()(result1[0]).convert('RGB')
            result1 = result1.resize((im_size,im_size))
            result1.save('transformed1.'+img.format)

            # pseudo_batched_img_64 = transform_real_64(img)
            # result2 = G2(pseudo_batched_img_64) 
            result2 = G2(pseudo_batched_img)           
            result2 = transforms.ToPILImage()(result2[0]).convert('RGB')
            result2 = result2.resize((im_size,im_size))
            result2.save('transformed2.'+img.format)
            
            result3 = G3(pseudo_batched_img) 
            result3 = transforms.ToPILImage()(result3[0]).convert('RGB')
            result3 = result3.resize((im_size,im_size))
            result3.save('transformed3.'+img.format)

        if self.image is None:
            w, h = result1.size 
            self.render01 = ImageTk.PhotoImage(result1)
            self.image = self.canvas.create_image((w/2,h/2), anchor=CENTER, image=self.render01)
            root.geometry("%dx%d" % (w, h))
            w, h = result2.size 
            self.render02 = ImageTk.PhotoImage(result2)
            self.image = self.canvas.create_image((w / 2, h/2), image=self.render02)
            root.geometry("%dx%d" % (w, h))
            w, h = result3.size 
            self.render03 = ImageTk.PhotoImage(result3)
            self.image = self.canvas.create_image((w/2,h/2), anchor=CENTER, image=self.render03)
            root.geometry("%dx%d" % (w, h))
        else:
            # Shows output for the weights from 3 different datasets
            w, h = result1.size # (w / 2, h / 2) = 64,64
            width, height = root.winfo_screenmmwidth(), root.winfo_screenheight()
            self.render11 = ImageTk.PhotoImage(result1)
            self.image3 = self.canvas.create_image((50,64),anchor=CENTER,image=self.render11)
            text = Label(self, text="Dataset 1")
            text.place(x=520,y=130)
            self.canvas.move(self.image3, 500, 0) 

            w, h = result2.size
            width, height = root.winfo_screenmmwidth(), root.winfo_screenheight()
            self.render12 = ImageTk.PhotoImage(result2)
            self.image4 = self.canvas.create_image((200,64), image=self.render12)
            text = Label(self, text="Dataset 2")
            text.place(x=670,y=130)
            self.canvas.move(self.image4, 500, 0)
             
            w, h = result3.size ##  THIS PLAYS
            width, height = root.winfo_screenmmwidth(), root.winfo_screenheight()
            self.render13 = ImageTk.PhotoImage(result3)
            self.image5 = self.canvas.create_image((350,64), anchor=CENTER, image=self.render13)
            text = Label(self, text="Dataset 3")
            text.place(x=820,y=130)
            self.canvas.move(self.image5, 500, 0) 

if __name__=='__main__':
    # Begins loops to instantiate GUI
    root = tk.Tk()
    root.geometry("%dx%d" % (980, 600))
    root.title("Anime-ate Image GUI")
    img = ImageTk.PhotoImage(Image.open("./media/logo.png")) ###center ittt
    app = Window(root)
    app.pack(fill=tk.BOTH, expand=1)
    root.mainloop()