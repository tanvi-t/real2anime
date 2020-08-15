# Real2Anime GUI
In this project, we attempt to implement a face2anime GAN using CycleGAN. 
![exampleresult.png](https://xxx/result.png "result example")

# Usage of GUI

- in terminal, navigate to C:\...\AIproject\cartoon-gan
- run `pip install -r requirements.txt`
- download pre-trained weights Real2Anime.pth from https://github.com/xxx/releases to the same path
- run `make GUI`
- click on Upload Image to access and upload a local face from ./celebAs_test to the GUI
- click Cartoon to see the anime version of the face

# Acknowledgements
This project's was implemented with the listed frameworks and modules, and we fully acknowledge the support they have provided:
1. Pytorch
2. Tkinter for displaying GUI

Programming Languages Used:
1. Python

IDEs Used:
1. Visual Studio Code
2. Google Colab
3. Jupyter Notebook

# Other things to note:
If you are having difficulties installing pycocotools from the requirements.txt file on Windows, run the following (from https://github.com/cocodataset/cocoapi/issues/169): 
1. pip install git
2. pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

If you are having difficulties installing pytorch from the requirements.txt file on Windows, run the following: 
1. pip install torch===1.6.0 torchvision===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html 
