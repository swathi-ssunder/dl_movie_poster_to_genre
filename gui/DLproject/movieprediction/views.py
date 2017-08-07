from django.shortcuts import render
from movieprediction.models import UploadForm,Upload
from django.http import HttpResponseRedirect
from django.core.urlresolvers import reverse


import os
import PIL
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms, utils, datasets
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import copy
import shutil

TRAINED_MODEL_PATH = 'model/model.pth.tar'
num_classes = 29

# GPU-related configurations
USE_GPU = torch.cuda.is_available()
GPUS = [0, 1, 2]
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

class CustomNet(nn.Module):
    def __init__(self, model):
        """
        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.
        """
        super(CustomNet, self).__init__()
        self.model = model

        #if freeze == 'all':
        #    for param in self.model.parameters():
        #        param.requires_grad = False
        #else:
        #    for layer in freeze:
        #        for param in getattr(self.model, layer).parameters():
        #            param.requires_grad = False

        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables.
        """
        x = self.model(x)
#         x = nn.functional.softmax(x)
        return x

base_model = models.resnet152(pretrained=True)
the_model = CustomNet(base_model)

if USE_GPU:
    the_model = the_model.cuda()

the_model.load_state_dict(torch.load('model/model.pth.tar'))
classes = ['Animation',
     'Adventure',
     'Comedy',
     'Action',
     'Family',
     'Romance',
     'Drama',
     'Crime',
     'Thriller',
     'Fantasy',
     'Horror',
     'Biography',
     'History',
     'Mystery',
     'Sci-Fi',
     'War',
     'Sport',
     'Music',
     'Documentary',
     'Musical',
     'Western',
     'Short',
     'Film-Noir',
     'nan',
     'Talk-Show',
     'News',
     'Adult',
     'Reality-TV',
     'Game-Show']

def predict(model, image):

    model.train(False)

    image = Variable(image.cuda(), requires_grad=True)

    # forward
    outputs = model(image)
    j = 0
    j, preds = torch.exp(outputs.data[0]).topk(3)
    preds = [p for k, p in enumerate(preds) if j[k] != 0]

    predicted_classes = []
    for p in preds:
        predicted_classes.append(classes[p])

    return predicted_classes

trick = 0

# Create your views here.
def index(request):


    global trick

    if request.method=="POST":
        img = UploadForm(request.POST, request.FILES)
        if img.is_valid():
            Upload.objects.all().delete()
            trick = 1
            img.save()
            return HttpResponseRedirect(reverse('imageupload'))
    else:
        img=UploadForm()

    user_image=Upload.objects.latest('upload_date').pic
    image = PIL.Image.open(user_image)
    image = torch.from_numpy(np.transpose(np.array(image.resize((64,64))).T, (0,2,1))).float().unsqueeze(0)
    predictions = predict(the_model, image)
    if trick == 0:
        predictions = ["","",""]

    return render(request,'index.html',{'form':img,'predictions':predictions})
