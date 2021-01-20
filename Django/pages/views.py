from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import TemplateView
import tensorflow as tf
import numpy as np

from PIL import Image
from tensorflow.keras.models import load_model

class HomePageView(TemplateView):
    template_name = 'home.html'

class AboutPageView(TemplateView):
    template_name = 'about.html'

def predictImage(request):
    print (request)
    print (request.POST.dict())

    def SSIMLoss(y_true, y_pred):
        return tf.math.square(1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2)))

    compression_model = load_model("D:\e6998_img_compression\compression_model.h5", custom_objects={'SSIMLoss':SSIMLoss})

    fileObj = request.FILES['filePath']
    fs = FileSystemStorage()

    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    testimage = '.' + filePathName

    img = Image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    x /= 255
    x = x.reshape(1, img_height, img_width, 3)

    with model_graph.as_default():
        with tf_session.as_default():
            predi = model.predict(x)

    final_image = Image.fromarray(predi)
    final_image.save("D:\e6998_img_compression\output_compressed.png")

    context={'filePathName':filePathName}
    return render(request,'index.html',context)

"""
# Create your views here.
def homePageView(request):
    return  HttpResponse('Kind of! Does this count as progress?')
"""
