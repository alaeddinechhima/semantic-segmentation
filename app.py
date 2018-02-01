from flask import Flask , render_template,request
from infer import image_segmentation
from convolutional_autoencoder import train_model,Network
import convolutional_autoencoder
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
from flask.ext.images import Images
import os

app = Flask(__name__, template_folder='templates/')


network = convolutional_autoencoder.Network()


@app.route('/')
def index():
    return  render_template('index.html')

@app.route('/train')
def training():
    return  render_template('train.html')
@app.route('/segment')
def segmentation():
    return  render_template('segmentation.html')
@app.route('/folder_segmentation')
def segment_all():
    return  render_template('folder_segmentation.html')

@app.route('/segment',methods=['POST', 'GET'])
def segment():
    image_path = request.form.get('image_path')
    model_dir = request.form.get('model_dir')
    out = request.form.get('out')
    image_segmentation(str(image_path),str(model_dir),str(out),network)
    name=image_path.split(".")[0]
    url=name[len(name)-10:]+".jpg"
    url="static/result/"+url
    url_origin=image_path
    url_label=image_path.split("inputs")[0]+"targets/"+name[len(name)-10:]+".png"
    return  render_template('segmentation.html',url=url,url_origin=url_origin,url_label=url_label)

@app.route('/train',methods=['POST', 'GET'])
def train():
    dir_path = request.form.get('dir_path')
    save = request.form.get('save')
    train_model(dir_path,save,network)
    return  render_template('train.html')



@app.route('/folder_segmentation',methods=['POST', 'GET'])
def folder_segmentation():
    folder_path = request.form.get('folder_path')
    model_dir = request.form.get('model_dir')
    out = request.form.get('out')
    arr = os.listdir(str(folder_path))
    print(arr[1])
    for img in arr :
        image_path=folder_path+'/'+img
        print(image_path)
        image_segmentation(str(image_path),str(model_dir),str(out),network)
    return  render_template('folder_segmentation.html')



if __name__ == "__main__":

    app.run()
