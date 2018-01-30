from flask import Flask , render_template,request
from infer import image_segmentation
from convolutional_autoencoder import train_model
from flask.ext.images import Images
import os

app = Flask(__name__, template_folder='templates/')
app.secret_key = 'monkey'
app.debug = True
IMAGES_URL = "/result"
app.config['IMAGES_URL'] = IMAGES_URL
images = Images(app)

@app.route('/')
def index():
    return  render_template('index.html')

@app.route('/train')
def training():
    return  render_template('train.html')
@app.route('/segment')
def segmentation():
    return  render_template('segmentation.html')

@app.route('/segment',methods=['POST', 'GET'])
def segment():
    image_path = request.form.get('image_path')
    model_dir = request.form.get('model_dir')
    out = request.form.get('out')
    image_segmentation(str(image_path),str(model_dir),str(out))
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
    image_segmentation(str(image_path),str(model_dir),str(out))
    train_model(dir_path)
    url_label=image_path.split("inputs")[0]+"targets/"+name[len(name)-10:]+".png"
    return  render_template('segmentation.html',url=url,url_origin=url_origin,url_label=url_label)

@app.route('/folder_segmentation',methods=['POST', 'GET'])
def folder_segmentation():
    dir_path = request.form.get('dir_path')
    save = request.form.get('save')
    image_segmentation(str(image_path),str(model_dir),str(out))
    train_model(dir_path)
    url_label=image_path.split("inputs")[0]+"targets/"+name[len(name)-10:]+".png"
    return  render_template('segmentation.html',url=url,url_origin=url_origin,url_label=url_label)



if __name__ == "__main__":

    app.run()
