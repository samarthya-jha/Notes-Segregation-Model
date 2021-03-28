import base64
import io
from PIL import Image
import keras.preprocessing.image as kimg
from keras.models import load_model

def get_model():
    global model
    model=load_model('vgg16model.h5')
    print('Model Loaded')

def image(name):
    img=kimg.img_to_array(name)
    img=img.reshape(1,224,224,3)
    img=img.astype('float32')
    img=img-[123.68,116.779,103.939]
    return img

get_model()

img=open('C:/Users/HP/Desktop/flaskmodel/imageclassify/IMG_20191025_200251.jpg','rb')
encoded=base64.b64encode(img.read())
decoded=base64.b64decode(encoded)
name=Image.open(io.BytesIO(decoded))
name=name.resize((224,224))
processed=image(name)
predictions=model.predict(processed)
model_json=model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print(predictions)