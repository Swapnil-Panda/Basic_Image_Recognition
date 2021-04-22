import argparse
import tensorflow as tf

import tensorflow_hub as hub

from PIL import Image

import numpy as np

import argparse

import json
#reloaded_model = tf.keras.experimental.load_from_saved_model('./best_model.h5', custom_objects={'KerasLayer':hub.KerasLayer})
#reloaded_model.build((None, 224, 224, 3))
parser=argparse.ArgumentParser(
    description="predicting flower class"
)
parser.add_argument('indir', type=str, help='Input dir for image')
parser.add_argument('model', type=str, help='nn model for predicting')
parser.add_argument('--top_K', type=int, help='how many probabilities to show')
parser.add_argument('--category_names', help='a label map')

args=parser.parse_args()

#print(args)

#print(args.model)
with open(args.category_names, 'r') as f:
    class_names = json.load(f)
top_K=args.top_K
model =  tf.keras.models.load_model(args.model,custom_objects={'KerasLayer':hub.KerasLayer})

#model=tf.keras.models.load_model(saved_model,custom_objects={'KerasLayer':hub.KerasLayer})
#model.build((None, 224, 224, 3))
#model.summary()
image_path=args.indir
image_size=224
im = Image.open(image_path)
test_image = np.asarray(im)
tensor_image = tf.convert_to_tensor(test_image)
image=tf.image.resize(tensor_image, (image_size, image_size))
image /= 255
image = image.numpy()
image=np.expand_dims(image, axis=0)
a=model.predict(image)

b=np.argsort(a,axis=1)
b=np.fliplr(b)
classes=b[0,:top_K]
probs=a[0,classes]
strs = ['']*top_K
for i in range(top_K):
    strs[i]=class_names[str(classes[i]+1)]
    print(strs[i],':',probs[i])
    

