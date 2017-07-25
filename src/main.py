import numpy as np
from keras.applications import vgg16
from keras import backend as K
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave
import cv2
from PIL import Image
import time

img_b = cv2.imread("~/images/frog.jpg",1)
img_s = cv2.imread("~/images/starry_nights.jpg", 1)
img_b = cv2.resize(img_b, (512, 512))
img_s = cv2.resize(img_s, (512, 512))
img_b = np.expand_dims(img_b, axis=0)
img_s = np.expand_dims(img_s, axis=0)

img_b[:, :, :, 0] = np.subtract(img_b[:, :, :, 0], 103.939, out=img_b[:, :, :, 0], casting="unsafe")
img_b[:, :, :, 1] = np.subtract(img_b[:, :, :, 1], 116.779, out=img_b[:, :, :, 1], casting="unsafe")
img_b[:, :, :, 2] = np.subtract(img_b[:, :, :, 2], 123.68, out=img_b[:, :, :, 2], casting="unsafe")
img_b = img_b[:, :, :, ::-1]

img_s[:, :, :, 0] = np.subtract(img_s[:, :, :, 0], 103.939, out=img_s[:, :, :, 0], casting="unsafe")
img_s[:, :, :, 1] = np.subtract(img_s[:, :, :, 1], 116.779, out=img_s[:, :, :, 1], casting="unsafe")
img_s[:, :, :, 2] = np.subtract(img_s[:, :, :, 2], 123.68, out=img_s[:, :, :, 2], casting="unsafe")
img_s = img_s[:, :, :, ::-1]

img_b = K.variable(img_b)
img_s = K.variable(img_s)
combination_image = K.placeholder((1, 512, 512, 3))

input_tensor = K.concatenate([img_b, img_s, combination_image], axis=0)
model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)

layers = dict([(layer.name, layer.output) for layer in model.layers])
content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0
loss = K.variable(0.)
height = 512
width = 512

def content_loss(content, combination):
    return K.sum(K.square(combination - content))

layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * content_loss(content_image_features,
                                      combination_features)

def gram_matrix(x):
    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']
for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

    
def total_variation_loss(x):
    a = K.square(x[:, :height-1, :width-1, :] - x[:, 1:, :width-1, :])
    b = K.square(x[:, :height-1, :width-1, :] - x[:, :height-1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))

loss += total_variation_weight * total_variation_loss(combination_image)

grads = K.gradients(loss, combination_image)

outputs = [loss]
outputs += grads
f_outputs = K.function([combination_image], outputs)

def eval_loss_and_grads(x):
    x = x.reshape((1, height, width, 3))
    outs = f_outputs([x])
    loss_value = outs[0]
    grad_values = outs[1].flatten().astype('float64')
    return loss_value, grad_values

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()


x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.

iterations = 10

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, 
                                     x.flatten(), 
                                     fprime=evaluator.grads,
                                     maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))



x = x.reshape((height, width, 3))

x = x[:, :, ::-1]
x[:, :, 0] += np.add(x[:, :, 0], 103.939, out=x[:, :, 0], casting="unsafe")
x[:, :, 1] += np.add(x[:, :, 1], 116.779, out=x[:, :, 1], casting="unsafe")
x[:, :, 2] += np.add(x[:, :, 2], 123.68, out=x[:, :, 2], casting="unsafe")
x = np.clip(x, 0, 255).astype('uint8')

Image.fromarray(x)
