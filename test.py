#After training the model we want to perform testing on a new image to test the validity of our model

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import load_model

# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(32, 32))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 32, 32, 3)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img
 
# load an image and predict the class
def test():
    # load the image
    img = load_image('.//test_images//bird.png')
    # load model
    model = load_model('final_model.h5')
    # predict the class
    result = model.predict_classes(img)
    if(result[0] == 0):
        print('Airplane Detected')
    elif(result[0] == 1):
        print('Automobile detected')
    elif(result[0] == 2):
        print('Bird detected')
    elif(result[0] == 3):
        print('Cat detected')
    elif(result[0] == 4):
        print('Deer detected')
    elif(result[0] == 5):
        print('Dog detected')
    elif(result[0] == 6):
        print('Frog detected')
    elif(result[0] == 7):
        print('Horse detected')
    elif(result[0] == 8):
        print('Ship detected')
    else:
        print('Truck detected')
        
 
# entry point, run the example
test()