from lib import DataProvider
import Config
from models import STAE
from lib import Helpers
import numpy as np

def evaluate():
    # seq = STAE.get_model(re=Config.RELOAD_MODEL)
    seq = STAE.get_my_model(Config.MODEL_PATH)
    print(seq.predict)
    print("got model")
    
    dataProvider = DataProvider
    images = np.array(dataProvider.get_test_set(Config.SINGLE_TEST_PATH))
    #images = dataProvider.get_n_video_frames(Config.SINGLE_TEST_PATH)
    print("got data")
    x_axis_values = []
    min_et = 1e9
    max_et = 0
    for i in range(images.shape[0]):
        x = np.zeros((1, 10, 256, 256, 1))
        print(images.shape)
        
        x[0] = images[i]
        
        print(x.shape)
        
        output = seq.predict(x)
        print(i," == ",output)
        print()
        
        for j in range(0,10):
            et = np.sum(np.square(np.subtract(x[0,:,:,j],output[0,:,:,j])))
            min_et=min(min_et,et)
            max_et=max(max_et,et)
            x_axis_values.append(et)
    x_axis_values =1.0 - (x_axis_values - min_et)/max_et
    x_axis_values = Helpers.movingaverage(x_axis_values,20)
    import matplotlib.pyplot as plt
    plt.plot(x_axis_values)
    plt.ylabel('regularity score')
    plt.show()

evaluate()
