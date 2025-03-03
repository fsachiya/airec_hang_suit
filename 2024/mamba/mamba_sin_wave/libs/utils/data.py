import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import torch


def normalization(data, indataRange, outdataRange):
    '''
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        indataRange (float list):  List of maximum and minimum values of original data, e.g. indataRange=[0.0, 255.0].
        outdataRange (float list): List of maximum and minimum values of output data, e.g. indataRange=[0.0, 1.0].
    Return:
        data (np.array): Normalized data array
    '''
    data = (data - indataRange[0]) * (outdataRange[1] - outdataRange[0]) / (indataRange[1] - indataRange[0]) + outdataRange[0]
    
    return data


def getLissajous(total_step, num_cycle, x_mag, y_mag, delta, dtype=np.float32):
    '''
    Function to generate a Lissajous curve
    Reference URL: http://www.ne.jp/asahi/tokyo/nkgw/www_2/gakusyu/rikigaku/Lissajous/Lissajous_kaisetu/Lissajous_kaisetu_1.html
    Args:
        total_step (int): Sequence length of Lissajous curve.
        num_cycle  (int): Iteration of the Lissajous curve.
        x_mag    (float): Angular frequency of the x direction.
        y_mag    (float): Angular frequency of the y direction.
        delta    (float): Initial phase of the y direction
    Return:
        data (np.array): Array of Lissajous curves. data shape is [total_step, 2]
    '''
    
    t = (2.0 * np.pi / (total_step / num_cycle)) * np.arange(total_step)
    x = np.cos(x_mag * t)
    y = np.cos(y_mag * t + delta)
    
    return np.c_[x,y].astype(dtype)


def getLissajousMovie(total_step, num_cycle, x_mag, y_mag, delta, imsize, circle_r, color, vmin=-0.9, vmax=0.9):
    '''
    Function to generate a Lissajous curve with movie
    Args:
        total_step (int): Sequence length of Lissajous curve.
        num_cycle  (int): Iteration of the Lissajous curve.
        x_mag    (float): Angular frequency of the x direction.
        y_mag    (float): Angular frequency of the y direction.
        delta    (float): Initial phase of the y direction
        imsize     (int): Pixel size of the movie
        circle_r   (int): Radius of the circle moving in the movie.
        color     (list): Color of the circle. Specify color in RGB list, e.g. red is [255,0,0].
        vmin     (float): Minimum value of output data
        vmax     (float): Maximum value of output data
    Return:
        data (np.array): Array of movie and curve. movie shape is [total_step, imsize, imsize, 3], curve shape is [total_step, 2].
    '''

    #Use the normalization function.
    xy = getLissajous( total_step, num_cycle, x_mag, y_mag, delta )
    x, y = np.split(xy, indices_or_sections=2, axis=-1)
    
    _color = tuple((np.array(color)).astype(np.uint8))
    
    imgs = []
    for _t in range(total_step):
        # xy position in the image
        _x = x[_t]
        _y = y[_t]
        img = Image.new("RGB", (imsize, imsize), "white")
        draw = ImageDraw.Draw(img)
        # Draws a circle with a specified radius
        draw.ellipse((_x-circle_r, _y-circle_r, _x+circle_r, _y+circle_r) , fill=_color)
        imgs.append(np.expand_dims(np.asarray(img), 0))
    imgs = np.vstack(imgs)
    
    ### normalization
    imgs = normalization(imgs.astype(np.float32), [0, 255], [vmin, vmax])
    seq = normalization(np.c_[x,y].astype(np.float32), [-1.0, 1.0], [vmin, vmax])
    return imgs, seq    # seq: sinカーブのyの値


# # test func
# def getCycle(p_p_n, p_patt, a_patt, total_step, flag, dtype=np.float32): # total_step = 1000
#     '''
#     Function to generate a Lissajous curve
#     Reference URL: http://www.ne.jp/asahi/tokyo/nkgw/www_2/gakusyu/rikigaku/Lissajous/Lissajous_kaisetu/Lissajous_kaisetu_1.html
#     Args:
#         total_step (int): Sequence length of Lissajous curve.
#         num_cycle  (int): Iteration of the Lissajous curve.
#         x_mag    (float): Angular frequency of the x direction.
#         y_mag    (float): Angular frequency of the y direction.
#         delta    (float): Initial phase of the y direction
#     Return:
#         data (np.array): Array of Lissajous curves. data shape is [total_step, 2]
#     '''
    
#     p_patt = np.array(p_patt * p_p_n)
#     n = len(p_patt)

#     if n%len(a_patt) == 0:
#         a_patt = np.array(a_patt * int(n/len(a_patt)))
#     else:
#         a_patt = np.array(a_patt * int(n/len(a_patt)) + a_patt[0 : n%len(a_patt)])

#     p_arr = p_patt * (np.ones(n) * 1.0*np.pi)
#     r_arr = [sum(p_arr[:i]) for i in range(n+1)]
#     # print(r_arr[-1])

#     cycle_n = int(r_arr[-1] / (2.0 * np.pi))
#     x_arr = np.linspace(0.0, 2.0 * np.pi * cycle_n, total_step)
#     # print(x_arr[-1])
#     y_arr = np.ones_like(x_arr)

#     for i, x in enumerate(x_arr):
#         for j in range(n):
#             r = r_arr[j:j+2]
#             if (r[0] <= x) and (x < r[1]):
#                 y_arr[i] = a_patt[j] * (-1)**j * np.sin(np.pi / p_arr[j] * (x - r[0]))
#                 # print(y_arr[i])
    
#     x, y = x_arr, y_arr
#     if flag:
#         plt.plot(x, y)
#         plt.show()
#     return np.c_[x,y].astype(dtype)


# # test func
# def getCycleMovie(p_p_n, p_patt, a_patt, total_step, imsize, circle_r, color, flag, vmin=-0.9, vmax=0.9):
#     '''
#     Function to generate a Lissajous curve with movie
#     Args:
#         total_step (int): Sequence length of Lissajous curve.
#         num_cycle  (int): Iteration of the Lissajous curve.
#         x_mag    (float): Angular frequency of the x direction.
#         y_mag    (float): Angular frequency of the y direction.
#         delta    (float): Initial phase of the y direction
#         imsize     (int): Pixel size of the movie
#         circle_r   (int): Radius of the circle moving in the movie.
#         color     (list): Color of the circle. Specify color in RGB list, e.g. red is [255,0,0].
#         vmin     (float): Minimum value of output data
#         vmax     (float): Maximum value of output data
#     Return:
#         data (np.array): Array of movie and curve. movie shape is [total_step, imsize, imsize, 3], curve shape is [total_step, 2].
#     '''

#     #Use the normalization function.
#     xy = getCycle(p_p_n, p_patt, a_patt, total_step, flag)
#     x, y = np.split(xy, indices_or_sections=2, axis=-1)
    
#     _color = tuple((np.array(color)).astype(np.uint8))
    
#     imgs = []
#     for _t in range(total_step):
#         # xy position in the image
#         _x = x[_t]
#         _y = y[_t]
#         img = Image.new("RGB", (imsize, imsize), "white")
#         draw = ImageDraw.Draw(img)
#         # Draws a circle with a specified radius
#         draw.ellipse((_x-circle_r, _y-circle_r, _x+circle_r, _y+circle_r) , fill=_color)
#         imgs.append(np.expand_dims(np.asarray(img), 0))
#     imgs = np.vstack(imgs)
    
#     ### normalization
#     imgs = normalization(imgs.astype(np.float32), [0, 255], [vmin, vmax])
#     seq = normalization(np.c_[x,y].astype(np.float32), [-1.0, 1.0], [vmin, vmax])
#     return imgs, seq    # seq: sinカーブのyの値


# test func
def getSquareWave(cycle_cnt, n_th, wavelen, ampl, total_step, flag, dtype=np.float32): # total_step = 1000
    
    x_arr = np.linspace(0, wavelen*cycle_cnt, total_step)
    y_arr = np.ones_like(x_arr)
    
    for i, x in enumerate(x_arr):
        y = 0.0
        for j in range(n_th+1):
            # import ipdb; ipdb.set_trace()
            y += ampl * 4.0/np.pi * np.sin((2.0*j+1.0) * 2.0*np.pi/wavelen * x) / (2.0*j+1.0)
        y_arr[i] = y
        
    x, y = x_arr, y_arr
    if flag:
        plt.plot(x, y)
        plt.show()
    return np.c_[x,y].astype(dtype)
    


# test func
def getSquareWaveMovie(cycle_cnt, n_th, wavelen, ampl, total_step, imsize, circle_r, color, flag):  # , vmin=-1.0, vmax=1.0
    
    #Use the normalization function.
    xy = getSquareWave(cycle_cnt, n_th, wavelen, ampl, total_step, flag)
    x, y = np.split(xy, indices_or_sections=2, axis=-1)
    
    _color = tuple((np.array(color)).astype(np.uint8))
    
    imgs = []
    for _t in range(total_step):
        # xy position in the image
        _x = x[_t]
        _y = y[_t]
        img = Image.new("RGB", (imsize, imsize), "white")
        draw = ImageDraw.Draw(img)
        # Draws a circle with a specified radius
        draw.ellipse((_x-circle_r, _y-circle_r, _x+circle_r, _y+circle_r) , fill=_color)
        imgs.append(np.expand_dims(np.asarray(img), 0))
    imgs = np.vstack(imgs)
    
    ### normalization
    # imgs = normalization(imgs.astype(np.float32), [0, 255], [vmin, vmax])
    imgs = imgs.astype(np.float32)
    # seq = normalization(np.c_[x,y].astype(np.float32), [-1.0, 1.0], [vmin, vmax])
    seq = np.c_[x,y].astype(np.float32)
    return imgs, seq    # seq: sinカーブのyの値


def deprocess_img(data, vmin=-0.9, vmax=0.9):
    '''
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        vmin (float):  Minimum value of input data
        vmax (float):  Maximum value of input data
    Return:
        data (np.array with np.uint8): Normalized data array from 0 to 255.
    '''
    
    #Use the normalization function.
    data = normalization(data, [vmin, vmax], [0, 255])
    
    return data



def get_batch( x, BATCH_SIZE):
    '''
    Shuffle the input data and extract data specified by batch size.
    '''
    index = list(range(len(x)))
    # print(index)
    np.random.shuffle(index)
    
    x = x.detach().numpy()
    batch = np.array([x[i] for i in index][:BATCH_SIZE])
    # print(batch.shape)
    batch = torch.from_numpy(batch.astype(np.float32))
    
    return  batch 

def tensor2numpy(x):
    '''
    Convert tensor to numpy array.
    '''
    x = x.to('cpu').detach().numpy().copy()
    
    return x



# def gen_params_combination(n_th_arr, wavele_arr, ampl_arr):
#     params = {"n_th": 0, "wavelen": 0, "ampl": 0}
#     for i in range(n_th_arr):
#         params["n_th"] = n_th_arr[i]
#         for j in range(wavele_arr):
#             params["wavelen"] = wavele_arr[j]
#             for k in range(ampl_arr):
#                 params["ampl"] = ampl_arr[k]
#     return params


# def sin_patt(is_normal):
#     if is_normal:
#         return  {
#             "train":[
#                         [1,1,1],
#                         [1.5,1.5,1.5],
#                         [2,2,2],
#                         [3,3,3],
#                         [5,5,5]
#                     ],
#             "test": [
#                         [0.5,0.5,0.5],
#                         [1.7,1.7,1.7],
#                         [2.5,2.5,2.5],
#                         [4,4,4],
#                         [6,6,6]
#                     ]
#         }
       
#     else:
#         return  {
#             "train":[
#                         [1.5, 1.3, 1, 1/1.5, 1/2, 1/3, 1/4, 1/5, 1/6, 1/5, 1/4, 1/3, 1/2, 1/1.5, 1, 1.3],
#                         [1/6, 1/5, 1/4, 1/3, 1/2, 1/1.5, 1, 1.3,1.5],
#                         [1.5,1.3,1,1.5,1/2,1/3,1/4,1/5,1/6],
#                         [1.5,1/6,1.3,1/5,1,1/4,1/1.5,1/3,1/2],
#                         [1,2,3,4,5,4,3,2],
#                         [1,2,3,4,5],
#                         [5,4,3,2,1],
#                         [1,5,2,4,3]
#                     ],
#             "test": [
#                         [2.5, 2.3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.2, 0.3, 0.4, 0.5, 0.7, 1, 2, 2.3],
#                         [0.2, 0.3, 0.4, 0.5, 0.7, 1, 2, 2.3, 2.5],
#                         [2.5, 2.3, 2, 1, 0.7, 0.5, 0.4, 0.3, 0.2],
#                         [2.5, 0.2, 2.3, 0.3, 2, 0.4, 1, 0.5, 0.7],
#                         [1.5,2.5,3.5,4.5,5.5,4.5,3.5,2.5],
#                         [1.5,2.5,3.5,4.5,5.5],
#                         [5.5,4.5,3.5,2.5,1.5],
#                         [1.5,5.5,2.5,4.5,3.5]
#                     ]
#         }
    

# def gen_sin_patt(sin_type):
#     if sin_type == "normal":
#         return sin_patt(is_normal=True), sin_patt(is_normal=True)
#     elif sin_type == "ampl":
#         return sin_patt(is_normal=True), sin_patt(is_normal=False)
#     elif sin_type == "wavelen":
#         return sin_patt(is_normal=False), sin_patt(is_normal=True)
#     else:
#         return sin_patt(is_normal=False), sin_patt(is_normal=False)
    
def get_unique_list(seq):
    seen = []
    return [x for x in seq if x not in seen and not seen.append(x)]