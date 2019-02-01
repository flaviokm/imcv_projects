import cv2
import numpy as np 
import argparse
import time
# from sklearn.cluster import MiniBatchKMeans


class parameters():
    def __init__(self, w_pixels=7, h_pixels=7, pixel_style=cv2.INTER_AREA, reverse = True, color='normal', image_from='file'):
        self.w_pixels = w_pixels
        self.h_pixels = h_pixels
        self.pixel_style = pixel_style
        self.color = color
        self.view_window = 'window'
        self.image_from = 'file'
        self.reverse = False
        self.double_reverse = True
        cv2.namedWindow(self.view_window, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.view_window, 340, 0)
        
        
def pixelate(image,parameters):
    image = cv2.resize(image, (image.shape[0]/parameters.h_pixels,
                               image.shape[1]/parameters.w_pixels), interpolation=parameters.pixel_style)
    image = cv2.resize(image, (param.h,param.w), interpolation=parameters.pixel_style)
    return image

def record(frames,parameters,view = True):
    n_times = 2 if parameters.reverse == True else 1
    pause = 4
    if parameters.double_reverse == True:
        frames.reverse()
    for n in range(1,1+n_times):        
        for n in range(pause):
            parameters.video_writer.write(frames[0])
        for j in range(len(frames)):
            parameters.video_writer.write(frames[j])
            if view==True:
                cv2.imshow(parameters.view_window, frames[j])
                cv2.resizeWindow(parameters.view_window, 640, 460)
                cv2.waitKey(50)
        for n in range(pause):
            parameters.video_writer.write(frames[-1])
        if parameters.reverse == True:
            frames.reverse()
 
def get_image(image_file,param):
    if image_file == 'None':
        param.image_from = 'camera'
    elif image_file=='debug':
        image_file = "/home/flavio/Pictures/Fotos/porto.png"
    if param.image_from == 'camera':
        cap = cv2.VideoCapture(0)
        assert (cap.isOpened()), "Unable to read camera feed"
        ret, image = cap.read()
        cap.release()
    else:
        image = cv2.imread(image_file)
        print("original image size:{}".format(image.shape))
        biggest_size = image.shape[0] if image.shape[0]>image.shape[1] else image.shape[1]
        while biggest_size>1080:
            image = cv2.resize(image, (image.shape[0]/2, image.shape[1]/2))
            biggest_size = image.shape[0] if image.shape[0]>image.shape[1] else image.shape[1]
        print("usable image size:{}".format(image.shape))

    return image

def adjust_color(image,param):
    if param.color == 'gray':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif param.color == 'blue':
        image[:,:,1] = 0
        image[:,:,2] = 0
    elif param.color == 'green':
        image[:,:,0] = 0
        image[:,:,2] = 0
    elif param.color == 'red':
        image[:, :, 0] = 0
        image[:, :, 1] = 0
    elif param.color == 'cyan':
        image[:, :, 2] = 0
    elif param.color == 'yellow':
        image[:, :, 0] = 0
    elif param.color == 'magenta':
        image[:, :, 1] = 0
    return image

# def sk_quantization(image,param,clusters):
#     (h, w) = image.shape[:2]
#     image = image.reshape((image.shape[0] * image.shape[1], 3))
#     clt = MiniBatchKMeans(n_clusters=clusters)
#     labels = clt.fit_predict(image)
#     quant = clt.cluster_centers_.astype("uint8")[labels]
#     quant = quant.reshape((h, w, 3))
#     return quant

def quantization(image,param,clusters):
    (h, w) = image.shape[:2]
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((h,w,3))
    return res2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pixelating!')
    parser.add_argument('--image_file', metavar='image_file', nargs='?', const='image_file', default='None', type=str,
                        help='image file')
    parser.add_argument('--pixels', metavar='pixels', nargs='?', const=30, default=30, type=int,
                        help='max pixelated the image gets')
    parser.add_argument('--color', metavar='color', nargs='?', const='normal', default='normal', type=str,
                        help='if you want colorful pixelated')
    parser.add_argument('--clusters', metavar='clusters', nargs='?', const=0, default=0, type=int,
                        help='if you want to quantizide the pixelated colors')


    parsed = parser.parse_args()
    param = parameters(color=parsed.color)
    image_file = parsed.image_file
    param.max_pixel = parsed.pixels
    
    
    image = get_image(image_file, param)
    image =  adjust_color(image, param)
    if parsed.clusters>1:
        # start = time.time()
        # _ =  sk_quantization(image, param, parsed.clusters)
        # print ("sk quantization took:{}".format(time.time()-start))
        start = time.time()
        image = quantization(image,param,parsed.clusters)
        print ("opencv quantization took: {:.3}s".format(time.time()-start))

    param.w,param.h,_ = image.shape
    param.fps = 10
    param.video_writer = cv2.VideoWriter(
        'output.mp4', 0x00000021, param.fps, (image.shape[1], image.shape[0]))
    frames = []
    for i in range(param.max_pixel):
        param.h_pixels,param.w_pixels = i+1,i+1
        frames.append(pixelate(image,param))
    record(frames,param)
   


