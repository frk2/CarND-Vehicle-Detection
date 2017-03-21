#!/usr/bin/python3
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pickle
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

from pathlib import Path
import pdb

debug = 0
# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))
                        
def color_hist(img, nbins=32):    #bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features
# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', spatial_size=(16,16), hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        local_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # image = image.astype(np.float32)/255.
        # apply color conversion if other than 'RGB'
        fig = plt.figure()
        plt.subplot(241)
        plt.title('orig')
        plt.imshow(image)
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
        plt.subplot(242)
        plt.title('CH1')
        plt.imshow(feature_image[:,:,0], cmap='gray')
        plt.subplot(243)
        plt.title('CH2')
        plt.imshow(feature_image[:,:,1], cmap='gray')
        plt.subplot(244)
        plt.title('CH3')
        plt.imshow(feature_image[:,:,2], cmap='gray')
        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hfeatures, hogimage = get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=True, feature_vec=False)
                hog_features.append(hfeatures)
                plt.subplot(2,4 , 6 + channel)
                plt.title('HOG1')
                plt.imshow(hogimage, cmap='gray')
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)

        if debug:
          plt.show()
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        
        hist_features = color_hist(feature_image, nbins=hist_bins)
        
        # Append the new feature vector to the features list
        features.append(np.concatenate((hog_features, spatial_features, hist_features)))

    # Return list of feature vectors
    return features

def classify():
  file = 'svc-yuv.p'
  if Path(file).is_file():
    try:
      dist_pickle = pickle.load( open(file, 'rb'))
      if (len(dist_pickle) > 0):
        return dist_pickle["svc"], dist_pickle["orient"], dist_pickle["pix_per_cell"], dist_pickle["cell_per_block"], dist_pickle["colorspace"], dist_pickle["scaler"], dist_pickle["hist_bins"], dist_pickle["spatial_size"]
      else:
        print('Empty pickle? hmmm.. repickle dat pickle!')
    except EOFError:
      print("Error reading cal data")
  ### TODO: Tweak these parameters and see how the results change.
  colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
  orient = 14
  pix_per_cell = 8
  cell_per_block = 2
  hist_bins = 40
  spatial_size = (32,32)
  hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

  t=time.time()
  cars = glob.glob('vehicles/**/*.png', recursive=True)
  notcars = glob.glob('non-vehicles/**/*.png', recursive=True)
  minfeatures = min(len(cars), len(notcars))
  cars = cars[0:minfeatures]
  notcars = notcars[0:minfeatures]
  print('features: {}'.format(minfeatures))
  notcar_features = extract_features(notcars, spatial_size=spatial_size, hist_bins=hist_bins, cspace=colorspace, orient=orient, 
                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                          hog_channel=hog_channel)
  car_features = extract_features(cars,spatial_size=spatial_size, hist_bins=hist_bins,  cspace=colorspace, orient=orient, 
                          pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
                          hog_channel=hog_channel)


  print('Got {} Features. Training'.format(len(car_features)))
  X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
  # Fit a per-column scaler
  X_scaler = StandardScaler().fit(X)
  # Apply the scaler to X
  scaled_X = X_scaler.transform(X)

  # Define the labels vector
  y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


  # Split up data into randomized training and test sets
  rand_state = np.random.randint(0, 100)
  X_train, X_test, y_train, y_test = train_test_split(
      scaled_X, y, test_size=0.2, random_state=rand_state)

  print('Using:',orient,'orientations',pix_per_cell,
      'pixels per cell and', cell_per_block,'cells per block')
  print('Feature vector length:', len(X_train[0]))
  # Use a linear SVC 
  svc = LinearSVC()
  # Check the training time for the SVC
  t=time.time()
  svc.fit(X_train, y_train)
  print('SVC Score: {}'.format(svc.score(X_test, y_test)))
  dist_pickle = {}
  dist_pickle["svc"]= svc
  dist_pickle["orient"] = orient
  dist_pickle["pix_per_cell"] = pix_per_cell
  dist_pickle["cell_per_block"] = cell_per_block
  dist_pickle["colorspace"] = colorspace
  dist_pickle["scaler"] = X_scaler
  dist_pickle["hist_bins"] = hist_bins
  dist_pickle["spatial_size"] = spatial_size
  pickle.dump( dist_pickle, open( file, "wb" ))
 
def find_cars(img, ystart, ystop, svc, scale, X_scaler, cspace, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
  bboxes = []
  draw_img = np.copy(img)
  img = img.astype(np.float32)/255

  img_tosearch = img[ystart:ystop,:,:]
  if cspace != 'RGB':
    if cspace == 'HSV':
      feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HSV)
    elif cspace == 'LUV':
      feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2LUV)
    elif cspace == 'HLS':
      feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2HLS)
    elif cspace == 'YUV':
      feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YUV)
    elif cspace == 'YCrCb':
      feature_image = cv2.cvtColor(img_tosearch, cv2.COLOR_RGB2YCrCb)
  else: feature_image = np.copy(img_tosearch)
      
  if scale != 1:
    imshape = feature_image.shape
    feature_image = cv2.resize(feature_image, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

  ch1 = feature_image[:,:,0]
  ch2 = feature_image[:,:,1]
  ch3 = feature_image[:,:,2]

  # Define blocks and steps as above
  nxblocks = (ch1.shape[1] // pix_per_cell)-1
  nyblocks = (ch1.shape[0] // pix_per_cell)-1 
  nfeat_per_block = orient*cell_per_block**2
  # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
  window = 64
  nblocks_per_window = (window // pix_per_cell)-1 
  cells_per_step = 2  # Instead of overlap, define how many cells to step
  nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
  nysteps = (nyblocks - nblocks_per_window) // cells_per_step

  # Compute individual channel HOG features for the entire image
  hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
  hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
  hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

  for xb in range(nxsteps):
    for yb in range(nysteps):
      ypos = yb*cells_per_step
      xpos = xb*cells_per_step
      # Extract HOG for this patch
      hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
      hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
      hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
      hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

      xleft = xpos*pix_per_cell
      ytop = ypos*pix_per_cell

      # Extract the image patch
      subimg = cv2.resize(feature_image[ytop:ytop+window, xleft:xleft+window], (64,64))
      spatial_features = bin_spatial(subimg, size=spatial_size)
      hist_features = color_hist(subimg, nbins=hist_bins)

      # Scale features and make a prediction
      test_features = X_scaler.transform(np.hstack((hog_features, spatial_features, hist_features)).reshape(1, -1))    
      test_prediction = svc.predict(test_features)
      xbox_left = np.int(xleft*scale)
      ytop_draw = np.int(ytop*scale)
      win_draw = np.int(window*scale)

      if test_prediction == 1:
        bboxes.append(( (xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)  ))
        # cv2.rectangle(draw_img,(xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart),(0,255,0),3)


  # if debug:
    # plt.imshow(draw_img)
    # plt.show()

  return bboxes, draw_img


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(image, labels):
    # Iterate through all detected cars
    img = np.copy(image)
    boxes = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        boxes.append(bbox)
        print("Drawing rectangle : {}".format(bbox))
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img, boxes

class Processor():
  def __init__(self, n):
    self.svc, self.orient, self.pix_per_cell, self.cell_per_block, self.colorspace, self.X_scaler, self.hist_bins, self.spatial_size = classify()
    self.boxwindow = []
    self.n = n

  def generateHeatMap(self, img):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    if(len(self.boxwindow) > self.n):
      self.boxwindow = self.boxwindow[1:]
    for frame in self.boxwindow:
      for box in frame:
        heat = add_heat(heat, box)
    return heat

  def processImage(self, img):
    scales = [(1.,360), (1.5,400), (1.8,400)]
    frameboxes = []
    fig = plt.figure()

    i = 1
    for scale,ystart in scales:
      boxes, modimg = find_cars(img, ystart, img.shape[0], self.svc, scale, self.X_scaler, self.colorspace, self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins)
      frameboxes.append(boxes)
      # plt.subplot(2,3,i)
      # plt.axis('off')

      # i += 1
      # # plt.imshow(modimg)
      # plt.title('Scale {}'.format(scale))
    
    self.boxwindow.append(frameboxes)
    heat = self.generateHeatMap(img)

    heat = apply_threshold(heat, 2 * min(self.n, len(self.boxwindow)))
    heatmap = np.clip(heat, 0, 255)
    if debug:
      plt.subplot(234)
      plt.axis('off')
      plt.imshow(heat, cmap='gray')
      plt.title('Heatmap')
  # Find final boxes from heatmap using label function
    labels = label(heatmap)
    image, boxesdrawn = draw_labeled_bboxes(img, labels)
    print("Num cars: {}, boxes: {}".format(labels[1],boxesdrawn))
    
    if debug:
      plt.subplot(235)
      plt.axis('off')  
      plt.imshow(image)
      plt.title('Bounding Boxes')
      fig.tight_layout()
      plt.show()
    return image    

def main():
  classify()
  processor = Processor(10)
  clip = VideoFileClip('project_video.mp4')
  out_clip = clip.fl_image(processor.processImage)
  out_clip.write_videofile('project_output_yuv.mp4',audio=False)

if __name__ == "__main__":
  main()
