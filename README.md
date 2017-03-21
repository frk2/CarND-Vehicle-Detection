# Vehicle Detection Project

We learned how to train a HOG classifier, do extraction using it and create a labeled training set using heatmaps which then is able to classify objects reasonably well.

[//]: # (Image References)
[image1]: ./output_images/car-train-yuv.png
[image2]: ./output_images/not-car-train-yuv.png
[image3]: ./output_images/allwindows.png
[image4]: ./output_images/scalingwindow.png
[image5]: ./output_images/heatmap.png
[image6]: ./output_images/allboxes.png
[video1]: ./project_output_yuv.mp4

### Training

`classify.py` is the one source of all the functions. It checks if a pickle file is on disk, if not it starts the training process using images from the cars and not cars dataset.

I started by reading in all the `vehicle` and `non-vehicle` images in YUV and extracting HOG features from them. Here's what that looks like broken down by each of the three channels:

![image1]
![image2]

I explored multiple parameters for HOG (higher orientations mean higher number of features!) and different colorspaces. It was soon clear that YUV/YCrCb was the most effective in all of these. So I decided to carry forward with simple YUV.

#### HOG Parameters
The only parameter that made a huge difference was the numer of orientations. Other than that the pix per cell / cell per block I left pretty much alone. I would've wanted to change them around but want to submit this project on time hence was unable to.

#### SVM
I used a combination of features (HOG, spatial binning and color histograms) and combined them using a scaler to be fed into a linear SVC:
```
svc = LinearSVC()
svc.fit(X_train, y_train)
```

20% of the data was retained for testing using `train_test_split`

all of this was written to a pickle file so training could only be done once.

### Sliding Window Search
Sliding window search was done at varying scales and from varying starting y positions. I chose the following `scale, y-start` tuples:
```
scales = [(1.,360), (1.5,400), (1.8,400)]
```
The actual search was done using HOG window subsampling code which is able to extract HOG features only once per image. To make sure the search was working as expected I made it draw windows all over the image for debugging:

![image3]

For each of the scales, I used the bounding boxes to generate a heat map. This is visible from the collection below:
![image4]
And here is the above heatmap is higher resolution:
![image5]

And here is some debug output with the raw boxes (in green) show on each car. The heatmaps are used to generate the labeled boxes shown in blue to form a bounding box:

![image6]

#### Optimizations
The detection of the boxes was quite janky in the begging with enough false positives to give me concern. I used a moving window for the heatmaps with a higher threshold that helped a lot is reducing jitter in the boxes and eliminating false positives. The current implementation constructs a heatmap from the last 10 frames (which is configurable)

### Video Implementation

Here's a [link to my video result](./project_output_yuv.mp4)


### Discussion
My HOG based algorithm is good but not flawless. Even though its highly accurate in a test scenario - it still generates an alarming number of false positives. I would probably switch to a neural net for image recognition in this scenario.

Some sort of motion tracking can probably make this more robust. For example we know where the cars are headed which we can use to make an educated guess of were they might end up in the next frame. 

Combining lane detection (so that we can mark the end of the road) would also help since it allows us to eliminate detecting cars on the other side of the wall.
