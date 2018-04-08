#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import random
import math
# %matplotlib inline

# reading in an image
image = mpimg.imread('test_images/solidWhiteRight.jpg')

# printing out some stats and plotting
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')




def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=4):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
  # print(lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.


def single_line_slope(lines):
    # slope = (line[3]-line[1])/(line[2]-line[0])
    slope_all = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1)/(x2 - x1)
            slope_all.append(slope)
          # print(x1, y1, x2, y2)
    return slope_all


# 函数测试
# points = np.array([[[1, 3, 3, 4]], [[5, 8, 7, 9]]])
# a = single_line_slope(points)
# print("This ia a test，a = ", a)

def GetGeneralEquation(x1, y1, x2, y2):
    # A*x + B*y + c = 0
    A = y2 - y1
    B = x1 - x2
    C = y1 * x2 - x1 * y2
    return A, B, C


def GetIntersectionPoint(x1, y1, x2, y2, x3, y3, x4, y4):
    # A1*x + B1*y + C1 =0;
    # A2*x + B2*y + C2 = 0
    A1, B1, C1 = GetGeneralEquation(x1, y1, x2, y2)
    A2, B2, C2 = GetGeneralEquation(x3, y3, x4, y4)
    m = A1 * B2 - A2 * B1
    if m == 0:
        print("无交点")
    else:
        x0 = (C2 * B1 - C1 * B2)/m
        y0 = (C1 * A2 - C2 * A1)/m
    return x0, y0


def lines_average(img, rho, theta, threshold, min_line_len, max_line_gap):
    height = img.shape[0]
    positive_index = []
    positive_slope = []
    negtive_index = []
    negtive_slope = []

    # Get Hough Lines, namley segment end points
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    print(lines.shape)
    # calculate slope of each segment, and store in the slopes
    slopes = single_line_slope(lines)
    print(slopes)
    # To get positive and negtive slopes and index of lines
    n = len(slopes)

    for i in range(n):
        if slopes[i] > 0:
            positive_index.append(i)
            positive_slope.append(slopes[i])
        else:
            negtive_index.append(i)
            negtive_slope.append(slopes[i])
    # print("negtive index is : ", negtive_index)
    # print("positive index is: ", positive_index)

    # to get the random two sengments of each positive and negtive line set
    random_num_neg = random.sample(range(len(negtive_index)), 2)
    random_num_pos = random.sample(range(len(positive_index)), 2)

    # print("random number negtive is :", random_num_neg)
    # print("radom number positive is :", random_num_pos)

    # to define the end points needed to calculate the intersection points
    # Get 1st pair of left points

    x1_Left = lines[negtive_index[random_num_neg[0]], 0, 0]
    y1_Left = lines[negtive_index[random_num_neg[0]], 0, 1]
    x2_Left = lines[negtive_index[random_num_neg[0]], 0, 2]
    y2_Left = lines[negtive_index[random_num_neg[0]], 0, 3]

    # Get 2nd pair of left points
    x3_Left = lines[negtive_index[random_num_neg[1]], 0, 0]
    y3_Left = lines[negtive_index[random_num_neg[1]], 0, 1]
    x4_Left = lines[negtive_index[random_num_neg[1]], 0, 2]
    y4_Left = lines[negtive_index[random_num_neg[1]], 0, 3]

    # Get 1st pair of left points
    x1_Righ = lines[positive_index[random_num_pos[0]], 0, 0]
    y1_Righ = lines[positive_index[random_num_pos[0]], 0, 1]
    x2_Righ = lines[positive_index[random_num_pos[0]], 0, 2]
    y2_Righ = lines[positive_index[random_num_pos[0]], 0, 3]

    # Get 2nd pair of left points
    x3_Righ = lines[positive_index[random_num_pos[1]], 0, 0]
    y3_Righ = lines[positive_index[random_num_pos[1]], 0, 1]
    x4_Righ = lines[positive_index[random_num_pos[1]], 0, 2]
    y4_Righ = lines[positive_index[random_num_pos[1]], 0, 3]

    # to get the intersection points
    x_intersec1, y_intersec1 = GetIntersectionPoint(x1_Left, y1_Left, x2_Left, y2_Left, x1_Righ, y1_Righ, x2_Righ, y2_Righ)
    x_intersec2, y_intersec2 = GetIntersectionPoint(x3_Left, y3_Left, x4_Left, y4_Left, x3_Righ, y3_Righ, x4_Righ, y4_Righ)

    # to get the middle point of intersection points
    x_m = (x_intersec1 + x_intersec2)/2
    y_m = (y_intersec1 + y_intersec2)/2

    # calculate the average slope of left lane and right lane
    slope_left = np.mean(positive_slope)
    slope_right = np.mean(negtive_slope)

    # to get the end points of average lines
    y_start_left = height
    x_start_left = (height - y_m)/slope_left + x_m
    y_start_right = height
    x_start_right = (height - y_m)/slope_right + x_m

    x_endpoint_left = int(0.1 * x_start_left + 0.9 * x_m)
    y_endpoint_left = int((x_endpoint_left - x_m) * slope_left + y_m)
    x_endpoint_right = int(0.1 * x_start_right + 0.9 * x_m)
    y_endpoint_right = int((x_endpoint_right - x_m) * slope_right + y_m)

    x_start_left = int(x_start_left)
    x_start_right = int(x_start_right)

    # print(x_start_left, y_start_left, x_endpoint_left, y_endpoint_left, x_start_right, y_start_right, x_endpoint_right, y_endpoint_right)
    # draw average lines

    endpoints = np.array([[[x_start_left, y_start_left, x_endpoint_left, y_endpoint_left]], [[x_start_right, y_start_right, x_endpoint_right, y_endpoint_right]]])
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, endpoints)
    return line_img


def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + γ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)


import os
os.listdir("test_images/")

# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images_output directory.

# Define vertices of ROI Polygon Region and Get the ROI
# to decrease calculation amount of Canny edge detection, I chose to select ROI first, but this caused new edges.
vertices = np.array([[(120, 540), (420, 330),(550, 330), (900, 540)]], dtype=np.int32)
region_selection = region_of_interest(image, vertices)
plt.imshow(region_selection)
# plt.show()

# Grayscalling of ROI
grayscaled_image = grayscale(region_selection)
plt.imshow(grayscaled_image, cmap='gray')
# plt.show()

# Gaussian Smoothing
Gray_blur = gaussian_blur(grayscaled_image, 5)
plt.imshow(Gray_blur, cmap='gray')
# plt.show()

# Canny Edge detection
low_threshold = 80
high_threshold = 180
Edges = canny(Gray_blur, low_threshold , high_threshold)
plt.imshow(Edges, cmap='gray')
# plt.show()

# update smaller vertices of ROI Polygon Region and Get the final ROI
vertices = np.array([[(140, 540), (430, 340), (540, 340), (880, 540)]], dtype=np.int32)
Final_region_selection = region_of_interest(Edges, vertices)

plt.imshow(Final_region_selection, cmap='gray')
# plt.show()

# print(Final_region_selection)

# plot the hough lines in the edge image
rho = 2
theta = np.pi/180
threshold = 15
min_line_len = 40
max_line_gap = 20
Lines_edges = hough_lines(Final_region_selection, rho, theta, threshold, min_line_len, max_line_gap)
plt.imshow(Lines_edges)
# plt.show()


rho = 2
theta = np.pi/180
threshold = 15
min_line_len = 30
max_line_gap = 5
Lines_average = lines_average(Final_region_selection, rho, theta, threshold, min_line_len, max_line_gap)
plt.imshow(Lines_average)
plt.show()

image_final = weighted_img(Lines_average, image, α=0.8, β=1.0, γ=0)
plt.imshow(image_final)
# plt.show()

# 
# *********************************************************************************************************************#
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML


def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image where lines are drawn on lanes)

    vertices1 = np.array([[(120, 540), (420, 330),(550, 330), (900, 540)]], dtype=np.int32)
    region_selection1 = region_of_interest(image, vertices1)
    grayscaled_image1 = grayscale(region_selection1)
    Gray_blur1 = gaussian_blur(grayscaled_image1, 5)
    low_threshold1 = 80
    high_threshold1 = 180
    Edges1 = canny(Gray_blur1, low_threshold1, high_threshold1)
    vertices_update1 = np.array([[(130, 540), (430, 340),(540, 340), (890, 540)]], dtype=np.int32)
    Final_region_selection1 = region_of_interest(Edges1, vertices_update1)

    rho1 = 2
    theta1 = np.pi/180
    threshold1 = 15
    min_line_len1 = 30
    max_line_gap1 = 8
    Lines_average1 = lines_average(Final_region_selection1, rho1, theta1, threshold1, min_line_len1, max_line_gap1)
    image_final1 = weighted_img(Lines_average1, image, α=0.8, β=1.0, γ=0)
    return image_final1


white_output = 'test_videos_output/solidWhiteRight.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4").subclip(0,5)
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
# %time white_clip.write_videofile(white_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(white_output))

yellow_output = 'test_videos_output/solidYellowLeft.mp4'
## To speed up the testing process you may want to try your pipeline on a shorter subclip of the video
## To do so add .subclip(start_second,end_second) to the end of the line below
## Where start_second and end_second are integer values representing the start and end of the subclip
## You may also uncomment the following line for a subclip of the first 5 seconds
##clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4').subclip(0,5)
clip2 = VideoFileClip('test_videos/solidYellowLeft.mp4')
yellow_clip = clip2.fl_image(process_image)
# %time yellow_clip.write_videofile(yellow_output, audio=False)

HTML("""
<video width="960" height="540" controls>
  <source src="{0}">
</video>
""".format(yellow_output))