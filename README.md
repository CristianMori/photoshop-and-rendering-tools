# photoshop-and-rendering-tools
CS 445: Computational Photography Projects

# Programming Project #1: Hybrid Images
Part I: Hybrid Images
This part of the project is intended to familiarize you with image filtering and frequency representations. The goal is to create hybrid images using the approach described in the SIGGRAPH 2006 paper by Oliva, Torralba, and Schyns. Hybrid images are static images that change in interpretation as a function of the viewing distance. The basic idea is that high frequency tends to dominate perception when it is available, but, at a distance, only the low frequency (smooth) part of the signal can be seen. By blending the high frequency portion of one image with the low-frequency portion of another, you get a hybrid image that leads to different interpretations at different distances.

Here, I have included two sample images (of me and my former cat Nutmeg) and some starter code that can be used to load two images and align them. The alignment is important because it affects the perceptual grouping (read the paper for details).

First, you'll need to get a few pairs of images that you want to make into hybrid images. You can use the sample images for debugging, but you should use your own images in your results. Then, you will need to write code to low-pass filter one image, high-pass filter the second image, and add (or average) the two images. For a low-pass filter, Oliva et al. suggest using a standard 2D Gaussian filter. For a high-pass filter, they suggest using the impulse filter minus the Gaussian filter (which can be computed by subtracting the Gaussian-filtered image from the original). The cutoff-frequency of each filter should be chosen with some experimentation (or equivalently choose the sigma/width of the Gaussian). The starter package also includes a gaussian_kernel function in utils.py to save you some time. Don't just use the gaussian_filter function in scipy! Use the result of gaussian_kernel as input to a general filter/convolution function.

For your favorite result, you should also illustrate the process through frequency analysis. Show the log magnitude of the Fourier transform of the two input images, the filtered images, and the hybrid image. In Python, you can compute and display the 2D Fourier transform using Matplotlib and Numpy with: plt.imshow(numpy.log(numpy.abs(numpy.fft.fftshift(numpy.fft.fft2(gray_image)))))

Try creating a variety of types of hybrid images (change of expression, morph between different objects, change over time, etc.). The site has several examples that may inspire.

Part II: Image Enhancement
You may sometimes find that your photographs do not quite have the vivid colors or contrast that you remember seeing. In this part of the project, we'll look at three simple types of enhancement. You can do two out of three of these. The third is worth 10 pts as a bells and whistle.

Contrast Enhancement: The goal is to improve the contrast of the images. The poor constrast could be due to blurring in the capture process or due to the intensities not covering the full range. Choose an image (ideally one of yours, but from web is ok) that has poor contrast and fix the problem. Potential fixes include Laplacian filtering, gamma correction, and histogram equalization. Explain why you chose your solution.

Color Enhancement: Now, how to make the colors brighter? You'll find that if you just add some constant to all of the pixel values or multiply them by some factor, you'll make the images lighter, but the colors won't be more vivid. The trick is to work in the correct color space. Convert the images to HSV color space and divide into hue, saturation, and value channels (hsv = cv2.cvtColor(im,cv2.COLOR_BGR2HSV) in OpencCv). Then manipulate the appropriate channel(s) to make the colors (but not the intensity) brighter. Note that you want the values to map between the range defined by the imported library (in OpenCv 0-255), so you shouldn't just add or multiply with some constant. Show this with at least one photograph. Show the original and enhanced images and explain your method.

Color Shift: Take an image of your choice and create two color-modified versions that are (a) more red; (b) less yellow. Show the original and two modified images and explain how you did it and what color space you've used. Note that you should not change the luminance of the photograph (i.e., don't make it more red just by increasing the values of the red channel). In OpenCv use cv2.cvtColor(image, cv2.COLOR_BGR2Lab) for converting between RGB and LAB spaces, in case you want to use LAB space.

Bells & Whistles (Extra Points)
Try using color to enhance the effect of hybrid images. Does it work better to use color for the high-frequency component, the low-frequency component, or both? (5 pts)
Illustrate the hybrid image process by implementing Gaussian and Laplacian pyramids and displaying them for your favorite result. This should look similar to Figure 7 in the Oliva et al. paper. (15 pts)
Do all three image enhancement tasks. (10 pts)

# Programming Project #2: Image Quilting
Overview
The goal of this assignment is to implement the image quilting algorithm for texture synthesis and transfer, described in this SIGGRAPH 2001 paper by Efros and Freeman. Texture synthesis is the creation of a larger texture image from a small sample. Texture transfer is giving an object the appearance of having the same texture as a sample while preserving its basic shape (see the face on toast image above). For texture synthesis, the main idea is to sample patches and lay them down in overlapping patterns, such that the overlapping regions are similar. The overlapping regions may not match exactly, which will result in noticeable edges. To fix this, you will compute a path along pixels with similar intensities through the overlapping region and use it to select which overlapping patch from which to draw each pixel. Texture transfer is achieved by encouraging sampled patches to have similar appearance to a given target image, as well as matching overlapping regions of already sampled patches. In this project, you will apply important techniques such as template matching, finding seams, and masking. These techniques are also useful for image stitching, image completion, image retargeting, and blending.

Here, I have included some sample textures to get you started (these images are taken from the paper). You will implement the project in several steps.


Randomly Sampled Texture (10 pts)
Create a function quilt_random(sample, out_size, patch_size) that randomly samples square patches of size patch_size from a sample in order to create an output image of size out_size. Start from the upper-left corner, and tile samples until the image are full. If the patches don't fit evenly into the output image, you can leave black borders at the edges. This is the simplest but least effective method. Save a result from a sample image to compare to the next two methods.


Overlapping Patches (30 pts)
Create a function quilt_simple(sample, out_size, patch_size, overlap, tol) that randomly samples square patches of size patch_size from a sample in order to create an output image of size out_size. Start by sampling a random patch for the upper-left corner. Then sample new patches to overlap with existing ones. For example, the second patch along the top row will overlap by patch_size pixels in the vertical direction and overlap pixels in the horizontal direction. Patches in the first column will overlap by patch_size pixels in the horizontal direction and overlap pixels in the vertical direction. Other patches will have two overlapping regions (on the top and left) which should both be taken into account. Once the cost of each patch has been computed, randomly choose on patch whose cost is less than a threshold determined by tol (see description of choose_sample below).

I suggest that you create two helper functions ssd_patch and choose_sample. ssd_patch performs template matching with the overlapping region, computing the cost of sampling each patch, based on the sum of squared differences (SSD) of the overlapping regions of the existing and sampled patch. I suggest using a masked template. The template is the patch in the current output image that is to be filled in (many pixel values will be 0 because they are not filled in yet). The mask has the same size as the patch template and has values of 1 in the overlapping region and values of 0 elsewhere. The SSD of the masked template with the input texture image can be computed efficiently using filtering operations (see tips section down below), producing an image in which the output is the overlap cost (SSD) of choosing a sample centered at each pixel.

choose_sample should take as input a cost image (each pixel's value is the cost of selecting the patch centered at that pixel) and select a randomly sampled patch with low cost, as described in the paper. One way to do this is to first find the minimum cost minc and then to sample a patch within a percentage of that value:
row, col = np.where(cost < minc*(1+tol)) . If the minimum is approximately zero (which can happen initially), it might make sense to set minc to a larger value, e.g., minc=max(minc,small_cost_value);. Another way is to sample one of the K lowest-cost patches.

After a patch is sampled, its pixels should be copied directly into the corresponding position in the output image. Note that it is very easy to make alignment mistakes when computing the cost of each patch, sampling a low-cost patch, and copying the patch from the source to the output. Use an odd value for patch_size so that its center is well-defined. Be sure to thoroughly debug, for example, by checking that the overlapping portion of the copied pixels has the same SSD as the originally computed cost. As a sanity check, try generating a small texture image with low tolerance (e.g., 0.00001), with the first patch sampled from the upper-left of the source image. This should produce a partial copy of the source image. Once you have this function working, save a result (with higher tolerance for more stochastic texture) generated from the same sample as used for the random method.


Seam Finding (20 pts)
Next, incorporate seam finding to remove edge artifacts from the overlapping patches (section 2.1 of the paper):

Use the cut function in utils.py (download starter_codes at the top), or, if you want a challenge and 10 bonus points, create your own function cut(bndcost) that finds the min-cost contiguous path from the left to right side of the patch according to the cost indicated by bndcost. The cost of a path through each pixel is the square differences (summed over RGB for color images) of the output image and the newly sampled patch. Use dynamic programming to find the min-cost path. Use this path to define a binary mask that specifies which pixels to copy from the newly sampled patch. Note that if a patch has top and left overlaps, you will need to compute two seams, and the mask can be defined as the intersection of the masks for each seam (mask1&mask2). To find a vertical path, you can apply cut to the transposed patch, e.g., cut(bndcost.T).T. If you do use the included function, take the time to understand it.

Create a function quilt_cut that incorporates the seam finding and use it to create a result to compare to the previous two methods.

Texture Transfer (30 pts)
Your final task is to create a function texture_transfer, based on your quilt_cut function for creating a texture sample that is guided by a pair of sample/target correspondence images (section 3 of the paper). You do not need to implement the iterative method described in the paper (you can do so for extra points: see Bells and Whistles). The main difference between this function and quilt_cut is that there is an additional cost term based on the difference between the sampled source patch and the target patch at the location to be filled.


Bells & Whistles
(10 pts) Create and use your own version of cut function. To get these points, you should create your own implementation without basing it directly on the provided function (you're on the honor code for this one).
(15 pts) Implement the iterative texture transfer method described in the paper. Compare to the non-iterative method for two examples.
(up to 20 pts) Use a combination of texture transfer and blending to create a face-in-toast image like the one on top. To get full points, you must use some type of blending, such as feathering or Laplacian pyramid blending.
(up to 40 pts) Extend your method to fill holes of arbitrary shape for image completion. In this case, patches are drawn from other parts of the target image. For the full 40 pts, you should implement a smart priority function (e.g., similar to Criminisi et al.).

# Programming Project #3: Gradient-Domain Fusion
Overview
This project explores gradient-domain processing, a simple technique with a broad set of applications including blending, tone-mapping, and non-photorealistic rendering. For the core project, we will focus on "Poisson blending"; tone-mapping and NPR can be investigated as bells and whistles.

The primary goal of this assignment is to seamlessly blend an object or texture from a source image into a target image. The simplest method would be to just copy and paste the pixels from one image directly into the other. Unfortunately, this will create very noticeable seams, even if the backgrounds are well-matched. How can we get rid of these seams without doing too much perceptual damage to the source region?

The insight is that people often care much more about the gradient of an image than the overall intensity. So we can set up the problem as finding values for the target pixels that maximally preserve the gradient of the source region without changing any of the background pixels. Note that we are making a deliberate decision here to ignore the overall intensity! So a green hat could turn red, but it will still look like a hat.

We can formulate our objective as a least squares problem. Given the pixel intensities of the source image "s" and of the target image "t", we want to solve for new intensity values "v" within the source region "S":

Here, each "i" is a pixel in the source region "S", and each "j" is a 4-neighbor of "i". Each summation guides the gradient values to match those of the source region. In the first summation, the gradient is over two variable pixels; in the second, one pixel is variable and one is in the fixed target region.

The method presented above is called "Poisson blending". Check out the Perez et al. 2003 paper to see sample results, or to wallow in extraneous math. This is just one example of a more general set of gradient-domain processing techniques. The general idea is to create an image by solving for specified pixel intensities and gradients.

Toy Problem (20 pts)
The implementation for gradient domain processing is not complicated, but it is easy to make a mistake, so let's start with a toy example. Reconstruct this image from its gradient values, plus one pixel intensity. Denote the intensity of the source image at (x, y) as s(x,y) and the value to solve for as v(x,y). For each pixel, then, we have two objectives:
1. minimize (v(x+1,y)-v(x,y) - (s(x+1,y)-s(x,y)))^2
2. minimize (v(x,y+1)-v(x,y) - (s(x,y+1)-s(x,y)))^2
Note that these could be solved while adding any constant value to v, so we will add one more objective:
3. minimize (v(1,1)-s(1,1))^2
For 20 points, solve this in Python as a least squares problem. If your solution is correct, then you should recover the original image.

Implementation Details

The first step is to write the objective function as a set of least squares constraints in the standard matrix form: (Av-b)^2. Here, "A" is a sparse matrix, "v" are the variables to be solved, and "b" is a known vector. It is helpful to keep a matrix "im2var" that maps each pixel to a variable number, such as:
im_h, im_w = im.shape
im2var = np.arange(im_h * im_w).reshape(im_w, im_h).T

Then, you can write objective 1 above as:
e = e + 1;
A[e][im2var[y][x+1]] = 1
A[e][im2var[y][x]] = -1
b[e] = im[y][x+1] - im[y][x]
Here, "e" is used as an equation counter. Note that the y-coordinate is the first index. As another example, objective 3 above can be written as:
e = e + 1;
A[e][im2var[0][0]] = 1
b[e] = s[0][0]

To solve for v, use v = np.linalg.solve(A, b); Then, copy each solved value to the appropriate pixel in the output image.

Poisson Blending (50 pts)
Step 1: Select source and target regions. Select the boundaries of a region in the source image and specify a location in the target image where it should be blended. Then, transform (e.g., translate) the source image so that indices of pixels in the source and target regions correspond. I've provided starter code (getMask.m, alignSource.m) to help with this. You may want to augment the code to allow rotation or resizing into the target region. You can be a bit sloppy about selecting the source region -- just make sure that the entire object is contained. Ideally, the background of the object in the source region and the surrounding area of the target region will be of similar color.

Step 2: Solve the blending constraints:


Step 3: Copy the solves values into your target image. For RGB images, process each channel separately. Show at least three results of Poisson blending. Explain any failure cases (e.g., weird colors, blurred boundaries, etc.).

Tips

1. Consider to use sparse matrix(scipy.sparse.csr_matrix) to save space and speed up computation
2. When solving a set of linear equations, considering which function to choose if the matrix rank is deficient: 1. numpy.linalg.solve 2. scipy.sparse.linalg.lstsq
3. Before trying new examples, try something that you know should work, such as the included penguins on top of the snow in the hiking image.
4. Object region selection can be done very crudely, with lots of room around the object.
5. The default color space for opencv is BGR. It might be easier for you to first convert it to RGB to avoid weird coloring.

Mixed Gradients (20 pts)
Follow the same steps as Poisson blending, but use the gradient in source or target with the larger magnitude as the guide, rather than the source gradient:


Here "d_ij" is the value of the gradient from the source or the target image with larger magnitude. Note that larger magnitude is not the same as greater value. For example, if the two gradients are -0.6 and 0.4, you want to keep the gradient of -0.6. Show at least one result of blending using mixed gradients. One possibility is to blend a picture of writing on a plain background onto another image.

Bells & Whistles (Extra Points)
Color2Gray (20 pts)
Sometimes, in converting a color image to grayscale (e.g., when printing to a laser printer), we lose the important contrast information, making the image difficult to understand. For example, compare the color version of the image on right with its grayscale version produced by rgb2gray.
Can you do better than rgb2gray? Gradient-domain processing provides one avenue: create a gray image that has similar intensity to the rgb2gray output but has similar gradients to the original RGB image. This is an example of a tone-mapping problem, conceptually similar to that of converting HDR images to RGB displays. For your solution, use only the RGB space (e.g., don't convert to Lab or HSV). Test your solution on colorBlind8.png and colorBlind4.png, included with the sample images. Your code should be in PDF not on the website. Hint: your solution may be a combination of the toy problem and mixed gradients.

Laplacian pyramid blending (20 pts)
Another technique for blending is to decompose the source and target images using a laplacian pyramid and to combine using alpha mattes. For the low frequencies, there should be a slow transition in the alpha matte from 0 to 1; for the high frequencies, the transition should be sharp. Try this method on some of your earlier results and compare to the Poisson blending.

More gradient domain processing (up to 20 pts)
Many other applications are possible, including non-photorealistic rendering, edge enhancement, and texture or color transfer. See Perez et al. 2003 or Gradient Shop for further ideas.
