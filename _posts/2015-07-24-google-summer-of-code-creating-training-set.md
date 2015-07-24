---
layout: post
title: "Google Summer of Code: Creating Training set."
description:
comments: true
categories:
- gsoc
---


I describe a process of the creating a dataset for training classifier that I use for Face Detection.

___

## Positive samples (Faces).

For this task I decided to take the [Web Faces database](www.vision.caltech.edu/Image_Datasets/Caltech_10K_WebFaces/). It consists of `10000` faces. Each face has eye coordinates which is very
useful, because we can use this information to align faces.

Why do we need to align faces?
Take a look at this photo:

![Not aligned face]({{ site.url }}/assets/img/not_aligned_face.jpg)

If we just crop the faces as they are, it will be really hard for classifier to learn from it.
The reason for this is that we don't know how all of the faces in the database are positioned.
Like in the example above the face is rotated.
In order to get a good dataset we first align faces and then add small random transformations
that we can control ourselves. This is really convinient because if the training goes bad,
we can just change the parameters of the random transformations and experiment.

In order to align faces, we take the coordinates of eyes and draw a line through them.
Then we just rotate the image in order to make this line horizontal. Before running
the script the size of resulted images is specified and the amount of the area above and
below the eyes, and on the right and the left side of a face. The cropping also takes care
of the proportion ratio. Otherwise, if we blindly `resize` the image the resulted face will
be spoiled and the classifier will work bad. That way we can be sure now that all our faces
are placed cosistently and we can start to run random transformations. The idea that I described
was taken from the [following page](http://www.bytefish.de/blog/aligning_face_images/).

Have a look at the aligned faces:

![Aligned face one]({{ site.url }}/assets/img/aligned_face_one.jpg)
![Aligned face two]({{ site.url }}/assets/img/aligned_face_two.jpg)
![Aligned face three]({{ site.url }}/assets/img/aligned_face_three.jpg)

As you see the amount of area is consistent across images. The next stage is to transform them
in order to augment our dataset. For this purpose we will use `OpenCv` [create_samples utility](http://docs.opencv.org/doc/user_guide/ug_traincascade.html). This utility takes all the images and creates new
images by randomly transforming the images and changing the intensity in a specified manner. For my purposes I have chosen the following parameters `-maxxangle 0.5 -maxyangle 0.5 -maxzangle 0.3 -maxidev 40`. The angles specify the maximum rotation angles in `3d` and the `maxidev` specifies the maximum deviation in the intesity changes. This script also puts images on the specified by user background.

This process is really complicated if you want to extract images in the end and not the `.vec` file
format of the `OpenCv`.

This is a small description on how to do it:

1. Run the bash command `find ./positive_images -iname "*.jpg" > positives.txt` to get a list of
   positive examples. `positive_images` is a folder with positive examples.
2. Same for the negative `find ./negative_images -iname "*.jpg" > negatives.txt`.
3. Run the `createtrainsamples.pl` file like this 
   `perl createtrainsamples.pl positives.txt negatives.txt vec_storage_tmp_dir`. Internally
   it uses `opencv_createsamples`. So you have to have it compiled. It will create a lot of
   `.vec` files in the specified directory. You can get this script from [here](http://note.sonots.com/SciSoftware/haartraining.html#w0a08ab4). This command transforms each image in the `positives.txt` and places the results as `.vec` files in the `vec_storage_tmp_dir` folder. We will have to concatenate them on the next step.
4. Run `python mergevec.py -v vec_storage_tmp_dir -o final.vec`. You will have one `.vec` file
   with all the images. You can get this file from [here](https://github.com/wulfebw/mergevec).
5. Run the `vec2images final.vec output/%07d.png -w size -h size`. All the images will be in
   the output folder. `vec2image` has to be compiled. You can get the source from [here](http://note.sonots.com/SciSoftware/haartraining/vec2img.cpp.html).

You can see the results of the script now:

![Aligned face one]({{ site.url }}/assets/img/transformed_1.png)
![Aligned face one]({{ site.url }}/assets/img/transformed_2.png)
![Aligned face one]({{ site.url }}/assets/img/transformed_3.png)
![Aligned face one]({{ site.url }}/assets/img/transformed_4.png)
![Aligned face one]({{ site.url }}/assets/img/transformed_5.png)