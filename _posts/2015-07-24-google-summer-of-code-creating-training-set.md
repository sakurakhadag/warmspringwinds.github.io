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
