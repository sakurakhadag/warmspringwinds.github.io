---
layout: post
title: "Google Summer of Code: Creating Training set"
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

![Not aligned face]({{ site.url }}/assets/not_aligned_face.jpg)

If we just crop the faces as they are, it will be really hard for classifier to learn from it.
The reason for this is that we don't know how all of the faces in the database are positioned.
Like in the example above the face is rotated.
In order to get a good dataset we first align faces and then add small random transformations
that we can control ourselves. This is really convinient because if the training goes bad,
we can just change the parameters of the random transformations and experiment.