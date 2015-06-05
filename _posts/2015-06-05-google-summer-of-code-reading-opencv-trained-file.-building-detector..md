---
layout: post
title: "Google Summer of Code: reading OpenCv trained file. Building detector."
description:
comments: true
categories:
- gsoc
---



A post about how I read the `OpenCv` trained file and implemented a prototype of detector.

___


## Example of a face Evaluation by detector

This is an example of what stages and respective weak classifiers are stored
in the `OpenCv` trained file for `Mb-LBP`. There are 20 stages overall
with increasing accuracy. The cascade structure speeds up the process 
by orders of magnitude because the non-faces patches are rejected on the
first stages of the classifier.


![Cascade of classifiers]({{ site.url }}/assets/img/face_evaluation.gif)


## Sliding window and scale example.

I also implemented an naive sliding window and scale search.

Because the classifier trained only for `24x24` window size we, have to scale
the image and also slide the window to find a face. To fire, the face should fit
into the `24x24` window.

This is an example. I started with `(118, 118)` image with a face.
On this scale classifier didn't detect any faces which is correct.

![Cascade of classifiers]({{ site.url }}/assets/img/0.png)

Then, the program scaled the image by the ratio of `0.5` (size `(59, 59)`) and detected one
image. Actually it's a part of the face and we don't want detection like this,
but it can be overcome by using `pruning` technique.

![Cascade of classifiers]({{ site.url }}/assets/img/1.png)

Then the the images was downscaled once again and this is where the face was found.
Once again, this big amount of detections of the same face can be removed by `pruning`.

![Cascade of classifiers]({{ site.url }}/assets/img/2.png)