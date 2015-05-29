---
layout: post
title: "Google Summer of Code: Multi-Block Local Binary patterns implementation. Pure Python"
description:
categories:
- gsoc
---


A post describing my way of implementing MB-LBP and achieved results.

___


## The selected API

The [API][opencv_api] of the function was taken from OpenCv.
It was done in order to allow users to use OpenCv for training their object detector
and connect it later to `Scikit-Image`.
The training part will also be implemented for Scikit-Image.
This step will just allow users to be more flexible and also by doing this Scikit-Image
Face detection Framework will be tested on much earlier stage.

This is the visual description of MB-LBP that OpenCv uses:

![Cascade of classifiers]({{ site.url }}/assets/img/opencv_mblbp_api.png)


## Implemented functions

For this part two function were implemented:

1. Function that computes the `MB-LBP` given the left-top corner coordinates, width and height
   of one of 9 equal rectangles (See the previous part with OpenCv API).
   
2. Function that takes the computed `MB-LBP` feature and visualizes it on the selected image.

## MB-LBP visualization

The hatched regions show the rectangles that has less summed intensity value than
the central one.

![Cascade of classifiers]({{ site.url }}/assets/img/mblbp_vis_lena.png)
![Cascade of classifiers]({{ site.url }}/assets/img/mblbp_vis_coins.png)
![Cascade of classifiers]({{ site.url }}/assets/img/mblbp_vis_test.png)

## Future plans

The next stage will be to implement the same thing using `Cython`.

## Code

The code of functions can be found [here][branch_link].
The code that was used for generating images in the post is [here][blog_images_gist].

## Test coverage

Implemented functions were covered with [tests][tests].

[opencv_api]: http://stackoverflow.com/questions/22565531/understanding-opencv-lbp-implementation
[branch_link]: https://github.com/warmspringwinds/scikit-image/blob/mb-lbp/skimage/feature/texture.py#L300
[blog_images_gist]: https://gist.github.com/warmspringwinds/0394bbc39380ebd3c90d
[tests]: https://github.com/warmspringwinds/scikit-image/blob/mb-lbp/skimage/feature/tests/test_texture.py#L234