---
layout: post
title: "Google Summer Of Code: Optimizmizing existing code. Creating object detection module."
description:
comments: true
categories:
- gsoc
---


The post describes the steps that were made in order to speed-up Face Detection.

___

## Rewriting of `MB-LBP` function into `Cython`

As the `MB-LBP` is called many times during the Face Detection.
For example, in a region of an image that contains face of size `(42, 35)` 
the function was called `3491` times. The sliding window approach was used.
These numbers will be much greater if we use bigger image. This is why the
function was rewritten in `Cython`. In order to make it fast all the `Python`
calls were eliminated and the function now uses `nogil` mode.

## Implementing the `Cascade` function and rewriting it in `Cython`
 
In the approach that we use for Face Detection the cascade of classifiers is
used in order to detect the face. Only the face passes all stages and is detected.
All the non-faces are rejected on some stage of cascade. The cascade function is also called
a lot of times. This is why the class that has all the data was written in `Cython`.
As opposed to native `Python` classes, `cdef` classes are implemented using `struct` `C` structure.
`Python` classes use `dict` for properties and method search which is slow.

Other additional entities that are needed for cascade to work were implemented using pure `struct`
`C` structure.

## New module

For the current project I decided to put all my work in `skimage.future.objdetect`. I did this
because the functions can be changed a lot in the future.
The name `objdetect` was used because the approach that I use will make it possible to detect
not only faces but other objects on which the classifier can be trained.