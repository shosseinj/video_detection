# Video Object Detection and Segmentation

This repository contains experiments on combining object detection with image and video segmentation for temporally consistent foreground extraction.

## Pipeline

The main prototype combines:

- **YOLOv8** for object detection;
- **Segment Anything (SAM)** for mask initialization from detected bounding boxes;
- a **SAM2-style video predictor** for propagating object masks across subsequent frames;
- **OpenCV** for frame extraction, mask processing, visualization, and video writing.

A typical sequence is:

```text
video frame
  -> YOLO object detection
  -> bounding-box prompt
  -> SAM mask initialization
  -> SAM2 temporal propagation
  -> mask cleanup
  -> foreground/background video output
```

## Research Motivation

The project explores how a detector can provide automatic prompts to a segmentation model and how the resulting object state can be propagated through a video without rerunning full detection/segmentation initialization on every frame.

## Implementation Notes

The code maintains a bounded temporal state and removes older frame data during long sequences to limit memory growth. The current prototype is configured primarily around person detection and foreground extraction, but the detection stage can be adapted to other classes supported by the detector.

## Requirements

The implementation uses PyTorch, Ultralytics YOLO, OpenCV, Segment Anything, and local SAM2 training/inference modules. Model checkpoints are expected to be available locally.

## Upstream Attribution

The `sam2_train/` tree contains code from Meta's [Segment Anything 2](https://github.com/facebookresearch/sam2) project, including files adapted from [Segment Anything](https://github.com/facebookresearch/segment-anything), and retains Meta copyright notices. This repository's integration experiments should not be interpreted as an original implementation of SAM or SAM 2.

## Status

Research prototype. The code emphasizes experimentation with detector-guided temporal segmentation rather than a packaged end-user application.
