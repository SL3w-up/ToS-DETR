# Data Preparation

ToS-DETR expects 2.5D CT images and aligned 2.5D cartilage masks. This document describes the preprocessing contract used by the model.

## CT Windowing

Convert each CT slice to 8-bit intensity using:

- window width: `400`
- window level: `100`

Clip the HU range to `[level - width / 2, level + width / 2]`, then linearly map to `[0, 255]`.

## 2D Slice and Mask Preparation

For each case:

1. Export axial 2D CT slices after windowing.
2. Export the corresponding cartilage or costal-cartilage ROI mask for each slice.
3. Keep CT image and mask file names aligned.
4. Use PNG or another lossless image format for reproducibility.

If TotalSegmentator or another segmentation tool is used to generate cartilage masks, keep that tool outside this repository and document the version and label mapping in your experiment notes.

## 2.5D Construction

Build one three-channel image from adjacent slices. The intended channel semantics are:

| Channel | Content |
| --- | --- |
| C0 | previous slice |
| C1 | anchor / center slice |
| C2 | next slice |

Use the same three-slice grouping for the mask. The center channel is the anchor slice used for detection supervision and evaluation.

For stride-2 sliding, a case may be sampled as `[t, t+1, t+2]`, then `[t+2, t+3, t+4]`; in that convention the middle channel remains the supervised anchor.

## Mask-Guided Cropping

Each 2.5D image and its same-name 2.5D mask are cropped together using the mask ROI:

1. Find all non-zero pixels in the mask.
2. Crop a region that keeps the entire non-zero ROI.
3. Adjust the crop to target size height `448`, width `224`.
4. If the crop is smaller than the target size, expand around the ROI where possible.
5. If the crop is larger than the target size, resize or crop according to the experiment's preprocessing policy.
6. Apply the exact same crop/resize transform to bounding-box labels.

For empty masks, use a deterministic policy such as copying the nearest non-empty mask crop box within the same case. Avoid leaving empty-mask images at their original uncropped size if the model was trained on fixed-size inputs.

## Annotation Format

The training code consumes COCO-style JSON annotations. Each image entry should include:

```json
{
  "id": 1,
  "file_name": "CASE001_120_240.png",
  "width": 224,
  "height": 448,
  "case_id": "CASE001",
  "slice_index": 120,
  "num_slices": 240
}
```

Each fracture annotation should use a box on the anchor slice. The current model is single-class, so all positive boxes use category id `0` after conversion.

If labels start as YOLO-style text files, each row should be:

```text
0 x_center y_center width height
```

where coordinates are normalized to the current image size. When converting after mask-guided cropping, transform boxes from the original image coordinate system into the cropped/resized coordinate system.

## Expected Directory Layout

```text
data/coco/
├── train2017/
├── val2017/
├── anatomy_masks/
│   ├── train2017/
│   └── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

The mask file name should match the image file name unless an explicit mask path is stored in the COCO image metadata.
