# V-System Vascular Network Dataset Generator

Synthetic **vascular volumes + centre-line skeletons + graph metadata** generator inspired by
[psweens/V-System](https://github.com/psweens/V-System), built for training and benchmarking
**skeletonization / centre-line** prediction models.

This implementation produces per-sample folders with:
- **Volumes** in **Zarr**, **NIfTI**, or **TIFF slices** for both *vessels* and *centre-line*
- **Graph CSVs**: nodes, edges, per-edge samples, branches, adjacency
- Metadata-rich folder names that encode shape/format/pixel size/seed/etc.

---

## Features
- Curved branches (Catmull-Rom / Bézier), diameter taper, Murray’s law split
- Self-avoidance to prevent cross-overs (spatial hash + configurable clearance)
- Inward directed at start and stop based on minimal diameter
- Chunked painting for very large volumes (Zarr), optional multi-process generation

---

## Example Binary Volumes and Paired Centre-line Skeletons

![Binary Volume 1.](figure/binary1.png)
![Binary Centreline.](figure/skel1.png)

*Figure: Example output of binary volume and paired centre-line skeleton in dense/thin vascular parametrisation.*

![Binary Volume 1.](figure/binary2.png)
![Binary Centreline.](figure/skel2.png)

*Figure: Example output of binary volume and paired centre-line skeleton in sparse/thick vascular parametrisation.*

