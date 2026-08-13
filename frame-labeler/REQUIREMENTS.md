# Frame Labeler v1 Requirements

Status: Draft

Date: 2026-08-12

## Goal

Provide a local desktop GUI, launched from the command line with one image or video, for creating precise axis-aligned object-detection boxes and exporting reviewed work in Ultralytics YOLO detection format.

## Working Assumptions

- The working tool name and command are `frame-labeler`.
- Version 1 covers object-detection boxes with one class per box. It does not cover image-level classification, segmentation, keypoints, or rotated boxes.
- A project contains one source image or video and one stable class catalog.
- Video stride is a positive integer measured in source frames. The default is `1`.
- Work is local and single-user.
- YOLO detection is the required export. Persistence remains independent of any export format.

## Command-Line Contract

The intended interface is:

```text
frame-labeler open SOURCE --project PROJECT --classes CLASSES [--stride N] [--split train|val|test]
frame-labeler export PROJECT --format yolo --output DIRECTORY
```

- `SOURCE` must be a readable local image or video.
- `PROJECT` identifies the persistent annotation project. Reopening it resumes work.
- `CLASSES` is a UTF-8 text file with one unique, non-empty class name per line. It is required only when creating a project.
- `--stride` applies only to video and must be greater than zero.
- `--split` records the intended dataset split and defaults to `train`.
- Invalid arguments, unreadable media, unsupported media, and source/project mismatches must fail with actionable messages before the GUI opens.
- Minimum verified media support is JPEG, PNG, and an MP4 video containing H.264 video. Additional decoder-supported media may work but is not part of v1 acceptance.

## Functional Requirements

### FR-1: Frame sampling and navigation

- An image is presented as one review item.
- A video presents source frames `0, stride, 2 * stride, ...` while the index is less than the source frame count.
- The GUI shows the source frame index, sampled-frame position, total sampled-frame count when known, and review state.
- Users can move to the previous or next sampled frame and jump to a source-frame index that resolves to the sampling sequence.
- Frames are decoded on demand with bounded prefetching. Opening a project must not extract the entire video.
- Display orientation, dimensions, and box coordinates must match the decoded source frame.

### FR-2: Box editing

- Dragging on the image creates an axis-aligned box using the active class.
- Users can select, move, resize, delete, and change the class of a box.
- Boxes are clipped to image bounds and cannot have zero width or height.
- Selected boxes show corner and edge resize handles, directional resize cursors, and a readable class label.
- A stable, synchronized box list permits direct selection when boxes overlap and supports cycling through the current frame's boxes.
- Keyboard controls nudge selected boxes by one source pixel, or ten source pixels with a modifier, while preserving image bounds.
- Undo and redo cover box creation, movement, resize, deletion, and class changes within the current session.

### FR-3: Precision viewing

- The image initially fits within the available canvas without changing its aspect ratio.
- Mouse-wheel or trackpad input zooms around the pointer position.
- Space-drag, right-button drag, or middle-button drag pans the image without changing annotations.
- Controls provide fit-to-window; mouse-wheel or trackpad input controls all incremental zooming.
- Mouse and keyboard mappings remain visible in the labeling window.
- Drawing and editing at any zoom level map to the same source-pixel coordinates.

### FR-4: Carry-forward boxes

- The first sampled frame in a new project starts with no boxes because it has no preceding frame.
- When moving forward to an unvisited sampled frame, boxes from the immediately preceding reviewed sampled frame are copied at the same source-pixel coordinates and with the same classes.
- Copied boxes are visibly marked as draft.
- Editing a copied box never changes the preceding frame.
- Draft boxes are not exportable until the user marks the frame reviewed.
- Existing annotations on a revisited frame are never overwritten by carry-forward behavior.
- A reviewed frame may intentionally contain zero boxes.

### FR-5: Review and save behavior

- Each sampled frame has one of three states: `unreviewed`, `draft`, or `reviewed`.
- Marking a frame reviewed confirms its current boxes, including an intentionally empty set.
- Annotation mutations and review-state changes are saved transactionally without a separate save command.
- The GUI displays whether the current state has been persisted.
- Reopening a project restores its source, class catalog, stride, split, last position, review states, and boxes.

### FR-6: Class catalog

- Class IDs are stable, zero-based integers assigned from the class-file order.
- Class names must be unique after trimming whitespace.
- A class name may be renamed without changing its ID.
- A class used by an annotation cannot be deleted silently.
- The active class can be changed without leaving the canvas.

## Persistence Requirements

- The canonical project state is a versioned SQLite database stored at the requested project path.
- Export files are derived artifacts and are never read as canonical project state.
- Box coordinates are stored as `x_min`, `y_min`, `x_max`, and `y_max` in original source-pixel space using sufficient precision for subpixel transforms.
- The schema records source identity and metadata, class IDs and names, sampling configuration, dataset split, frame index and timestamp, frame dimensions, review state, stable box IDs, box origin, and creation/update timestamps.
- Box origin distinguishes boxes drawn on the current frame from boxes copied from the preceding frame.
- Schema changes require explicit versioning and tested migrations.
- A source mismatch must never silently attach existing annotations to different media.
- Writes must be atomic. An interrupted write must leave the last committed state readable.
- Memory usage must not grow in proportion to total video duration or annotation count during ordinary navigation.

## YOLO Export Requirements

- Only reviewed frames are exported. Draft and unreviewed frames are excluded.
- Every exported video frame becomes a deterministic image file keyed by source identity and source-frame index.
- Output uses `images/<split>/` and `labels/<split>/` directories plus a `dataset.yaml` class map.
- Each non-empty label file contains one object per line as `class_id x_center y_center width height`.
- Class IDs are zero-based. Coordinates are normalized to `[0, 1]` using the exported image dimensions.
- A reviewed frame with zero boxes is exported as an image without a required label file.
- A JSON Lines manifest maps every exported image to its project, source identity, source-frame index, timestamp, dimensions, split, and review timestamp.
- Re-exporting the same project is deterministic and removes stale files previously generated for that project.
- Export must not overwrite unrelated files or combine projects with incompatible class catalogs.
- Export runs incrementally with bounded memory and reports completed, skipped, and failed items.

The YOLO representation follows the current Ultralytics detection dataset specification: one label file per image, zero-based classes, and normalized center-based box coordinates.

## Success Criteria

### Functional acceptance

1. Opening a 12-frame test video with stride `3` presents source frames `0`, `3`, `6`, and `9` in order.
2. A box drawn after zooming and panning is stored within `0.5` source pixel of the intended coordinates.
3. Advancing from a reviewed frame creates independent draft boxes at identical source-pixel coordinates on the next sampled frame.
4. Draft boxes do not appear in an export. Marking that frame reviewed makes them exportable.
5. A reviewed empty frame remains distinguishable from an unreviewed frame and is included as an unlabeled image in the export.
6. Moving, eight-direction resizing, keyboard nudging, list selection, deleting, reclassifying, undoing, and redoing boxes produces the expected persisted state.
7. Force-closing and reopening after a completed mutation recovers the last committed project state without partial records.
8. Reopening with different media produces a source-mismatch error rather than displaying old annotations.
9. YOLO export round-trips every box to within `1` source pixel and loads through the pinned supported Ultralytics version without dataset-format errors.
10. Repeating an unchanged export produces equivalent content and no duplicate or stale project files.

### Scale acceptance

- A one-hour 1080p, 30 FPS video opened with stride `30` becomes interactive without pre-extracting all frames.
- Resident memory growth remains below `250 MB` while navigating 1,000 sequential sampled frames on the reference test environment, excluding initial GUI and decoder allocation.
- After warm-up, next-frame presentation has a p95 latency of at most `500 ms` for the reference local H.264 fixture.
- Exporting 10,000 reviewed frames uses bounded memory; throughput is recorded as a benchmark rather than enforced across different hardware.

### Human-in-the-loop acceptance

A tester must complete and record this workflow on a real 1080p video:

1. Open the video at a non-default stride.
2. Draw a small box at fit-to-window zoom.
3. Zoom in, pan without moving a box, and refine all four edges.
4. Select overlapping boxes from the box list and refine one with keyboard nudging.
5. Mark the frame reviewed and advance.
6. Edit and confirm the carried-forward box.
7. Mark one frame reviewed with no boxes.
8. Navigate backward and confirm prior annotations are unchanged.
9. Restart the tool and resume at the saved position.
10. Export and visually inspect representative images and labels.

## Test Strategy

- Unit tests cover stride calculation, coordinate transforms, clipping, state transitions, carry-forward copying, class validation, and export conversion.
- Property-based tests cover view/source coordinate round-trips and pixel/normalized YOLO round-trips across image sizes and valid boxes.
- Persistence tests cover transactions, restart recovery, migrations, source matching, and independent copied boxes.
- Integration tests use generated image and video fixtures to exercise decoding, navigation, review, and deterministic export.
- GUI tests cover core input events and state changes without relying solely on screenshots.
- A manual acceptance record covers visual precision, input ergonomics, and the complete workflow above.
- A repeatable benchmark records navigation latency, memory growth, and export throughput on the reference environment.

## Non-Goals

- Polygon, mask, keypoint, or rotated-box annotation
- Multi-user or network collaboration
- Dataset augmentation, training, evaluation, or compilation
- Video editing, transcoding, or playback as an editor
- General media asset management

## Source Reference

- [Ultralytics object-detection dataset format](https://docs.ultralytics.com/datasets/detect/), accessed 2026-08-12.
