# Data

The full image sets are **not included** (size and licensing). This folder contains the manual annotations and file lists needed to rebuild them, plus a few sample images.

The Stage 2 ultrasound dataset came from a confidential lab collection. It is not included and was deleted after the project.

## Datasets

| Dataset | Source | Training | Test |
|---|---|---|---|
| Dogs | [dog.ceo](https://dog.ceo/) (Stanford Dogs images, plus a few other photos) | 50 | 20 |
| Landscape | [Unsplash](https://unsplash.com/s/photos/landscape) | 50 | 20 |

## Structure

```
data/
├── dogs/
│   ├── test_names.txt              # filenames of the 20 test images
│   ├── samples/                    # 5 example images
│   └── training/
│       ├── labelsTexts.json        # manual annotations (50 entries)
│       └── original_names.txt      # original filenames of the training images
└── landscape/
    └── (same structure)
```

## Annotations

`labelsTexts.json` is a list of lists. Entry N lists the descriptors visible in the Nth training image, in numeric order:
`dog  (1).jpg`, `dog  (2).jpg`, ... `dog  (50).jpg` (note the two spaces in the filenames). Landscape images are named `Landscape  (1).jpg` and so on.

The link between image and label is the **numeric order**, so the files must not be reordered.

## Descriptors

- **Dogs (11):** brown, black, white, golden, straight fur, curly fur, long fur, short fur, protruded snout, flat snout, spotted
- **Landscape (13):** trees, clouds, sand, dark, bright, mountain, flower, snow, animals, plants, rocks, grass, water

## Rebuilding the dataset

1. Download the images listed in `original_names.txt` (training) and `test_names.txt` (test). For dogs, names like `n02091134_232.jpg` come from the Stanford Dogs dataset. For landscapes, filenames contain the Unsplash photographer and photo ID.
2. Rename the training images to `dog  (1).jpg` ... `dog  (50).jpg`, following the annotation order.
3. Place them in `data/<dataset>/training/` and the test images in `data/<dataset>/testing/`.

Note: `original_names.txt` for landscape lists 36 of the 50 training images. The other 14 original filenames were not kept.

## Licenses

The sample images belong to their original authors (Stanford Dogs / dog.ceo, Unsplash photographers) and are included for demonstration only. The annotations are original work for this project.
