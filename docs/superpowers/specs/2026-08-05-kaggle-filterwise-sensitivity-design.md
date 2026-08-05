# Kaggle Filterwise Sensitivity Notebook Design

## Goal

Provide one self-contained Kaggle Jupyter notebook that measures the filterwise pruning sensitivity of every structurally prunable YOLOv8n convolution layer. Its results will be used to select layers and cluster sizes for the later Jetson Orin Nano structured-pruning experiments.

## Scope

The notebook will contain all experiment code, configuration, dependency installation, layer discovery, pruning, evaluation, result persistence, plotting, and interpretation helpers. It will not depend on this repository at runtime.

The Kaggle input dataset will contain only:

- the validation JPG/PNG images;
- YOLO-format validation labels;
- a small dataset YAML with class names and paths;
- the baseline YOLO checkpoint (`best.pt`).

Duplicate NPY image files are explicitly excluded.

## Notebook workflow

1. Install pinned runtime dependencies: PyTorch, Ultralytics, Torch-Pruning, NumPy, Pandas, Matplotlib, and PyYAML.
2. Configure input/output paths and validate the Kaggle GPU, the checkpoint, the validation set, and the YAML class mapping.
3. Load the dense checkpoint and discover candidate `torch.nn.Conv2d` modules. Exclude the YOLO `Detect` head, convolutions with protected input-channel constraints, and unsupported graph-dependent targets. Print the final candidate list before any experiment runs.
4. Evaluate the dense baseline once and save its mAP50, mAP50-95, precision, recall, parameter count, and run metadata.
5. For each candidate layer, repeatedly:
   - reload the dense checkpoint to keep layer experiments independent;
   - select the current minimum-L1 output filter for that layer;
   - structurally remove one output filter and propagate the dependency change;
   - evaluate mAP50-95, mAP50, precision, and recall on the validation set;
   - append one durable result row.
6. Continue until one output filter remains, unless the notebook configuration explicitly enables an optional early-stop safeguard. Full-curve mode is the default.
7. After every candidate, rewrite `results.csv`, `manifest.json`, and `progress.json` in the Kaggle working directory. A later notebook session may attach a prior output archive and resume rows that already exist.
8. Generate one mAP50-95 curve and one recall curve per layer, plus a ranked layer summary that identifies the greatest filter-removal count meeting configurable mAP50-95 and recall-drop limits.

## Experimental rules

- Pruning criterion: minimum L1 norm of the current layer's output filters.
- Unit of removal: one output filter per evaluation step.
- Candidate experiments are independent: pruning a layer never carries into the next layer's experiment.
- Validation is run at a fixed image size, confidence threshold, IoU threshold, split, and seed defined near the top of the notebook.
- The notebook reports RTX/Kaggle GPU metrics only as sensitivity evidence. It does not claim Jetson latency or energy performance.
- The notebook stores results and milestone checkpoints; it does not retain every intermediate checkpoint.

## Outputs

The notebook writes the following downloadable artifacts:

- `results.csv`: one row for the baseline and every evaluated layer/filter-removal point;
- `manifest.json`: structured experiment metadata and all rows;
- `progress.json`: resumable completion state;
- `plots/<layer>_sensitivity.png`: mAP50-95 and recall response curves;
- `layer_summary.csv`: layer rankings and recommended removal ranges;
- `milestones/`: baseline plus selected checkpoints at configured retention points.

## Failure handling

- Each candidate failure is captured as a row with its layer, removal count, and error text; the notebook continues to the next candidate where possible.
- GPU availability, missing input files, invalid labels, invalid output shapes, and unsupported dependency-graph operations fail early with a clear message.
- Resume only accepts previous rows whose baseline checkpoint path, dataset YAML identity, image size, and pruning criterion match the current notebook configuration.

## Success criteria

- A Kaggle user can upload the small validation/checkpoint package, run the notebook without cloning this repository, and obtain a complete or resumable filterwise sensitivity result set.
- Each supported convolution layer has an independently measured 1-filter-at-a-time curve through one remaining output filter.
- Results clearly distinguish safe, borderline, and unsuitable pruning levels using configurable accuracy and recall limits.
- The generated CSV, JSON, summary table, and plots are sufficient to select candidates for the subsequent Jetson cluster-pruning matrix.
