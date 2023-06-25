Hero Name Recognition

# Requirements
```
onnxruntime
numpy
json
argparse
PIL
```

# Directories Structures
```
- source.py: main source.
- hero_name_mapping.json: all hero names with index.
- model_qt.onnx: model weight.
```

# Usage:
- For predicting all images in `test_dir` and export results into `output file`, run:
```
python source.py -i <test_dir> -o <path of output file> 
```
- For check accuracy between `groundtruth txt file` and `predict txt file`, run:
```
python source.py -m 1 -o <predict txt file> -g <groundtruth txt file>
```