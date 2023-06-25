import onnxruntime
import argparse
import os
import numpy as np
import json
from PIL import Image

ONNX_PATH = 'model_qt.onnx'
HERO_NAME = 'hero_name_mapping.json'
GTH_PATH = 'test_data/test.txt'

def load_json(path):
    with open(path, 'r') as f:
        res = json.load(f)
    return res

def preprocesing(path):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    img = img.crop([0, 0, int(w*0.2), h])
    w, h = img.size
    if w < h:
        new_img = Image.new(img.mode, (h, h), color=img.getpixel((0, 0)))
        new_img.paste(img, ((h-w)//2, 0))
    elif w > h:
        new_img = Image.new(img.mode, (w, w), color=img.getpixel((0, 0 )))
        new_img.paste(img, ((w-h)//2, 0))
    else:
        new_img = img
    img = new_img.resize((96, 96), Image.ANTIALIAS)
    img = np.array(img, dtype=np.float32)
    img = img/255
    img = img[np.newaxis, :]
    return img

def eval(dir, output):
    ort = onnxruntime.InferenceSession(ONNX_PATH)
    hero_mapping = load_json(HERO_NAME)
    index_to_name = dict()
    for hn in hero_mapping:
        index_to_name[hero_mapping[hn]] = hn
    res = {}
    for fi in os.listdir(dir):
        input = preprocesing(os.path.join(dir, fi))
        pred = ort.run(None, {'input': input})[0]
        pred = np.argmax(pred[0])
        res[fi] = index_to_name[pred]
    
    with open(output, 'w') as f:
        for fi in res:
            f.write(fi + '\t' + res[fi] + '\n')

def read_file(path):
    res = dict()
    with open(path, 'r') as f:
        for line in f.readlines():
            fi, lb = line.strip().split()
            res[fi] = lb
    return res

def compare_results(gth_path, pred_path):
    gth = read_file(gth_path)
    pred = read_file(pred_path)
    count = 0
    for fi in gth:
        if gth[fi] == pred[fi]:
            count += 1
    print(count, len(gth), count / len(gth))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_dir', '-i', type=str, default='test_data/test_images')
    parser.add_argument('--output', '-o', type=str, default='output.txt')
    parser.add_argument('--mode', '-m', type=int, default=0)
    parser.add_argument('--gth_path', '-g', type=str, default=GTH_PATH)
    args = parser.parse_args()

    if args.mode == 0:
        eval(dir=args.test_dir, output=args.output)
    elif args.mode == 1:
        compare_results(gth_path=args.gth_path, pred_path=args.output)
    else:
        raise NotImplementedError

