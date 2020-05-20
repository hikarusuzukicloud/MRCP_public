# coding: utf-8

import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import tensorflow as tf


class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, frozen_path):
        self.graph = tf.Graph()
        graph_def = None
        with open(frozen_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

    def returnSize(self,image):
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        return target_size


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label, colormap):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]


def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.show()

def label2seg(seg_map):
    seg = np.array(seg_map)
    l1 = seg.shape[0]
    l2 = seg.shape[1]
    count=0
    for i in range(l1):
        for j in range(l2):
            if seg_map[i][j] > 0:
                seg_map[i][j] = 250
            else:
                count+=1

    print(l1*l2)
    print(count)
    plt.clf()
    plt.figure()
    plt.imshow(seg_map)
    plt.show()


# label setting
LABEL_NAMES = np.asarray([
    'background', 'pancreatic_duct', 'common_bile_duct', 'ntrahepatic_bile_duct'
]+ ["a"]*21)

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
colormap = create_pascal_label_colormap()
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP, colormap)


def main():
    #imgdir = "/media/hikaru/ubuntuhdd/mrcp_dataset/ALL_original_jpg"
    #img_path = "AIP2_11.jpg"
    #imgdir = "/media/hikaru/ubuntuhdd/mrcp_dataset/AIP_original_only/original_AIP79"
    #img_path = "10.png"
    model_path = "/media/hikaru/ubuntuhdd/mrcp_dataset/log/export/frozen_inference_graph.pb"
    output = "/media/hikaru/ubuntuhdd/mrcp_dataset/test_output"

    # load model
    model = DeepLabModel(model_path)

    input_dir = "/media/hikaru/ubuntuhdd/mrcp_dataset/AIP_original_only"
    patlist = [69,70,73,74,76,79,81,83,87,91,92,93,98,100,101,104,105,108,109,110,115]
    dirs = []
    for x in patlist:
        n = "original_AIP" + str(x)
        dirs.append(n)
    for d in dirs:
        for p in ["2.png", "6.png", "10.png"]:
            original_im = Image.open(os.path.join(os.path.join(input_dir, d), p)).convert("RGB")
            resized_im, seg_map = model.run(original_im)
            seg_image = label_to_color_image(seg_map, colormap).astype(np.uint8)
            seg_image_pil = Image.fromarray(seg_image)
            seg_image_pil.save(os.path.join(output, d + "_" + p))

    """
    # read image
    original_im = Image.open(os.path.join(imgdir, img_path)).convert("RGB")
    #original_im = Image.open(os.path.join(imgdir, img_path))
    #original_im_resize = original_im.resize((520, 520), Image.LANCZOS)
    # inferences DeepLab model
    resized_im, seg_map = model.run(original_im)

    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    seg_image_pil = Image.fromarray(seg_image)
    seg_image_pil.save(os.path.join(output, img_path))
    # show inference result
    #vis_segmentation(resized_im, seg_map)

    #label2seg(seg_map)
    """

if __name__ == '__main__':
    main()
