# -*- coding:utf-8 -*-
import os, sys
import mxnet as mx

def _test_ocr():
    from cnocr import CnOcr
    #conv-lite-fc densenet-lite-gru densenet-lite-fc densenet-lite-s-gru densenet-lite-s-fc
    ocr = CnOcr(model_name='densenet-lite-gru', model_epoch=20) # 22 25
    img_fp = 'examples/00199978.jpg'    # 00199978 00199980
    # img_fp = 'examples/multi-line_cn1.png'
    img = mx.image.imread(img_fp, 1)
    res = ocr.ocr(img)
    # res = ocr.ocr_for_single_line(img)
    print("Predicted Chars:", res)


def _test_path():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(BASE_DIR)
    from cnocr.utils import data_dir
    print(data_dir())


if __name__ == "__main__":
    _test_path()


