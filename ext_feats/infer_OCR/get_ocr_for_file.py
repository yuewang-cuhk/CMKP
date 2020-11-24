from PIL import Image
import os
import time
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'E:/Program Files/Tesseract-OCR/tesseract.exe'


def get_ocr_from_file(img_dir, trg_fn):
    t0 = time.time()
    if os.path.exists(trg_fn):
        fr = open(trg_fn, 'r', encoding='utf-8')
        used_set = set([line.split(':')[0].strip() for line in fr])
        fr.close()
    else:
        used_set = set()
    print('There are already %d images in %s' % (len(used_set), trg_fn))
    with open(trg_fn, 'a', encoding='utf-8') as fa:
        for idx, fn in enumerate(os.listdir(img_dir)):
            if fn in used_set:
                continue
            img_fn = os.path.join(img_dir, fn)
            try:
                text = pytesseract.image_to_string(Image.open(img_fn))
                fa.write(fn + ': ' + text.replace('\n', ' ') + '\n')
            except:
                print('Error image: %s' % img_fn)
                continue
            if idx % 10 == 0:
                print('Processing %d lines, takes %.2f seconds' % (idx, time.time() - t0))
    print('Finish writing %d lines into %s, takes %.2f seconds' % (idx, trg_fn, time.time() - t0))


if __name__ == '__main__':
    img_dir = 'D:/CMKG_images'
    # img_dir = '/research/lyu1/yuewang/workspace/MMKG_images'
    ocr_files = 'CMKG_ocr.txt'
    get_ocr_from_file(img_dir, ocr_files)
