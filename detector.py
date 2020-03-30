import argparse
import sys
import time

import cv2
import numpy as np
import torch

from retinanet import model

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    sys.argv.append('--coco_path')
    sys.argv.append('/home/jht/github/deep-high-resolution-net.pytorch/data/coco')
    sys.argv.append('--model_path')
    sys.argv.append('coco_resnet_50_map_0_335_state_dict.pt')

    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)
    parser = parser.parse_args(args)

    retinanet = model.resnet50(num_classes=80, pretrained=True)
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model_path))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    # 初始化,并读取第一帧,rval表示是否成功获取帧,frame是捕获到的图像
    vc = cv2.VideoCapture('/home/jht/16_2.MP4')
    rval, frame = vc.read()

    # 获取视频fps
    fps = vc.get(cv2.CAP_PROP_FPS)
    # 获取视频总帧数
    frame_all = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print("[INFO] 视频FPS: {}".format(fps))
    print("[INFO] 视频总帧数: {}".format(frame_all))
    print("[INFO] 视频时长: {}s".format(frame_all / fps))

    mean = np.array([[[0.485, 0.456, 0.406]]])
    std = np.array([[[0.229, 0.224, 0.225]]])
    while rval:

        with torch.no_grad():
            st = time.time()
            rval, img_bgr = vc.read()
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = img_rgb.astype(np.float32) / 255.0
            frame = (img_rgb - mean) / std
            rows, cols, cns = frame.shape
            pad_w = 32 - rows % 32
            pad_h = 32 - cols % 32
            rows = rows + pad_w
            cols = cols + pad_h
            new_image = cv2.resize(frame, (cols, rows))
            img = torch.from_numpy(new_image)

            if torch.cuda.is_available():
                scores, labels, boxes = retinanet(img.permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = retinanet(img.permute(2, 0, 1).float().unsqueeze(dim=0))
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.7)

            tensor = img * std + mean
            img = tensor.mul(255).clamp(0, 255).byte().cpu().numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for j in range(idxs[0].shape[0]):
                bbox = boxes[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                print(int(labels[idxs[0][j]]))
                if int(labels[idxs[0][j]]) == 0:
                    draw_caption(img, (x1, y1, x2, y2), "person")
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            cv2.imshow('img', img)
            cv2.waitKey(0)


if __name__ == '__main__':
    main()
