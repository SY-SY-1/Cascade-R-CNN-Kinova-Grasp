'''
this file is to detect image from camera, you should receive image from camera and save them
in your directory in image root, we set frequency equal to 1fps
input:
image_root, your image receive dir
output:
detection result
author lxl 2021-06-06
'''
import glob
import os
from mmdet.apis import init_detector, inference_detector
import numpy as np
from shapely.geometry import Polygon
from matplotlib import pyplot as plt
import matplotlib.image as mimage
import time
import shutil
# config parameter
image_root = '/data/lixulong/bishe/rcv_images/0605/' # save image from camera
# image_root = '/home/lixulong/Desktop/untitled folder/0512_evening/'
data_root = image_root
checkpoint_file = '/data/lixulong/mm_grasp_dete/log/model/20210517/epoch_60.pth'
config_file = '/home/lixulong/catkin_ws_grasp/src/mmdetection/configs/first_5_5_5_train.py'
save_detection = False
score_threshold = 0.6
# return newest image name to be detected
def test_exist_image():
    new_add_file = glob.glob(image_root + "*.png")
    print("current receiving image name:")
    print(new_add_file)
    print("return first image:")
    try:
        return new_add_file[-1]
    except:
        return ''


# begin detecting
# input: an image file name
# return : box(center, x, y, degree)
def detecting_one_image(img_file):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    try:
        img_data = mimage.imread(img_file)
    except:
        return
    result = inference_detector(model, img_file)
    # visualize the result
    effective_result = []
    # return parameter
    grasp_rectangle = []
    cur_index = 1
    for each_result in result:
        if len(each_result) != 0:
            # pdb.set_trace()
            # angle_rad = -np.pi / 20 * (cur_index)
            angle_rad = -np.pi / 2 - np.pi / 20 * (cur_index)
            for j in range(len(each_result)):
                if each_result[j][4] > score_threshold:
                    x1 = each_result[j][0]
                    x2 = each_result[j][2]
                    y1 = each_result[j][1]
                    y2 = each_result[j][3]
                    x_cnt = (x1 + x2) / 2
                    y_cnt = (y1 + y2) / 2
                    pts = np.array([[x1, y1], [x2, y1],
                                    [x2, y2], [x1, y2]])
                    cnt = np.array([x_cnt, y_cnt])

                    r_bbox = np.dot(pts - cnt, np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                                                         [-np.sin(angle_rad), np.cos(angle_rad)]])) + cnt
                    predict_polygon = Polygon([(r_bbox[0, 0], r_bbox[0, 1]), (r_bbox[1, 0], r_bbox[1, 1]),
                                               (r_bbox[2, 0], r_bbox[2, 1]), (r_bbox[3, 0], r_bbox[3, 1])])

                    effective_result.append(predict_polygon)
                    grasp_rectangle.append([x_cnt, y_cnt, angle_rad / 3.14 * 180])
        else:
            pass
        cur_index += 1
    plt.plot()
    plt.imshow(img_data)
    for each_angle_prediction in effective_result:
        pred_x, pred_y = each_angle_prediction.exterior.xy
        plt.plot(pred_x[0:2], pred_y[0:2], color='k', alpha=0.7, linewidth=1, solid_capstyle='round',
                 zorder=2)
        plt.plot(pred_x[1:3], pred_y[1:3], color='r', alpha=0.7, linewidth=3, solid_capstyle='round',
                 zorder=2)
        plt.plot(pred_x[2:4], pred_y[2:4], color='k', alpha=0.7, linewidth=1, solid_capstyle='round',
                 zorder=2)
        plt.plot(pred_x[3:5], pred_y[3:5], color='r', alpha=0.7, linewidth=3, solid_capstyle='round',
                 zorder=2)

    # return [x,y,degree]
    if save_detection:
        if not os.path.exists(data_root + '/grasp_detection_result'):
            os.makedirs(data_root + '/grasp_detection_result')
        plt.savefig(data_root + '/grasp_detection_result/' + str(count_pic) + '.png')
        count_pic = count_pic + 1
        plt.close()
    plt.show()
    time.sleep(1)
    plt.close()
    return grasp_rectangle

def vis_result_in_angle():
    count_pic = 0
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    img_file = glob.glob(data_root + '/*.png')
    # pdb.set_trace()
    random.shuffle(img_file)
    for each_image in img_file:
        try:
            img_data = mimage.imread(each_image)
        except:
            continue
        # img_data = cv2.imread(each_image)
        result = inference_detector(model, each_image)
        # visualize the result
        effective_result = []
        cur_index = 1
        for each_result in result:
            if len(each_result) != 0:
                # pdb.set_trace()
                angle_rad = -np.pi / 2 - np.pi / 20 * (cur_index)
                for j in range(len(each_result)):
                    if each_result[j][4] > score_threshold:
                        x1 = each_result[j][0]
                        x2 = each_result[j][2]
                        y1 = each_result[j][1]
                        y2 = each_result[j][3]
                        x_cnt = (x1 + x2) / 2
                        y_cnt = (y1 + y2) / 2
                        pts = np.array([[x1, y1], [x2, y1],
                                        [x2, y2], [x1, y2]])
                        cnt = np.array([x_cnt, y_cnt])

                        r_bbox = np.dot(pts - cnt, np.array([[np.cos(angle_rad), np.sin(angle_rad)],
                                                             [-np.sin(angle_rad), np.cos(angle_rad)]])) + cnt
                        predict_polygon = Polygon([(r_bbox[0, 0], r_bbox[0, 1]), (r_bbox[1, 0], r_bbox[1, 1]),
                                                   (r_bbox[2, 0], r_bbox[2, 1]), (r_bbox[3, 0], r_bbox[3, 1])])

                        effective_result.append(predict_polygon)
            else:
                pass
            cur_index += 1
        plt.plot()
        plt.imshow(img_data)
        for each_angle_prediction in effective_result:
            pred_x, pred_y = each_angle_prediction.exterior.xy
            plt.plot(pred_x[0:2], pred_y[0:2], color='k', alpha=0.7, linewidth=1, solid_capstyle='round',
                     zorder=2)
            plt.plot(pred_x[1:3], pred_y[1:3], color='r', alpha=0.7, linewidth=3, solid_capstyle='round',
                     zorder=2)
            plt.plot(pred_x[2:4], pred_y[2:4], color='k', alpha=0.7, linewidth=1, solid_capstyle='round',
                     zorder=2)
            plt.plot(pred_x[3:5], pred_y[3:5], color='r', alpha=0.7, linewidth=3, solid_capstyle='round',
                     zorder=2)
        if save_detection:
            if not os.path.exists(data_root + '/grasp_detection_result'):
                os.makedirs(data_root + '/grasp_detection_result')
            plt.savefig(data_root + '/grasp_detection_result/' + str(count_pic) + '.png')
            count_pic = count_pic + 1
            plt.close()
        plt.show()


if __name__ == '__main__':
    if not os.path.exists(image_root + 'detected'):
        os.makedirs(image_root + 'detected')
    while True:
        img_file = test_exist_image()
        if img_file != '':
            grasp_rectangle = detecting_one_image(img_file)
            print("current detection result:")
            try:
                for each_result in grasp_rectangle:
                    print("[x,y,degree]")
                    print(each_result[0],each_result[1],each_result[2]+270)
            except:
                pass
            src_dir = img_file
            dst_dir = os.path.join(image_root, "detected", os.path.basename(img_file))
            shutil.move(src_dir, dst_dir)
            print("file has been move from " + src_dir + " to " + dst_dir)
        time.sleep(1)  # pause 1 second