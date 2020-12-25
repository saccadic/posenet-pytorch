import torch
import cv2
import time
import argparse
from tqdm import tqdm
import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--show', action='store_true', default=False, help='whether to show img')
parser.add_argument('--movie_path', type=str, default="")
parser.add_argument('--save_movie_path', type=str, default="")
parser.add_argument('--scale_factor', type=float, default=0.7125)


def main():
    args = parser.parse_args()

    if args.show:
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)

    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if (args.movie_path == ""):
        print("Please input a image or movie file.")
        exit()

    video = cv2.VideoCapture(args.movie_path)

    if video.isOpened() == False:
        print("Movie load faile.")
        exit()

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_rate = int(video.get(cv2.CAP_PROP_FPS)) 

    if len(args.save_movie_path) > 0:
        fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')  # ファイル形式(ここではmp4)
        writer = cv2.VideoWriter(args.save_movie_path, fmt, frame_rate, (width, height))  # ライター作成

    start = time.time()
    for i in tqdm(range(frame_count)):
        input_image, display_image, output_scale = posenet.read_cap(
            video,
            scale_factor=args.scale_factor,
            output_stride=output_stride
        )

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.15
            )

        keypoint_coords *= output_scale

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)

        if args.show:
            cv2.imshow('result', overlay_image)

        if len(args.save_movie_path) > 0:
            writer.write(overlay_image)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print('Average FPS: ', frame_count / (time.time() - start))


    cv2.destroyAllWindows()

    if len(args.save_movie_path) > 0:
        writer.release()

if __name__ == "__main__":
    main()