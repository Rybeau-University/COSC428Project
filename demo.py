# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import multiprocessing as mp
import os
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger

from predictor import GolfSwingAnalyser

# constants
WINDOW_NAME = "Golf Swing Analysis"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Golf Swing Analysis Using Pose Estimation")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--ref-video", help="Path the swing reference video")
    parser.add_argument("--analysis-video", help="Path to video of swing to analyse")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--opposite-hands",
        help="The players in the two videos have difference dominant hands.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    analyser = GolfSwingAnalyser(cfg)

    if args.ref_video and args.analysis_video:
        analysis_video = cv2.VideoCapture(args.analysis_video)
        ref_video = cv2.VideoCapture(args.ref_video)
        width = int(ref_video.get(cv2.CAP_PROP_FRAME_WIDTH) + analysis_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(max(ref_video.get(cv2.CAP_PROP_FRAME_HEIGHT), analysis_video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frames_per_second = ref_video.get(cv2.CAP_PROP_FPS)
        num_frames = int(ref_video.get(cv2.CAP_PROP_FRAME_COUNT))
        basename = WINDOW_NAME

        if args.opposite_hands:
            analyser.opposite_hands = True


        if args.output:
            if os.path.isdir(args.output):
                output_fname = os.path.join(args.output, basename)
                output_fname = os.path.splitext(output_fname)[0] + ".mkv"
            else:
                output_fname = args.output
            assert not os.path.isfile(output_fname), output_fname
            output_file = cv2.VideoWriter(
                filename=output_fname,
                # some installation of opencv may not support x264 (due to its license),
                # you can try other format (e.g. MPEG)
                fourcc=cv2.VideoWriter_fourcc(*"MPEG"),
                fps=float(frames_per_second),
                frameSize=(width, height),
                isColor=True,
            )
        assert os.path.isfile(args.ref_video)
        assert os.path.isfile(args.analysis_video)
        for vis_frame in tqdm.tqdm(analyser.run_on_video(ref_video, analysis_video), total=num_frames):
            if args.output:
                output_file.write(vis_frame)
            else:
                cv2.namedWindow(basename, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(basename, 600, 600)
                cv2.imshow(basename, vis_frame)
                if cv2.waitKey(1) == 27:
                    break  # esc to quit
        ref_video.release()
        analysis_video.release()
        if args.output:
            output_file.release()
        else:
            cv2.destroyAllWindows()
