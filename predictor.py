# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import atexit
import bisect
import multiprocessing as mp
from collections import deque
import cv2
import torch

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer

from track_balls import get_ball_coordinates, test_ball_dropped

class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            num_gpu = torch.cuda.device_count()
            self.predictor = AsyncPredictor(cfg, num_gpus=num_gpu)
        else:
            self.predictor = DefaultPredictor(cfg)

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.

        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            vis_output = visualizer.draw_panoptic_seg_predictions(
                panoptic_seg.to(self.cpu_device), segments_info
            )
        else:
            if "sem_seg" in predictions:
                vis_output = visualizer.draw_sem_seg(
                    predictions["sem_seg"].argmax(dim=0).to(self.cpu_device)
                )
            if "instances" in predictions:
                instances = predictions["instances"].to(self.cpu_device)
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

        return predictions, vis_output

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def juggling_check(self, predictions, frame):
        """
        Perform ball segmentation and ball drop detection.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                    model. Following fields will be used:
                    "pred_keypoints"
            frame (ndarray): an BGR image of shape (H, W, C), in the range [0, 255].
            
        Yields:
            ndarray: BGR visualizations of each video frame.
        """
        if getattr(self, 'ball_dropped', None) is None:
            # NOTE
            # In the interest of reducing the scope of modifications, this has been
            # implemented here. This is not really a good approach, however.
            # You're much better off initialising these variables in the __init__ method.
            self.ball_dropped = [False, False, False] # Keeping track of whether the balls have been dropped (red, green, orange).
            self.DROPPED_BALL_THRESHOLD = 400 # Number of pixels below the nearest wrist to be considered "dropped".


        if len(predictions) > 0:
            keypoint_names = self.metadata.get("keypoint_names")
            keypoints = predictions.get('pred_keypoints').squeeze()
            left_wrist_index = keypoint_names.index('left_wrist')
            right_wrist_index = keypoint_names.index('right_wrist')
            left_wrist = keypoints[left_wrist_index]
            right_wrist = keypoints[right_wrist_index]

            # Find the locations of the three balls using colour segmentation in the HSV colour space.
            hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            red_center,red_radius,green_center,green_radius,orange_center,orange_radius = get_ball_coordinates(hsv_img)

            # Sometimes the balls are obscured by the juggler's hands. If that's the case, skip the test.
            # Otherwise, check whether the balls have been dropped or not.
            # If a ball has been dropped, update its state in the ball_dropped array, and mark the ball on the screen
            # with a filled in circle instead of an outline.
            if red_radius is not None:
                if test_ball_dropped(red_center, left_wrist, right_wrist, self.DROPPED_BALL_THRESHOLD):
                    self.ball_dropped[0] = True
                    cv2.circle(frame, red_center, red_radius, (0,0,255), cv2.FILLED)
                else:
                    cv2.circle(frame, red_center, red_radius, (0,0,255), 2)
            if green_radius is not None:
                if test_ball_dropped(green_center, left_wrist, right_wrist, self.DROPPED_BALL_THRESHOLD):
                    self.ball_dropped[1] = True
                    cv2.circle(frame, green_center, green_radius, (0,255,0), cv2.FILLED)
                else:
                    cv2.circle(frame, green_center, green_radius, (0,255,0), 2)
            if orange_radius is not None:
                if test_ball_dropped(orange_center, left_wrist, right_wrist, self.DROPPED_BALL_THRESHOLD):
                    self.ball_dropped[2] = True
                    cv2.circle(frame, orange_center, orange_radius, (255,0,0), cv2.FILLED)
                else:
                    cv2.circle(frame, orange_center, orange_radius, (255,0,0), 2)

        # Overlay the current values of the "ball_dropped" array. The values are in "Red, Green, Orange" order.
        cv2.putText(frame, str(self.ball_dropped), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

        return frame

    def run_on_video(self, video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: RGB visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions):
            if "instances" in predictions:
                # Move the predictions tensor off the GPU so we can access the
                # data with the CPU.
                predictions = predictions["instances"].to(self.cpu_device)

                # Perform the checks for the juggling algorithm, and draw
                # the associated overlays.
                vis_frame = self.juggling_check(predictions, frame)

                # Draw the neural network overlay (object bounding box, and body
                # keypoints).
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_BGR2RGB)
                vis_frame = video_visualizer.draw_instance_predictions(vis_frame, predictions)
                vis_frame = vis_frame.get_image()
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
            else:
                vis_frame = frame

            return vis_frame

        frame_gen = self._frame_from_video(video)
        if self.parallel:
            buffer_size = self.predictor.default_buffer_size

            frame_data = deque()

            for cnt, frame in enumerate(frame_gen):
                frame_data.append(frame)
                self.predictor.put(frame)

                if cnt >= buffer_size:
                    frame = frame_data.popleft()
                    predictions = self.predictor.get()
                    yield process_predictions(frame, predictions)

            while len(frame_data):
                frame = frame_data.popleft()
                predictions = self.predictor.get()
                yield process_predictions(frame, predictions)
        else:
            for frame in frame_gen:
                yield process_predictions(frame, self.predictor(frame))


class AsyncPredictor:
    """
    A predictor that runs the model asynchronously, possibly on >1 GPUs.
    Because rendering the visualization takes considerably amount of time,
    this helps improve throughput a little bit when rendering videos.
    """

    class _StopToken:
        pass

    class _PredictWorker(mp.Process):
        def __init__(self, cfg, task_queue, result_queue):
            self.cfg = cfg
            self.task_queue = task_queue
            self.result_queue = result_queue
            super().__init__()

        def run(self):
            predictor = DefaultPredictor(self.cfg)

            while True:
                task = self.task_queue.get()
                if isinstance(task, AsyncPredictor._StopToken):
                    break
                idx, data = task
                result = predictor(data)
                self.result_queue.put((idx, result))

    def __init__(self, cfg, num_gpus: int = 1):
        """
        Args:
            cfg (CfgNode):
            num_gpus (int): if 0, will run on CPU
        """
        num_workers = max(num_gpus, 1)
        self.task_queue = mp.Queue(maxsize=num_workers * 3)
        self.result_queue = mp.Queue(maxsize=num_workers * 3)
        self.procs = []
        for gpuid in range(max(num_gpus, 1)):
            cfg = cfg.clone()
            cfg.defrost()
            cfg.MODEL.DEVICE = "cuda:{}".format(gpuid) if num_gpus > 0 else "cpu"
            self.procs.append(
                AsyncPredictor._PredictWorker(cfg, self.task_queue, self.result_queue)
            )

        self.put_idx = 0
        self.get_idx = 0
        self.result_rank = []
        self.result_data = []

        for p in self.procs:
            p.start()
        atexit.register(self.shutdown)

    def put(self, image):
        self.put_idx += 1
        self.task_queue.put((self.put_idx, image))

    def get(self):
        self.get_idx += 1  # the index needed for this request
        if len(self.result_rank) and self.result_rank[0] == self.get_idx:
            res = self.result_data[0]
            del self.result_data[0], self.result_rank[0]
            return res

        while True:
            # make sure the results are returned in the correct order
            idx, res = self.result_queue.get()
            if idx == self.get_idx:
                return res
            insert = bisect.bisect(self.result_rank, idx)
            self.result_rank.insert(insert, idx)
            self.result_data.insert(insert, res)

    def __len__(self):
        return self.put_idx - self.get_idx

    def __call__(self, image):
        self.put(image)
        return self.get()

    def shutdown(self):
        for _ in self.procs:
            self.task_queue.put(AsyncPredictor._StopToken())

    @property
    def default_buffer_size(self):
        return len(self.procs) * 5
