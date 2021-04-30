import atexit
import bisect
import multiprocessing as mp
import cv2
import numpy as np
import torch
import angles

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.video_visualizer import VideoVisualizer
from detectron2.utils.visualizer import ColorMode, Visualizer


class GolfSwingAnalyser(object):
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

    def run_on_video(self, ref_video, analysis_video):
        """
        Visualizes predictions on frames of the input video.

        Args:
            video (cv2.VideoCapture): a :class:`VideoCapture` object, whose source can be
                either a webcam or a video file.

        Yields:
            ndarray: RGB visualizations of each video frame.
        """
        video_visualizer = VideoVisualizer(self.metadata, self.instance_mode)

        def process_predictions(frame, predictions, reference=None):
            if "instances" in predictions:
                # Move the predictions tensor off the GPU so we can access the
                # data with the CPU.
                predictions = predictions["instances"].to(self.cpu_device)

                # Perform the checks for the juggling algorithm, and draw
                # the associated overlays.
                # Draw the neural network overlay (object bounding box, and body
                # keypoints).
                vis_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                vis_frame = video_visualizer.draw_instance_predictions(vis_frame, predictions)
                vis_frame = vis_frame.get_image()
                vis_frame = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)

                if reference is not None:
                    vis_frame = angles.angles_check(vis_frame, predictions, reference, self.metadata)
                else:
                    vis_frame, generated_reference = angles.create_reference(vis_frame, predictions, self.metadata)
                    return vis_frame, generated_reference
            else:
                vis_frame = frame

            return vis_frame

        def match_frame_height(frame_1, frame_2):
            """
            Adds a border to whichever of frame_1 or frame_2 is shorter and then
            concatenates the frames together
            """
            height_1, height_2 = int(frame_1.shape[0]), int(
                frame_2.shape[0])
            difference = abs(height_1 - height_2)
            border_type = cv2.BORDER_CONSTANT
            if height_1 > height_2:
                extended_frame = cv2.copyMakeBorder(frame_2, 0, difference, 0, 0, border_type, None, [0, 0, 0])
                return np.hstack((frame_1, extended_frame))
            else:
                extended_frame = cv2.copyMakeBorder(frame_1, 0, difference, 0, 0, border_type, None, [0, 0, 0])
                return np.hstack((extended_frame, frame_2))

        def create_frame(ref_frame, analysis_frame):
            """
            Creates the analysed frame from the reference and analysis frame.
            """
            ref_pred, angle_reference = process_predictions(ref_frame, self.predictor(ref_frame))
            analysis_pred = process_predictions(analysis_frame, self.predictor(ref_frame), angle_reference)
            return match_frame_height(ref_pred, analysis_pred)

        frame_gen_ref = self._frame_from_video(ref_video)
        frame_gen_analysis = self._frame_from_video(analysis_video)

        for ref_frame, analysis_frame in zip(frame_gen_ref, frame_gen_analysis):
            yield create_frame(ref_frame, analysis_frame)


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
