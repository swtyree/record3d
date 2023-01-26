import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import tqdm

from record3d import IntrinsicMatrixCoeffs, Record3DStream


@dataclass
class Frame:
    rgb: np.ndarray
    depth: np.ndarray
    intrinsics: np.ndarray
    camera_pose: np.ndarray


class DemoApp:
    def __init__(self, base_recording_path: str):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1
        self.rotation = 0
        self.is_recording = False
        self.quit = False
        self.base_recording_path = Path(base_recording_path)
        self.recording_path = None
        self.frame_idx = -1

    def set_rotation(self, rotation: int) -> None:
        assert rotation % 90 == 0, "Must rotate by 0, 90, 180 or 270 degrees."
        self.rotation = rotation

    def on_new_frame(self) -> None:
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self) -> None:
        print("Stream stopped in Record3D app...")
        self.quit = True

    def connect_to_device(self, dev_idx: int) -> None:
        devs = Record3DStream.get_connected_devices()
        if len(devs) != 1:
            print("{} device(s) found".format(len(devs)))
            for dev in devs:
                print("\tID: {}\n\tUDID: {}\n".format(dev.product_id, dev.udid))

        if len(devs) <= dev_idx:
            raise RuntimeError(
                "Cannot connect to device #{}, try different index.".format(dev_idx)
            )

        dev = devs[dev_idx]
        self.session = Record3DStream()
        self.session.on_new_frame = self.on_new_frame
        self.session.on_stream_stopped = self.on_stream_stopped
        self.session.connect(dev)

    def get_intrinsic_mat_from_coeffs(
        self, coeffs: IntrinsicMatrixCoeffs
    ) -> np.ndarray:
        return np.array(
            [[coeffs.fx, 0, coeffs.tx], [0, coeffs.fy, coeffs.ty], [0, 0, 1]]
        )

    def retrieve_frame(self) -> Frame:
        rgb = self.session.get_rgb_frame()
        depth = self.session.get_depth_frame()
        intrinsics = self.get_intrinsic_mat_from_coeffs(
            self.session.get_intrinsic_mat()
        )
        camera_pose = (
            self.session.get_camera_pose()
        )  # quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])

        return Frame(rgb, depth, intrinsics, camera_pose)

    def post_processing(self, frame: Frame) -> Frame:
        rgb, depth = frame.rgb, frame.depth

        if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
            depth = cv2.flip(depth, 1)
            rgb = cv2.flip(rgb, 1)

        if self.rotation:
            depth = np.rot90(depth, -self.rotation / 90)
            rgb = np.rot90(rgb, -self.rotation / 90)

        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        return Frame(rgb, depth, frame.intrinsics, frame.camera_pose)

    def toggle_recording(self) -> None:
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        # create new directory for recording named after current date and time
        self.recording_path = self.base_recording_path / time.strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
        self.frame_idx = 0
        self.recording_path.mkdir(parents=True)
        self.is_recording = True
        print(f"\nRecording to {self.recording_path}...")

    def stop_recording(self) -> None:
        self.is_recording = False
        self.recording_path = None
        print("\nRecording stopped.")

    def record_frame(self, frame: Frame) -> None:
        pass

    def process_keypress(self, key: int) -> None:
        if key == ord("q"):
            self.quit = True

        if key == ord("s"):
            self.toggle_recording()

    def stream_iterator(self) -> Frame:
        while not self.quit:
            # wait for new frame to arrive
            self.event.wait()

            # retrieve newly arrived RGBD frame
            frame = self.retrieve_frame()

            # post-process the frame
            frame = self.post_processing(frame)

            # reset the event
            self.event.clear()

            yield frame

    def process_stream(self) -> None:
        for frame in tqdm.tqdm(self.stream_iterator()):
            # show the RGB-D stream
            cv2.imshow("RGB", frame.rgb)
            cv2.imshow("Depth", frame.depth)

            # record frame
            if self.is_recording:
                self.record_frame(frame)

            # check for key press
            key = cv2.waitKey(1)
            self.process_keypress(key)


if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="recordings/",
        help="Output directory for recordings",
    )
    parser.add_argument(
        "--rotation",
        type=int,
        default=0,
        help="Rotation of the output images",
        choices=[0, 90, 180, 270],
    )
    parser.add_argument("--device", type=int, default=0, help="Device index")
    args = parser.parse_args()

    # start app and connect to device
    app = DemoApp(args.output_dir)
    app.set_rotation(args.rotation)
    app.connect_to_device(dev_idx=args.device)

    # start processing the stream
    print("Press `q` in OpenCV window (or Ctrl+C in terminal) to exit...")
    try:
        app.process_stream()
    except KeyboardInterrupt:
        print()
        pass

    print("Done.")
