#! /usr/bin/env python3

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event

import cv2
import numpy as np
import tqdm

from record3d import IntrinsicMatrixCoeffs, Record3DStream


@dataclass(frozen=True)
class Frame:
    intrinsics: np.ndarray
    rgb: np.ndarray
    depth_raw: np.ndarray  # raw depth in meters
    depth_processed: np.ndarray = None  # clipped depth in mm, missing/clipped set to 0


class Recording:
    recording_path: Path
    frame_idx: int

    def __init__(self, recording_path: Path):
        self.recording_path = recording_path
        self.frame_idx = 0

        self.recording_path.mkdir(parents=True)
        (self.recording_path / "rgb").mkdir()
        (self.recording_path / "depth").mkdir()

    def __get_filename(self, sub_path: str, ext: str = "png") -> Path:
        return str(self.recording_path / sub_path / f"{self.frame_idx:06d}.{ext}")

    def save_frame(self, frame: Frame) -> None:
        cv2.imwrite(self.__get_filename("rgb"), frame.rgb)
        cv2.imwrite(
            self.__get_filename("depth"),
            frame.depth_processed.astype(dtype=np.uint16),
        )
        if self.frame_idx == 0:
            with open(self.recording_path / "intrinsics.txt", "w") as fp:
                np.savetxt(fp, frame.intrinsics)
        self.frame_idx += 1


class DemoApp:
    def __init__(self, base_recording_path: str):
        self.event = Event()
        self.session = None
        self.DEVICE_TYPE__TRUEDEPTH = 0
        self.DEVICE_TYPE__LIDAR = 1

        self.__rotation = 0
        self.__recording: Recording = None
        self.__is_recording = False
        self.__has_quit = False

        self.base_recording_path = Path(base_recording_path)
        self.depth_clip_distance_meters = 3.0

    def set_clip_distance(self, distance: float) -> None:
        self.depth_clip_distance_meters = distance

    def set_rotation(self, rotation: int) -> None:
        assert rotation % 90 == 0, "Must rotate by 0, 90, 180 or 270 degrees."
        self.__rotation = rotation

    def on_new_frame(self) -> None:
        """
        This method is called from non-main thread, therefore cannot be used for presenting UI.
        """
        self.event.set()  # notify the main thread to stop waiting and process new frame.

    def on_stream_stopped(self) -> None:
        print("Stream stopped in Record3D app...")
        self.__has_quit = True

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
        # camera_pose = (
        #     self.session.get_camera_pose()
        # )  # quaternion + world position (accessible via camera_pose.[qx|qy|qz|qw|tx|ty|tz])

        return Frame(intrinsics, rgb, depth)

    def post_processing(self, frame: Frame) -> Frame:
        # copy images
        rgb, depth_raw = frame.rgb.copy(), frame.depth_raw.copy()

        # rotate/flip images
        if self.session.get_device_type() == self.DEVICE_TYPE__TRUEDEPTH:
            rgb = cv2.flip(rgb, 1)
            depth_raw = cv2.flip(depth_raw, 1)

        if self.__rotation:
            rgb = np.rot90(rgb, -self.__rotation / 90)
            depth_raw = np.rot90(depth_raw, -self.__rotation / 90)

        # RGB processing
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # depth processing
        missing_mask = np.logical_or(
            np.isnan(depth_raw), depth_raw > self.depth_clip_distance_meters
        )
        depth_processed = depth_raw.copy()
        depth_processed[missing_mask] = 0
        depth_processed = np.clip(depth_processed, 0, self.depth_clip_distance_meters)
        depth_processed *= 1000

        return Frame(frame.intrinsics, rgb, depth_raw, depth_processed)

    def toggle_recording(self) -> None:
        if self.__is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        # create new directory for recording named after current date and time
        recording_path = self.base_recording_path / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.__recording = Recording(recording_path)
        self.frame_idx = 0
        self.__is_recording = True
        print(f"\nRecording to {recording_path}...")

    def stop_recording(self) -> None:
        self.__is_recording = False
        self.__recording = None
        print("\nRecording stopped.")

    def record_frame(self, frame: Frame) -> None:
        self.__recording.save_frame(frame)

    def process_keypress(self, key: int) -> None:
        if key == ord("q"):
            self.__has_quit = True

        if key == ord("s"):
            self.toggle_recording()

    def stream_iterator(self) -> Frame:
        while not self.__has_quit:
            # wait for new frame to arrive
            self.event.wait()

            # retrieve newly arrived RGBD frame
            frame = self.retrieve_frame()

            # post-process the frame
            frame = self.post_processing(frame)

            # reset the event
            self.event.clear()

            yield frame

    def display_frame(self, frame: Frame) -> None:
        # prepare depth frame for visualization
        depth_colorized = cv2.applyColorMap(
            cv2.convertScaleAbs(255 - frame.depth_processed / 1000 * 255),
            cv2.COLORMAP_PARULA,
        )
        depth_colorized[
            frame.depth_processed == 0
        ] = 0  # set missing depth values to black

        # display RGB-D stream
        rgbd_concat = cv2.hconcat([frame.rgb, depth_colorized])
        cv2.imshow("Record3D", rgbd_concat)
        cv2.setWindowProperty("Record3D", cv2.WND_PROP_TOPMOST, 1)

    def process_stream(self) -> None:
        for frame in tqdm.tqdm(self.stream_iterator()):
            # show the RGB-D stream
            self.display_frame(frame)

            # record frame
            if self.__is_recording:
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
    parser.add_argument(
        "--clip_distance",
        type=float,
        default=3.0,
        help="Maximum depth value (in meters)",
    )
    parser.add_argument("--device", type=int, default=0, help="Device index")
    args = parser.parse_args()

    # start app and connect to device
    app = DemoApp(args.output_dir)
    app.set_rotation(args.rotation)
    app.set_clip_distance(args.clip_distance)
    app.connect_to_device(dev_idx=args.device)

    # start processing the stream
    print("Press `q` in OpenCV window (or Ctrl+C in terminal) to exit...")
    try:
        app.process_stream()
    except KeyboardInterrupt:
        print()
        pass

    print("Done.")
