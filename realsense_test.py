"""Quick RealSense check: lists devices, then shows color + depth streams. Press q to quit."""
import numpy as np
import cv2
import pyrealsense2 as rs

devices = rs.context().query_devices()
if len(devices) == 0:
    raise SystemExit("No RealSense device found. Check cable/port (lsusb should show 8086:0bxx).")
for d in devices:
    print(d.get_info(rs.camera_info.name),
          "| serial", d.get_info(rs.camera_info.serial_number),
          "| fw", d.get_info(rs.camera_info.firmware_version),
          "| USB", d.get_info(rs.camera_info.usb_type_descriptor))

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline.start(config)
align = rs.align(rs.stream.color)

try:
    while True:
        frames = align.process(pipeline.wait_for_frames())
        color, depth = frames.get_color_frame(), frames.get_depth_frame()
        if not color or not depth:
            continue
        color_img = np.asanyarray(color.get_data())
        depth_img = cv2.applyColorMap(
            cv2.convertScaleAbs(np.asanyarray(depth.get_data()), alpha=0.03), cv2.COLORMAP_JET)
        h, w = color_img.shape[:2]
        dist = depth.get_distance(w // 2, h // 2)
        cv2.circle(color_img, (w // 2, h // 2), 4, (0, 255, 0), -1)
        cv2.putText(color_img, f"center: {dist:.3f} m", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("RealSense (color | depth)", np.hstack((color_img, depth_img)))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
