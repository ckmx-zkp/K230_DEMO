"""CSI capture without Display/OSD; frame storage lives through inference."""
import time
from media.sensor import Sensor, CAM_CHN_ID_0, CAM_CHN_ID_1
from media.media import MediaManager


class HeadlessPipeline:
    def __init__(self, sensor_id, size, preview=False):
        self.sensor_id = sensor_id
        self.size = size
        self.sensor = None
        self.frame = None
        self.media_ready = False
        self.running = False
        self.preview = preview
        self.display_ready = False
        self.rgb888p_size = size
        self.display_size = size
        self.osd_img = None

    def create(self):
        self.sensor = Sensor(id=self.sensor_id)
        self.sensor.reset()
        self.sensor.set_framesize(width=self.size[0], height=self.size[1],
                                  chn=CAM_CHN_ID_1)
        self.sensor.set_pixformat(Sensor.RGBP888, chn=CAM_CHN_ID_1)
        if self.preview:
            from media.display import Display
            self.sensor.set_framesize(width=self.size[0], height=self.size[1],
                                      chn=CAM_CHN_ID_0)
            self.sensor.set_pixformat(Sensor.RGB565, chn=CAM_CHN_ID_0)
            Display.init(Display.VIRT, width=self.size[0], height=self.size[1],
                         osd_num=1, to_ide=True)
            self.display_ready = True
        MediaManager.init()
        self.media_ready = True
        self.sensor.run()
        self.running = True

    def get_frame(self):
        self.frame = self.sensor.snapshot(chn=CAM_CHN_ID_1)
        return self.frame.to_numpy_ref()

    def begin_overlay(self):
        # Draw on a fresh RGB565 frame; never clear the camera background.
        self.osd_img = self.sensor.snapshot(chn=CAM_CHN_ID_0)

    def show_image(self):
        from media.display import Display
        Display.show_image(self.osd_img)

    def destroy(self):
        try:
            if self.running:
                self.sensor.stop()
        finally:
            self.running = False
            self.frame = None
            if self.display_ready:
                from media.display import Display
                Display.deinit()
                self.display_ready = False
            self.osd_img = None
            if self.media_ready:
                time.sleep_ms(50)
                MediaManager.deinit()
                self.media_ready = False
