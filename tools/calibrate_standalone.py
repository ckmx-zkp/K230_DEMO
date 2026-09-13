"""CanMV IDE 独立标定：直接运行，无需同时运行 main.py。
默认自动采集五类；只重采一类可改 LABELS = ('happy',)。
每类保存即写入 TF 卡，未采集的已有类别保持不变。
"""
import sys
import os
import time
import gc

LABELS = ('neutral', 'happy', 'angry', 'sad', 'surprise')
APP_DIR = '/sdcard/app/MyVisionHub'
while APP_DIR in sys.path:
    sys.path.remove(APP_DIR)
sys.path.insert(0, APP_DIR)
for name in ('config', 'camera_pipeline', 'modules.face_det', 'modules.face_pose',
             'modules.face_landmark', 'modules.face_target', 'modules.expression', 'modules'):
    if name in sys.modules:
        del sys.modules[name]

import config
import ulab.numpy as np
from camera_pipeline import HeadlessPipeline
from modules.face_det import FaceDetApp
from modules.face_pose import FacePoseApp
from modules.face_landmark import FaceLandMarkApp
from modules.face_target import FaceTarget
from modules.expression import Expression, LABELS as VALID_LABELS

HINTS = {
    'need_single_face': 'Keep ONE complete face in view',
    'clipped_face': 'Move back; keep face away from edges',
    'face_camera': 'Look straight at the camera',
    'Face too small': 'Move a little closer',
    'landmarks_outside_frame': 'Move back; center your face',
}


def fresh_profile_path(base):
    index = 1
    while True:
        candidate = base + '.sample_%04d.json' % index
        try:
            os.stat(candidate)
        except OSError as error:
            if error.args[0] == 2:
                return candidate
            raise
        index += 1


def copy_verified(source, destination):
    with open(source, 'rb') as f:
        data = f.read()
    with open(destination, 'wb') as f:
        f.write(data)
    with open(destination, 'rb') as f:
        if f.read() != data:
            raise OSError('Calibration copy verification failed')


def banner(pl, title, detail):
    pl.begin_overlay()
    # Only a small bottom panel; no face boxes, landmarks or recognition logs.
    y = pl.size[1] - 66
    pl.osd_img.draw_rectangle(0, y, pl.size[0], 66, color=(0, 0, 0), fill=True)
    pl.osd_img.draw_string_advanced(8, y+3, 22, title, color=(255, 255, 255))
    pl.osd_img.draw_string_advanced(8, y+34, 18, detail, color=(255, 255, 255))
    pl.show_image()


def main():
    pl = det = pose_model = landmark = None
    img = points = None
    active_profile = config.EXPRESSION_PROFILE_PATH
    try:
        print('Standalone calibration v2: unique sample filenames')
        if not LABELS or any(label not in VALID_LABELS for label in LABELS):
            raise ValueError('Invalid LABELS')
        for path in (config.FACE_DET_KMODEL, config.FACE_POSE_KMODEL,
                     config.FACE_LANDMARK_KMODEL, config.ANCHORS_PATH):
            os.stat(path)
        expression = Expression()
        # Preserve the previous active file before publishing any new samples.
        try:
            os.stat(active_profile)
        except OSError as error:
            if error.args[0] != 2:
                raise
        else:
            copy_verified(active_profile, fresh_profile_path(active_profile))
        # This standalone session owns sampling; ignore queued main.py requests.
        expression.poll_request = lambda: None
        pl = HeadlessPipeline(config.SENSOR_ID, config.RGB888P_SIZE, preview=True)
        pl.create()
        anchors = np.fromfile(config.ANCHORS_PATH, dtype=np.float)
        anchors = anchors.reshape((config.ANCHOR_LEN, config.DET_DIM))
        det = FaceDetApp(config.FACE_DET_KMODEL, config.FACE_DET_INPUT_SIZE,
                        anchors=anchors, confidence_threshold=config.CONFIDENCE_THRESHOLD,
                        nms_threshold=config.NMS_THRESHOLD,
                        rgb888p_size=config.RGB888P_SIZE, display_size=config.RGB888P_SIZE)
        det.config_preprocess()
        pose_model = FacePoseApp(config.FACE_POSE_KMODEL, config.FACE_POSE_INPUT_SIZE,
                                 rgb888p_size=config.RGB888P_SIZE,
                                 display_size=config.RGB888P_SIZE)
        landmark = FaceLandMarkApp(config.FACE_LANDMARK_KMODEL, [192, 192],
                                  rgb888p_size=config.RGB888P_SIZE,
                                  display_size=config.RGB888P_SIZE)
        print('Calibration ready: neutral / happy / angry / sad / surprise')
        for index, label in enumerate(LABELS):
            # Even the old board module can rename to this unused destination.
            config.EXPRESSION_PROFILE_PATH = fresh_profile_path(active_profile)
            target = FaceTarget()
            expression.reset()
            expression.pending = label
            expression.start_ms = time.ticks_ms()
            print('Prepare:', label)
            # Preview stays live during the five-second preparation period.
            prepare = time.ticks_ms()
            while time.ticks_diff(time.ticks_ms(), prepare) < 5000:
                left = 5 - time.ticks_diff(time.ticks_ms(), prepare)//1000
                banner(pl, '%d/%d %s' % (index+1, len(LABELS), label),
                       'Prepare expression: %d s' % left)
                time.sleep_ms(30)
            expression.start_ms = time.ticks_ms()
            while expression.pending:
                started = time.ticks_ms()
                img = pl.get_frame()
                boxes = det.run(img)
                count = len(boxes) if boxes is not None else 0
                box, changed = target.update(boxes if count else [], *config.RGB888P_SIZE)
                if changed:
                    expression.reset()
                points = None
                pose = None
                if count == 1 and box is not None:
                    pose_model.config_preprocess(box)
                    rotation, angles = pose_model.run(img)
                    pose = {'pitch': float(angles[0]), 'yaw': float(angles[1])}
                    landmark.config_preprocess(box)
                    points = landmark.run(img)
                result = expression.update(points, box, pose, count)
                reason = result.get('reason', '')
                if reason == 'calibration_timeout':
                    # Retry this class automatically, without collecting bad samples.
                    expression.pending = label
                    expression.start_ms = time.ticks_ms()
                    detail = 'Retry: center face and hold expression'
                elif reason == 'calibrating':
                    detail = 'Hold still: %d / 15' % result.get('samples', 0)
                else:
                    detail = HINTS.get(reason, 'Adjust face: ' + reason)
                banner(pl, '%d/%d %s' % (index+1, len(LABELS), label), detail)
                img = points = None
                gc.collect()
                remaining = 200 - time.ticks_diff(time.ticks_ms(), started)
                if remaining > 0:
                    time.sleep_ms(remaining)
            # Keep the numbered file as a recovery copy; main.py uses the usual path.
            copy_verified(config.EXPRESSION_PROFILE_PATH, active_profile)
            print('Activated:', active_profile, 'backup:', config.EXPRESSION_PROFILE_PATH)
            saved_at = time.ticks_ms()
            while time.ticks_diff(time.ticks_ms(), saved_at) < 1500:
                banner(pl, label + ' saved', 'Next expression follows automatically')
                time.sleep_ms(30)
        # Remove an obsolete queued request only after successful calibration.
        try:
            os.remove(config.EXPRESSION_REQUEST_PATH)
        except OSError:
            pass
        print('Calibration complete. Saved:', active_profile)
        print('Now run main.py to test recognition.')
        finished = time.ticks_ms()
        while time.ticks_diff(time.ticks_ms(), finished) < 3000:
            banner(pl, 'Calibration complete', 'Run main.py to test recognition')
            time.sleep_ms(30)
    except KeyboardInterrupt:
        print('Calibration stopped; saved classes retained.')
    except Exception as error:
        if str(error) == 'IDE interrupt':
            print('Calibration stopped; saved classes retained.')
        else:
            sys.print_exception(error)
    finally:
        config.EXPRESSION_PROFILE_PATH = active_profile
        img = points = None
        for model in (landmark, pose_model, det):
            if model is not None:
                try:
                    model.deinit()
                except Exception:
                    pass
        if pl is not None:
            pl.destroy()
        gc.collect()


if __name__ == '__main__':
    main()
