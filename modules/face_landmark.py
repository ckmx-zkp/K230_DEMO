# ???????????????????????????????
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.PipeLine import ScopedTiming
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import aidemo

class FaceLandMarkApp(AIBase):
    '''
    人脸关键点检测应用类
    Facial landmark detection application class
    '''
    def __init__(self,kmodel_path,model_input_size,rgb888p_size=[1920,1080],display_size=[1920,1080],debug_mode=0):
        '''
        初始化人脸关键点检测应用
        Initialize facial landmark detection application

        参数/Parameters:
            kmodel_path: AI模型路径/AI model path
            model_input_size: 模型输入尺寸/Model input size
            rgb888p_size: 原始图像分辨率/Original image resolution
            display_size: 显示分辨率/Display resolution
            debug_mode: 调试模式/Debug mode
        '''
        super().__init__(kmodel_path,model_input_size,rgb888p_size,debug_mode)
        self.kmodel_path=kmodel_path
        self.model_input_size=model_input_size
        self.rgb888p_size=[ALIGN_UP(rgb888p_size[0],16),rgb888p_size[1]]
        self.display_size=[ALIGN_UP(display_size[0],16),display_size[1]]
        self.debug_mode=debug_mode
        self.matrix_dst=None

        # 实例化AI2D对象/Initialize AI2D object
        self.ai2d=Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,nn.ai2d_format.NCHW_FMT,np.uint8, np.uint8)

    def config_preprocess(self,det,input_image_size=None):
        '''
        配置图像预处理
        Configure image preprocessing

        参数/Parameters:
            det: 人脸检测框/Face detection box
            input_image_size: 输入图像尺寸/Input image size
        '''
        with ScopedTiming("set preprocess config",self.debug_mode > 0):
            ai2d_input_size=input_image_size if input_image_size else self.rgb888p_size

            # 获取仿射变换矩阵/Get affine transformation matrix
            self.matrix_dst = self.get_affine_matrix(det)
            affine_matrix = [self.matrix_dst[0][0],self.matrix_dst[0][1],self.matrix_dst[0][2],
                           self.matrix_dst[1][0],self.matrix_dst[1][1],self.matrix_dst[1][2]]

            # 配置仿射变换/Configure affine transformation
            self.ai2d.affine(nn.interp_method.cv2_bilinear,0, 0, 127, 1,affine_matrix)
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],[1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self,results):
        '''
        后处理关键点检测结果
        Post-process landmark detection results

        参数/Parameters:
            results: 模型输出结果/Model output results

        返回/Returns:
            处理后的关键点坐标/Processed landmark coordinates
        '''
        with ScopedTiming("postprocess",self.debug_mode > 0):
            pred=results[0]
            half_input_len = self.model_input_size[0] // 2

            # 转换关键点坐标/Transform landmark coordinates
            pred = pred.flatten()
            for i in range(len(pred)):
                pred[i] += (pred[i] + 1) * half_input_len

            # 获取逆变换矩阵/Get inverse transformation matrix
            matrix_dst_inv = aidemo.invert_affine_transform(self.matrix_dst)
            matrix_dst_inv = matrix_dst_inv.flatten()

            # 对每个关键点进行逆变换/Apply inverse transform to each landmark
            half_out_len = len(pred) // 2
            for kp_id in range(half_out_len):
                old_x = pred[kp_id * 2]
                old_y = pred[kp_id * 2 + 1]
                new_x = old_x * matrix_dst_inv[0] + old_y * matrix_dst_inv[1] + matrix_dst_inv[2]
                new_y = old_x * matrix_dst_inv[3] + old_y * matrix_dst_inv[4] + matrix_dst_inv[5]
                pred[kp_id * 2] = new_x
                pred[kp_id * 2 + 1] = new_y
            return pred

    def get_affine_matrix(self,bbox):
        '''
        获取仿射变换矩阵
        Get affine transformation matrix

        参数/Parameters:
            bbox: 人脸检测框/Face detection box

        返回/Returns:
            仿射变换矩阵/Affine transformation matrix
        '''
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            x1, y1, w, h = map(lambda x: int(round(x, 0)), bbox[:4])

            # 计算缩放比例和中心点/Calculate scale ratio and center point
            scale_ratio = (self.model_input_size[0]) / (max(w, h) * 1.5)
            cx = (x1 + w / 2) * scale_ratio
            cy = (y1 + h / 2) * scale_ratio
            half_input_len = self.model_input_size[0] / 2

            # 构建仿射矩阵/Build affine matrix
            matrix_dst = np.zeros((2, 3), dtype=np.float)
            matrix_dst[0, 0] = scale_ratio
            matrix_dst[0, 1] = 0
            matrix_dst[0, 2] = half_input_len - cx
            matrix_dst[1, 0] = 0
            matrix_dst[1, 1] = scale_ratio
            matrix_dst[1, 2] = half_input_len - cy
            return matrix_dst
