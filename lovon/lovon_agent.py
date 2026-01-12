
import os
import torch
from ultralytics import YOLO
import logging
import cv2

# Use package-relative imports when possible, fallback to top-level imports so the
# module can be executed as a script or imported as an installed package.
try:
    # when imported as `import lovon` this relative import will work
    from .models_cxn_025.api_object_extraction_transformer import ObjectExtractionAPI
    from .models_cxn_025.api_language2motion_transformer import MotionPredictor
except Exception:
    # fallback for running the file directly (python lovon/lovon_agent.py)
    from models_cxn_025.api_object_extraction_transformer import ObjectExtractionAPI
    from models_cxn_025.api_language2motion_transformer import MotionPredictor

logging.getLogger('ultralytics').setLevel(logging.ERROR)

class LovonAgent:
    def __init__(self, yolo_model_dir=None,
                 wh_scale_factor=1.0,  # 缩放因子，可以根据需要调整
                 tokenizer_path=None,
                 object_extraction_model_path=None,
                 language2motion_model_path=None,
                 velocity_scale=1.0):
        # Resolve package base directory so defaults point to files inside the
        # installed/checked-out `lovon` package. Caller-supplied paths override these.
        package_dir = os.path.dirname(os.path.abspath(__file__))
        # sensible defaults inside the lovon package (override by passing args)
        if yolo_model_dir is None:
            yolo_model_dir = os.path.join(package_dir, "yolo-models", "yolo11x.pt")
        self.wh_scale_factor = wh_scale_factor
        if tokenizer_path is None:
            tokenizer_path = os.path.join(package_dir, "models_cxn_025", "tokenizer_language2motion_n1000000")
        if object_extraction_model_path is None:
            object_extraction_model_path = os.path.join(package_dir, "models_cxn_025", "model_object_extraction_n1000000_d64_h4_l2_f256_msl64_hold_success")
        if language2motion_model_path is None:
            language2motion_model_path = os.path.join(package_dir, "models_cxn_025", "model_language2motion_n1000000_d256_h8_l4_f1024_msl64_hold_success_cxn025_beta10_no_hold")
        self.velocity_scale = velocity_scale
        # 核心功能初始化
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # initialize subcomponents with resolved paths
        self.object_extractor = ObjectExtractionAPI(
            model_path=object_extraction_model_path,
            tokenizer_path=tokenizer_path
        )
        self.yolo_model = YOLO(yolo_model_dir)
        self.motion_predictor = MotionPredictor(
            model_path=language2motion_model_path,
            tokenizer_path=tokenizer_path
        )
        
        # 初始化状态变量
        self.mission_instruction_0 = "run to the person at speed of 0.5 m/s"
        self.mission_instruction_1 = self.mission_instruction_0
        self.state = {
            "predicted_object": "NULL",
            "confidence": [0.00],
            "object_xyn": [0.00, 0.00],
            "object_whn": [0.00, 0.00],
            "mission_state_in": "running",
            "search_state_in": "had_searching_0",
        }
        # 初始化提取的物体
        self.extracted_object = self.object_extractor.predict(self.mission_instruction_1)


    def _update_yolo_model_results(self):
        """更新YOLO模型结果"""
        self.results = self.yolo_model(self.image)

    def _yolo_image_post_process(self):
        """yolo图像检测结果处理"""
        # YOLO检测
        # results = self.yolo_model(image)

        # 筛选符合extracted_object的结果
        self.aim_results_box = []
        detections = []
        for result in self.results:
            for box in result.boxes:
                class_name = result.names[int(box.cls)]
                if class_name == self.extracted_object:
                    detections.append({
                        "object": class_name,
                        "confidence": float(box.conf),
                        "xyn": box.xywhn[0][:2].tolist(),
                        "whn": box.xywhn[0][2:].tolist()
                    })
                    self.aim_results_box.append(box)

        # 更新检测结果
        self.predicted_object = "NULL"
        if detections:
            best = max(detections, key=lambda x: x["confidence"])
            # 添加缩放因子并将w和h限制在0-1.0之间
            scaled_wh = [min(1.0, max(0.0, x * self.wh_scale_factor)) for x in best["whn"]]
            self.state.update({
                "predicted_object": best["object"],
                "confidence": [best["confidence"]],
                "object_xyn": best["xyn"],
                "object_whn": scaled_wh
            })
            self.predicted_object = best["object"]
        else:
            # 当没有检测到物体时，保持一致的格式
            self.state.update({
                "predicted_object": "NULL",
                "confidence": [0.00],
                "object_xyn": [0.00, 0.00],
                "object_whn": [0.00, 0.00]  # 已经在0-1范围内，无需缩放
            })

        # return aim_results_box, predicted_object

    def _update_extract_object(self):
        """提取物体"""
        # 使用ObjectExtractionAPI提取物体
        self.extracted_object = self.object_extractor.predict(self.mission_instruction_1)

    def _update_motion_control(self):
        """获取运动控制指令，更新mission_state_in"""
        input_data = {
            "mission_instruction_0": self.mission_instruction_0,
            "mission_instruction_1": self.mission_instruction_1,
            **self.state
        }
        prediction = self.motion_predictor.predict(input_data)
        self.state["mission_state_in"] = prediction["predicted_state"]
        self.state["search_state_in"] = prediction["search_state"]
        self.motion_vector = prediction["motion_vector"]
        # 缩放运动向量
        if self.motion_vector is not None:
            self.motion_vector = [v * self.velocity_scale for v in self.motion_vector]
    
    def run(self, image, mission_instruction_0=None, mission_instruction_1=None):
        """运行LovonAgent"""
        self.image = image

        # 更新YOLO模型结果
        self._update_yolo_model_results()

        # 更新提取的物体
        if mission_instruction_1 is not None:
            self.mission_instruction_1 = mission_instruction_1
        if mission_instruction_0 is not None:
            self.mission_instruction_0 = mission_instruction_0
        self._update_extract_object()

        # 处理YOLO检测结果
        self._yolo_image_post_process()
        
        # 更新运动控制指令
        self._update_motion_control()

        return self.state, self.motion_vector



if __name__ == "__main__":
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    vlm_models_folder = os.path.join(current_dir, 'models_cxn_025')
    lovon_agent = LovonAgent(
        yolo_model_dir=os.path.join(current_dir, "yolo-models/yolo11x.pt"),
        tokenizer_path=os.path.join(vlm_models_folder, "tokenizer_language2motion_n1000000"),
        object_extraction_model_path=os.path.join(vlm_models_folder, "model_object_extraction_n1000000_d64_h4_l2_f256_msl64_hold_success"),
        language2motion_model_path=os.path.join(vlm_models_folder, "model_language2motion_n1000000_d256_h8_l4_f1024_msl64_hold_success_cxn025")
    )

    # Example usage:
    # 读取图像
    image = cv2.imread("chair.png")
    # 运行LOVON Agent
    state, motion_vector = lovon_agent.run(image=image, mission_instruction_0="run to the chair at speed of 0.4 m/s", mission_instruction_1="run to the person at speed of 0.4 m/s")
    print(state)
    print(motion_vector)  
    state, motion_vector = lovon_agent.run(image=image, mission_instruction_0="run to the chair at speed of 0.4 m/s", mission_instruction_1="run to the person at speed of 0.4 m/s")
    print(state)
    print(motion_vector)  