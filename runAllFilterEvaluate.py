#runAllFilterEvaluate.py
import os
import subprocess

# 定义模型路径列表
model_paths = [
   "/home/dgxuser10/cryptonym/deakin_research/models/VGG/PoisonRate_0.01_checkpoints_11.pth",
   "/home/dgxuser10/cryptonym/deakin_research/models/VGG/PoisonRate_0.1_checkpoints_12.pth",
   "/home/dgxuser10/cryptonym/deakin_research/models/VGG/PoisonRate_0.03_checkpoints_12.pth",
   "/home/dgxuser10/cryptonym/deakin_research/models/VGG/PoisonRate_0.05_checkpoints_6.pth",
   "/home/dgxuser10/cryptonym/deakin_research/models/VGG/PoisonRate_0.08_checkpoints_8.pth",
    # 添加更多模型路径
]

# 定义滤镜参数列表
filters = [
    "1.0,1.0,1.0"
    # 添加更多滤镜参数
]

# 其他固定参数
image_folder = "/home/dgxuser10/cryptonym/data/GTSRB_dataset/asr_test_images_NOfilter/"
label_file = "/home/dgxuser10/cryptonym/data/GTSRB_dataset/ASR_annotation.txt"
output_file = "accuracy_results_WASR.txt"
batch_size = 64

# 循环每一个模型和滤镜参数运行评估
for filter_params in filters:
    with open(output_file, 'a') as f:
        for model_path in model_paths:
            # 构建命令行参数
            command = [
                'python', 'filterGenerateAndEvaluate.py',  # 替换为你的评估脚本的实际名称
                '--models', model_path,
                '--triggers',filter_params,
                '--image_folder', image_folder,
                '--label_file', label_file,
                '--batch_size', str(batch_size),
                '--output_file',output_file
            ]
            subprocess.run(command)    
        f.write(f"\n")
        print(f"\n")