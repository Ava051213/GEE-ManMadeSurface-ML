"""
完整风机检测工作流
1. 数据预处理
2. 上传到GEE
3. 生成GEE预测脚本
4. 本地模型训练（可选）
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """运行命令并检查结果"""
    print(f"🚀 {description}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ {description} failed:")
        print(result.stderr)
        sys.exit(1)
    else:
        print(f"✅ {description} completed")
    return result

def main():
    print("=== 风机检测完整工作流 ===")
    
    # 步骤1: 数据预处理
    run_command("python prepare_data.py", "数据预处理")
    
    # 步骤2: 上传到GEE
    run_command("python upload_to_gee.py", "上传数据到GEE")
    
    # 步骤3: 生成GEE预测脚本
    run_command("python generate_gee_prediction_scripts.py", "生成GEE预测脚本")
    
    # 步骤4: 本地训练（可选）
    choice = input("是否进行本地模型训练？(y/n): ")
    if choice.lower() == 'y':
        run_command("python local_training.py", "本地模型训练")
    
    print("\n🎉 完整工作流执行完成！")
    print("\n下一步操作：")
    print("1. 打开 gee_scripts/ 目录中的.js文件")
    print("2. 在GEE Code Editor中运行预测脚本")
    print("3. 查看预测结果和评估指标")

if __name__ == "__main__":
    main()