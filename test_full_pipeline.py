import numpy as np
from face_model import EmotionDetector
from prompt_maker import PromptMaker
import pickle
from transformers import BloomForCausalLM, AutoTokenizer
import os
import torch

class PipelineTester:
    def __init__(self):
        # 加载所有组件
        self.load_components()
        
    def load_components(self):
        """加载所有必要的组件"""
        try:
            # 加载CRF模型
            with open('crf_model.pkl', 'rb') as f:
                self.crf_model = pickle.load(f)
            print("✓ CRF模型加载成功")
            
            # 初始化情绪检测器
            self.emotion_detector = EmotionDetector()
            print("✓ 情绪检测器初始化成功")
            
            # 初始化提示词生成器
            self.prompt_maker = PromptMaker()
            print("✓ 提示词生成器初始化成功")
            
            # 加载BLOOM模型
            print("正在加载语言模型...")
            model_name = "bigscience/bloom-560m"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = BloomForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32
            )
            self.model.eval()
            print("✓ 文本生成模型加载成功")
            
        except Exception as e:
            print(f"组件加载失败: {str(e)}")
            raise
            
    def generate_text(self, prompt, max_length=1000):
        """使用BLOOM模型生成文本"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        
        outputs = self.model.generate(
            inputs.input_ids,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
    def test_pipeline(self):
        """测试完整的生成流程"""
        print("\n开始测试完整流程...")
        
        try:
            # 1. 获取最新的S Matrix
            current_s_matrix = self.prompt_maker.get_latest_s_matrix()
            print(f"\n当前S Matrix: {current_s_matrix}")
            
            # 2. 生成提示词
            prompt = self.prompt_maker.make_prompt(current_s_matrix)
            print(f"\n生成的提示词:\n{prompt}")
            
            # 3. 生成故事
            response = self.generate_text(prompt)
            print(f"\n生成的故事:\n{response}")
            
            # 4. 获取用户满意度
            satisfaction = self.emotion_detector.get_satisfaction_score()
            print(f"\n检测到的用户满意度: {satisfaction}")
            
            # 5. 保存数据
            self.prompt_maker.save_story_data(current_s_matrix, response, satisfaction)
            
            # 6. 生成下一个S Matrix预测
            features = [{f"feature_{idx}": val} for idx, val in enumerate(current_s_matrix)]
            next_s_matrix_pred = self.crf_model.predict([features])[0]
            print(f"\n预测的下一个状态: {next_s_matrix_pred}")
            
            return {
                'current_s_matrix': current_s_matrix,
                'prompt': prompt,
                'story': response,
                'satisfaction': satisfaction,
                'next_state': next_s_matrix_pred
            }
            
        except Exception as e:
            print(f"测试过程中出错: {str(e)}")
            raise

def main():
    print("开始系统测试...\n")
    try:
        tester = PipelineTester()
        results = tester.test_pipeline()
        
        print("\n测试完成！")
        print("="*50)
        print("测试结果摘要：")
        print(f"- 生成故事长度: {len(results['story'])}字")
        print(f"- 用户满意度: {results['satisfaction']:.2f}")
        print("="*50)
    except Exception as e:
        print(f"测试失败: {str(e)}")

if __name__ == "__main__":
    main() 