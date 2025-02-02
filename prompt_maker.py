import sqlite3
import json
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ 1. 加载 SentenceTransformer 模型（必须和数据库存储时使用的模型一致）
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

class PromptMaker:
    def __init__(self):
        self.db_path = 'story_data.db'
        self.style_map = {
            # S Matrix 各维度对应的风格特征
            0: ("史诗", "日常"),
            1: ("紧张", "轻松"),
            2: ("复杂", "简单"),
            3: ("严肃", "幽默"),
            4: ("神秘", "明朗"),
            5: ("冒险", "平静"),
            6: ("战斗", "和平"),
            7: ("悲伤", "欢乐")
        }
    
    def get_latest_s_matrix(self):
        """获取最新的 S Matrix"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT s_matrix_embedding FROM story_analysis ORDER BY id DESC LIMIT 1")
            result = cursor.fetchone()
            conn.close()
            
            if result:
                # 将存储的字符串转换回numpy数组
                return np.array(json.loads(result[0]))
            else:
                # 如果没有历史数据，返回默认值
                return np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
                
        except Exception as e:
            print(f"获取S Matrix时出错: {str(e)}")
            return np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    
    def save_story_data(self, s_matrix, story_text, satisfaction_score):
        """保存故事数据到数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 将numpy数组转换为JSON字符串存储
            s_matrix_json = json.dumps(s_matrix.tolist())
            
            cursor.execute("""
                INSERT INTO story_analysis (s_matrix_embedding, story_text, satisfaction_score)
                VALUES (?, ?, ?)
            """, (s_matrix_json, story_text, satisfaction_score))
            
            conn.commit()
            conn.close()
            print("故事数据保存成功！")
            
        except Exception as e:
            print(f"保存数据时出错: {str(e)}")
    
    def make_prompt(self, s_matrix):
        """将 S Matrix 转换为文字提示词"""
        style_prompts = []
        for idx, value in enumerate(s_matrix):
            if value >= 0.5:
                style_prompts.append(self.style_map[idx][0])
            else:
                style_prompts.append(self.style_map[idx][1])
                
        prompt = f"""请以下面的风格写一段西幻故事：
风格要求：{', '.join(style_prompts)}
要求：
1. 保持情节连贯
2. 字数在500-1000之间
3. 要有细腻的描写
4. 符合西方奇幻设定
"""
        return prompt

# ✅ 运行代码
if __name__ == "__main__":
    db_path = "wjps-C.db"
    print("\n📂 读取数据库数据...")

    # 获取最新的 S_matrix_embedding
    latest_s_matrix = get_latest_s_matrix(db_path)
    
    if latest_s_matrix is not None:
        print("\n🔄 计算最相似文本...")
        prompt = find_closest_text(db_path, latest_s_matrix)

        if prompt:
            print("\n🎯 生成的 Prompt：")
            print(prompt)
