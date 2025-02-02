import sqlite3
import numpy as np
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn_crfsuite import CRF

class StoryPredictor:
    def __init__(self, db_path, table_name="story_analysis", model_path="crf_model.pkl", max_iterations=50):
        self.db_path = db_path
        self.table_name = table_name
        self.max_iterations = max_iterations
        self.model_path = model_path  # CRF 模型保存路径
        self.crf = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 尝试加载已有的 CRF 模型
        self.load_model()

    def load_data_from_sqlite(self):
        """ 从 SQLite 读取 `s_matrix_embedding` 和 `k_satisfaction` """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"SELECT s_matrix_embedding, k_satisfaction FROM {self.table_name}")
        rows = cursor.fetchall()
        conn.close()

        S_matrices, satisfaction_scores = [], []
        for row in tqdm(rows, desc="📥 加载数据", unit="条"):
            embedding_blob = row[0]
            k_satisfaction = row[1]
            if embedding_blob:
                S_matrix = np.frombuffer(embedding_blob, dtype=np.float32)

                # 🚨 数据检查：如果 S_matrix 无效，直接跳过
                if len(S_matrix) == 0 or np.isnan(S_matrix).any():
                    continue

                S_matrices.append(S_matrix)
                satisfaction_scores.append(k_satisfaction)

        # 🚨 统一向量长度，补零
        max_len = max(len(x) for x in S_matrices)
        S_matrices = [np.pad(x, (0, max_len - len(x)), 'constant') for x in S_matrices]

        return np.array(S_matrices, dtype=np.float32), np.array(satisfaction_scores, dtype=np.float32)

    def train_crf(self, S_matrices, satisfaction_scores):
        """ 训练 CRF，并提供进度反馈，支持中断恢复 """
        X_train, Y_train = [], []
        for i in tqdm(range(len(S_matrices) - 1), desc="🔄 预处理数据", unit="条"):
            X_train.append([{f"feature_{idx}": val} for idx, val in enumerate(S_matrices[i])])
            Y_train.append(["increase" if S_matrices[i + 1][j] > S_matrices[i][j] else "decrease" for j in range(len(S_matrices[i]))])

        print("\n🚀 开始训练 CRF...\n")

        # 🚨 强制 CRF 训练更多轮
        if self.crf is None:
            self.crf = CRF(algorithm='lbfgs', c1=0.1, c2=0.1, max_iterations=self.max_iterations, verbose=True)

        for i in tqdm(range(0, self.max_iterations, 5), desc="⚡ 训练 CRF 进度", unit="轮"):
            self.crf.max_iterations = i + 5
            self.crf.fit(X_train, Y_train)
            self.save_model()
            print(f"✅ 已完成 {i+5}/{self.max_iterations} 轮训练，模型已保存！")

        print("\n✅ CRF 训练完成！")

    def save_model(self):
        """ 保存 CRF 模型 """
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.crf, f)

    def load_model(self):
        """ 尝试加载已保存的 CRF 模型 """
        if os.path.exists(self.model_path):
            print("\n🔄 加载已保存的 CRF 模型...")
            with open(self.model_path, 'rb') as f:
                self.crf = pickle.load(f)
            print("✅ 模型加载成功！继续训练...\n")

    def train(self):
        """ 加载数据并启动训练 """
        print("\n📂 读取数据库数据...")
        S_matrices, satisfaction_scores = self.load_data_from_sqlite()

        print("\n📊 训练数据量:", len(S_matrices))
        print("📊 满意度数据量:", len(satisfaction_scores))

        print("\n🎯 开始训练 CRF...")
        self.train_crf(S_matrices, satisfaction_scores)


if __name__ == "__main__":
    db_path = input("Enter database path: ")
    predictor = StoryPredictor(db_path=db_path, max_iterations=100)
    predictor.train()
