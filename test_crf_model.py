import pickle
import numpy as np

def test_crf_model():
    # 加载 CRF 模型
    try:
        with open('crf_model.pkl', 'rb') as f:
            crf_model = pickle.load(f)
        print("CRF模型加载成功！")
        
        # 测试数据
        test_s_matrix = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        features = [{f"feature_{idx}": val} for idx, val in enumerate(test_s_matrix)]
        
        # 进行预测
        predictions = crf_model.predict([features])
        print("\n测试S Matrix:", test_s_matrix)
        print("模型预测结果:", predictions[0])
        
        return True
        
    except Exception as e:
        print(f"模型测试失败: {str(e)}")
        return False

if __name__ == "__main__":
    test_crf_model() 