import sqlite3

def create_database():
    # 连接到数据库（如果不存在则创建）
    conn = sqlite3.connect('story_data.db')
    cursor = conn.cursor()
    
    # 创建故事分析表
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS story_analysis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        s_matrix_embedding TEXT,
        story_text TEXT,
        satisfaction_score FLOAT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    conn.commit()
    conn.close()
    print("数据库和表创建成功！")

if __name__ == "__main__":
    create_database() 