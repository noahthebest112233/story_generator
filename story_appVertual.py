import streamlit as st
import numpy as np
import time
import openai
import random

try:
    import cv2
    CAMERA_AVAILABLE = True
except ImportError:
    CAMERA_AVAILABLE = False
    st.warning("摄像头功能不可用 - 使用基础模式")

# Set OpenAI API key
openai.api_key = "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"

# 添加新的导入和人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

class NovelGenerator:
    def __init__(self):
        pass

    def generate_chapter(self, s_matrix, chapter_num, previous_content=""):
        """Generate a chapter"""
        try:
            prompt = self._create_chapter_prompt(s_matrix, chapter_num, previous_content)
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional fantasy novelist, skilled at creating engaging continuous stories based on style requirements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                top_p=0.9,
                frequency_penalty=0.5,
                presence_penalty=0.5
            )
            
            story = response.choices[0].message.content
            return story, prompt
            
        except Exception as e:
            st.error(f"Generation failed: {str(e)}")
            return "Generation failed", "Error"

    def _create_chapter_prompt(self, s_matrix, chapter_num, previous_content):
        """Create chapter prompt"""
        style_desc = self._get_style_description(s_matrix)
        
        prompt = f"""Create Chapter {chapter_num} of a fantasy novel.

Previous content:
{previous_content[:500] + '...' if len(previous_content) > 500 else previous_content}

Style requirements:
{style_desc}

Requirements:
1. Maintain continuity with previous content
2. Follow the style preferences
3. Include rich descriptions
4. Write in engaging English

Continue with Chapter {chapter_num}:"""
        
        return prompt

    def _get_style_description(self, s_matrix):
        """Interpret style matrix"""
        style_pairs = [
            ("Epic", "Slice of Life", s_matrix[0]),
            ("Tense", "Relaxed", s_matrix[1]),
            ("Complex", "Simple", s_matrix[2]),
            ("Serious", "Humorous", s_matrix[3]),
            ("Mysterious", "Clear", s_matrix[4]),
            ("Adventurous", "Peaceful", s_matrix[5]),
            ("Combat-focused", "Peace-focused", s_matrix[6]),
            ("Melancholic", "Cheerful", s_matrix[7])
        ]
        
        descriptions = []
        for pos, neg, value in style_pairs:
            if value > 0.5:
                descriptions.append(f"- Leaning towards {pos} ({value:.1f})")
            else:
                descriptions.append(f"- Leaning towards {neg} ({1-value:.1f})")
                
        return "\n".join(descriptions)

def initialize_session_state():
    """Initialize session state"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.generator = NovelGenerator()
        st.session_state.chapters = []
        st.session_state.current_s_matrix = np.array([0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
        st.session_state.full_content = ""

def get_random_emotion_values():
    """Generate random emotion values"""
    emotions = {
        "Happy": random.uniform(0, 1),
        "Sad": random.uniform(0, 1),
        "Angry": random.uniform(0, 1),
        "Surprised": random.uniform(0, 1),
        "Fearful": random.uniform(0, 1),
        "Neutral": random.uniform(0, 1)
    }
    return emotions

def process_frame(frame):
    """Process frame and detect faces"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Convert frame to RGB for Streamlit
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(rgb_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    # Generate simulated emotion values based on face detection
    if len(faces) > 0:
        emotions = {
            "Attention": np.random.uniform(0.6, 1.0),
            "Engagement": np.random.uniform(0.5, 0.9),
            "Focus": np.random.uniform(0.4, 0.8)
        }
    else:
        emotions = {
            "Attention": 0.1,
            "Engagement": 0.1,
            "Focus": 0.1
        }
        
    return rgb_frame, emotions

def main():
    initialize_session_state()
    
    st.title("🌟 AI Story Generator")
    st.write("个性化故事生成系统")
    
    try:
        if CAMERA_AVAILABLE:
            # Sidebar: Face Detection
            st.sidebar.header("Live Face Detection")
            
            # Initialize video capture
            cap = cv2.VideoCapture(0)
            
            # Create placeholders
            frame_placeholder = st.sidebar.empty()
            emotion_placeholders = {emotion: st.sidebar.empty() 
                                  for emotion in ["Attention", "Engagement", "Focus"]}
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    # Process frame
                    processed_frame, emotions = process_frame(frame)
                    
                    # Display frame
                    frame_placeholder.image(processed_frame)
                    
                    # Display emotion values
                    for emotion, value in emotions.items():
                        emotion_placeholders[emotion].progress(
                            value, 
                            text=f"{emotion}: {value:.2f}"
                        )
                        
            # Release video capture
            cap.release()
        else:
            st.sidebar.info("当前环境不支持摄像头功能")
            
        # Main Interface
        if st.button("Generate Next Chapter", key="generate"):
            with st.spinner('Creating...'):
                chapter_num = len(st.session_state.chapters) + 1
                content, prompt = st.session_state.generator.generate_chapter(
                    st.session_state.current_s_matrix,
                    chapter_num,
                    st.session_state.full_content
                )
                
                st.session_state.full_content += "\n\n" + content
                
                st.session_state.chapters.append({
                    'chapter_num': chapter_num,
                    'content': content,
                    'prompt': prompt,
                    's_matrix': st.session_state.current_s_matrix.copy(),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                })
        
        # Display Chapters
        for chapter in st.session_state.chapters:
            st.subheader(f"Chapter {chapter['chapter_num']}")
            st.write(chapter['content'])
            st.caption(f"Created: {chapter['timestamp']}")
            st.divider()
        
        # Export Feature
        if st.session_state.chapters:
            if st.button("Export Story"):
                st.download_button(
                    label="Download Story",
                    data=st.session_state.full_content,
                    file_name="fantasy_novel.txt",
                    mime="text/plain"
                )

    except Exception as e:
        st.error(f"Runtime error: {str(e)}")

if __name__ == "__main__":
    main() 