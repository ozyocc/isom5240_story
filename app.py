import streamlit as st
from PIL import Image
from transformers import pipeline, BlipProcessor, BlipForConditionalGeneration
from gtts import gTTS
import io
import random

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Magic Picture Storyteller",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 Magic Picture Storyteller")
st.write("Upload a picture, and let's create a fun story together!")

# ==========================================
# 2. MODEL LOADING (CACHED)
# ==========================================
@st.cache_resource
def load_caption_model():
    """Loads BLIP processor and model directly."""
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

@st.cache_resource
def load_story_pipeline():
    """Loads GPT-2 text generation pipeline."""
    return pipeline("text-generation", model="gpt2")

processor, blip_model = load_caption_model()
story_pipe = load_story_pipeline()

# ==========================================
# 3. CORE PROCESSING FUNCTIONS
# ==========================================
def generate_caption(image: Image.Image) -> str:
    """Generates an image caption using native BLIP classes."""
    inputs = processor(image, return_tensors="pt")
    out = blip_model.generate(**inputs, max_new_tokens=50)
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def generate_child_story(caption: str) -> str:
    """
    Expands an image caption into a non-repetitive child-friendly story 
    strictly between 50 and 100 words.
    """
    clean_caption = caption.strip().rstrip('.')
    
    # List of varied story starters
    starters = [
        f"Once upon a time, {clean_caption}. Suddenly, a gentle spark of magic filled the air.",
        f"In a cozy little town, everyone noticed {clean_caption}. It was the start of an unexpected adventure.",
        f"High up in the sky, a friendly cloud looked down and saw {clean_caption}. What a wonderful day to explore!",
        f"Far away in a happy meadow, {clean_caption}. All the animals gathered around to celebrate."
    ]
    
    # Pick a random starter every time the button is clicked
    prompt = random.choice(starters)
    
    # Text generation with anti-repetition parameters
    story_output = story_pipe(
        prompt, 
        max_new_tokens=85, 
        min_new_tokens=45, 
        num_return_sequences=1,
        do_sample=True,
        temperature=0.9,           # Slight increase in creativity
        top_p=0.92,                 # Nucleus sampling (picks from top cumulative probability)
        top_k=40,                   # Filters out low-probability words
        repetition_penalty=1.8,     # Strongly penalizes word/phrase repetition
        no_repeat_ngram_size=3,     # Strictly blocks repeated 3-word combinations
        pad_token_id=50256
    )
    
    story = story_output[0]['generated_text']
    
    # Ensure text ends at the last complete sentence
    if '.' in story:
        story = story[:story.rfind('.') + 1]
    
    words = story.split()
    
    # Enforce upper bound (Max 100 words)
    if len(words) > 100:
        story = " ".join(words[:100])
        if '.' in story:
            story = story[:story.rfind('.') + 1]
            
    # Enforce lower bound (Min 50 words)
    words = story.split()
    if len(words) < 50:
        extra_sentence = " Afterward, they played under the blue sky until the sun set happily."
        story += extra_sentence
        
    return story

def text_to_speech(text: str) -> io.BytesIO:
    """Converts input text into an MP3 audio stream using gTTS."""
    tts = gTTS(text=text, lang='en', slow=False)
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    return fp

# ==========================================
# 4. USER INTERFACE & WORKFLOW
# ==========================================
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    if st.button("✨ Generate Magic Story"):
        with st.spinner("Analyzing image and creating a story..."):
            # Step 1: Caption Generation & Large Font Display
            caption = generate_caption(image)
            st.subheader("🔍 What I see in the image:")
            
            # Larger Font HTML Display (22px bold font with colored background box)
            st.markdown(
                f"""
                <div style="
                    font-size: 22px; 
                    font-weight: 600; 
                    color: #1F2937; 
                    background-color: #F0F2F6; 
                    padding: 15px; 
                    border-radius: 8px; 
                    margin-bottom: 20px;">
                    {caption.capitalize()}
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Step 2: Story Generation
            story = generate_child_story(caption)
            word_count = len(story.split())
            
            st.subheader("📖 Story Time:")
            st.write(story)
            st.caption(f"📏 **Word Count:** {word_count} words")
            
            # Step 3: Text-to-Speech
            audio_fp = text_to_speech(story)
            st.subheader("🔊 Listen to the Story:")
            st.audio(audio_fp, format="audio/mp3")