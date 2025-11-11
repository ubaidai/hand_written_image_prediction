import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras


# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model('model.h5')
        return model
    except:
        st.error("Model file 'model.h5' not found. Please ensure the model is in the same directory.")
        return None

def preprocess_image(image):
    """Preprocess the uploaded image to match MNIST format"""
    # Convert to grayscale
    img = image.convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # Invert colors if needed (MNIST has white digits on black background)
    # Check if the image has more white pixels than black
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # Normalize pixel values to [0, 1]
    img_array = img_array.astype('float32') / 255
    
    # Reshape to (1, 784) for the model
    img_array = img_array.reshape(1, 784)
    
    return img_array

def main():
    st.set_page_config(
        page_title="Handwritten Digit Classifier",
        page_icon="🔢",
        layout="centered"
    )
    
    st.title("🔢 Handwritten Digit Classifier")
    st.write("Upload an image of a handwritten digit (0-9) and the model will predict what number it is!")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.stop()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg', 'bmp']
    )
    
    col1, col2 = st.columns(2)
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width=300)
        
        # Preprocess and predict
        with st.spinner('Processing image...'):
            processed_image = preprocess_image(image)
            
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)
            predicted_digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100
        
        with col2:
            st.subheader("Prediction")
            st.markdown(f"### Predicted Digit: **{predicted_digit}**")
            st.markdown(f"Confidence: **{confidence:.2f}%**")
            
            # Show probability distribution
            st.write("Probability Distribution:")
            prob_df = {
                'Digit': list(range(10)),
                'Probability': [f"{p*100:.2f}%" for p in prediction[0]]
            }
            st.dataframe(prob_df, hide_index=True, use_container_width=True)
        
        # Show preprocessed image
        with st.expander("View Preprocessed Image (28x28)"):
            preprocessed_display = processed_image.reshape(28, 28)
            st.image(preprocessed_display, width=200, caption="Preprocessed Image")
    
    # Instructions
    with st.sidebar:
        st.header("📝 Instructions")
        st.write("""
        1. Upload an image of a handwritten digit (0-9)
        2. The model will automatically process and predict the digit
        3. You'll see the predicted digit with confidence score
        
        **Tips for best results:**
        - Use clear, single digit images
        - The digit should be centered
        - Works best with black/dark digits on white/light background
        - Image will be automatically resized to 28x28 pixels
        """)
        
        st.header("📊 Model Info")
        st.write("""
        - Dataset: MNIST
        - Architecture: Dense Neural Network
        - Layers: 2 Dense layers (512 and 10 neurons)
        - Accuracy: ~98%
        """)

if __name__ == "__main__":
    main()