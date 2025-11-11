import streamlit as st

st.title("Testing TensorFlow Installation")

try:
    import tensorflow as tf
    st.success(f"✅ TensorFlow installed! Version: {tf.__version__}")
    
    try:
        from tensorflow import keras
        st.success("✅ Keras imported successfully!")
    except Exception as e:
        st.error(f"❌ Keras import failed: {e}")
        
except Exception as e:
    st.error(f"❌ TensorFlow not installed: {e}")

st.write("---")
st.write("Requirements.txt content:")
st.code("""
streamlit
tensorflow-cpu==2.13.0
pillow
numpy
""")
