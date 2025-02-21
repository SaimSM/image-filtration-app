import streamlit as st
import cv2
import numpy as np
from io import BytesIO

# Configure the page layout and title
st.set_page_config(page_title="Enhanced Image Filter App", layout="wide")
st.title("Enhanced Image Filter App")
st.markdown("Upload an image and apply a variety of filters with adjustable parameters.")


filter_option = st.sidebar.selectbox(
    "Choose a filter",
    ("Original", "Grayscale", "Blur", "Edge Detection", "Sketch", "Sepia", "Invert", "Contrast Adjustment")
)

# Additional parameter
if filter_option == "Blur":
    blur_kernel = st.sidebar.slider("Blur Kernel Size (odd value)", min_value=3, max_value=31, step=2, value=15)
elif filter_option == "Edge Detection":
    lower_thresh = st.sidebar.slider("Lower Threshold", 50, 150, 100)
    upper_thresh = st.sidebar.slider("Upper Threshold", 150, 300, 200)
elif filter_option == "Contrast Adjustment":
    contrast = st.sidebar.slider("Contrast (alpha)", 0.5, 3.0, 1.0)
    brightness = st.sidebar.slider("Brightness (beta)", -100, 100, 0)

# File uploader for image input
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the uploaded file to a NumPy array and decode the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image based on selected filter
    if filter_option == "Original":
        processed_image = image.copy()
    elif filter_option == "Grayscale":
        processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif filter_option == "Blur":
        processed_image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0)
    elif filter_option == "Edge Detection":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        processed_image = cv2.Canny(gray, lower_thresh, upper_thresh)
    elif filter_option == "Sketch":
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        inv_gray = 255 - gray
        blur_inv = cv2.GaussianBlur(inv_gray, (21, 21), 0)
        processed_image = cv2.divide(gray, 255 - blur_inv, scale=256)
    elif filter_option == "Sepia":
        # Apply a sepia filter using a transformation matrix
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        processed_image = cv2.transform(image, kernel)
        processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    elif filter_option == "Invert":
        processed_image = cv2.bitwise_not(image)
    elif filter_option == "Contrast Adjustment":
        processed_image = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)

    # Display original and processed images side by side
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Image")
        st.image(image, use_column_width=True)
    with col2:
        st.header(f"{filter_option} Filter")
        st.image(processed_image, use_column_width=True)

    # Prepare processed image for download
    # If the processed image is single-channel, convert it to 3-channel for saving as PNG
    if filter_option in ["Grayscale", "Edge Detection", "Sketch"] and len(processed_image.shape) == 2:
        processed_to_save = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)
    else:
        processed_to_save = processed_image

    # Encode the processed image to PNG format
    is_success, buffer = cv2.imencode(".png", cv2.cvtColor(processed_to_save, cv2.COLOR_RGB2BGR))
    io_buf = BytesIO(buffer)

    # Provide a download button
    st.download_button(
        label="Download Processed Image",
        data=io_buf,
        file_name="processed_image.png",
        mime="image/png"
    )
