import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import tensorflow as tf

# Function to authenticate user
def authenticate(username, password):
    # In a real application, validate username and password against stored credentials
    # For demonstration, hardcoding credentials
    if username == 'Nabil' and password == 'password':
        return True
    else:
        return False

# Streamlit app
def main():
    st.title("Diabetic Retinopathy Predictor")

    # Login page
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    # Check if login button is clicked
    if login_button:
        if authenticate(username, password):
            st.success(f"Welcome, {username}!")
            st.session_state.username = username
            st.session_state.is_authenticated = True
        else:
            st.error("Invalid username or password")

    # If user is authenticated
    if "is_authenticated" in st.session_state and st.session_state.is_authenticated:
        st.write(f"Welcome, {st.session_state.username}!")

        # File upload for image
        uploaded_file = st.file_uploader("Upload retinal image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

            # Load the trained model
            model_path = "C:/Users/Nabil/OneDrive/Desktop/Nabil/DSU/fourth year/Major Proejct/cnn.model"
            model = tf.keras.models.load_model(model_path)

            # Define the labels for grading
            grades = {
                0: 'No_DR',
                1: 'Mild_DR',
                2: 'Moderate_DR',
                3: 'Severe_DR',
                4: 'Proliferative_DR'
            }

            # Function to preprocess the input image
            def preprocess_image(image):
                # Resize the image to the required input size of the model
                resized_image = image.resize((224, 224))
                # Convert the image to array and normalize the pixel values
                img_array = np.array(resized_image) / 255.0
                # Expand the dimensions to match the input shape of the model
                img_array = np.expand_dims(img_array, axis=0)
                return img_array

            # Function to enhance contrast
            def enhance_contrast(image):
                enhancer = ImageEnhance.Contrast(image)
                enhanced_image = enhancer.enhance(3)  # Increase contrast by 50%
                return enhanced_image

            # Function to detect blobs and bright spots in the image
            def detect_blobs(image):
                # Convert the image to grayscale
                gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                # Threshold the grayscale image
                _, threshold = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
                # Find contours in the thresholded image
                contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Check the number of contours
                num_contours = len(contours)
                # Check for bright spots indicating severe DR
                mean_brightness = np.mean(gray)
                has_bright_spots = mean_brightness > 100  # Threshold for brightness
                # Determine the diagnosis based on the number of contours and brightness
                if num_contours == 2 or not has_bright_spots:
                    return 'No_DR', None
                elif num_contours > 1 and num_contours <= 2:
                    return 'Mild_DR', ["Name: Dr. N S Muralidhar", ["Clinic: Retina Institute Of Karnataka", "Address: 122, 5th Main Road, Landmark: Near Uma Theater, Bangalore", "Phone: 080 7196 6821", "Google Maps: https://www.google.com/maps/dir//12.976812168993668,77.59786248207092"]]
                elif num_contours > 2 and num_contours <= 10 and has_bright_spots:
                    return 'Moderate_DR', ["Name: Dr. Devaraj M",["Clinic: Dr. Agarwal's Eye Hospital", "Address: NKS Prime, #60/417, 20th Main Road, 1st Block, Rajajinagar, Below Rajajinagar Metro Station, Bangalore, Karnataka, Bangalore", "Phone: 080 7196 6843", "Google Maps: https://www.google.com/maps/dir//13.001089,77.5527394/@13.001076,77.4703375,12z?entry=ttu "]]
                elif num_contours > 4 and num_contours <= 20 and has_bright_spots:
                    return 'Severe_DR', ["Name: Dr. Rahul Jain",["Clinic: Dr. Jain's Eye Clinic", "Address: 2nd Block, BTM 4th Stage, Landmark: Besides Shreya Medical Center Hospital, Bangalore", "Phone:080 4680 1971", "Google Maps:https://www.google.com/maps/dir//12.8877485,77.6097614/@12.8878635,77.5272436,12z?entry=ttu"]]
                elif has_bright_spots:
                    return 'Proliferative_DR', ["Name: Dr. Rinku Das", ["Clinic: My Vision Eye Clinic", "Address: 10/3, 2055, 1st Floor, Kaikondrahalli, Landmark: Behind Anand Sweets, Above Stone Icecream and Opposite Fire Station, Bangalore", "Phone: 080 7191 0148", "Google Maps:https://www.google.com/maps/dir//12.916098757511,77.673055278139"]]
                else:
                    return 'Unknown', None

            # Enhance contrast of the image
            image = Image.open(uploaded_file)  # Open image file
            enhanced_image = enhance_contrast(image)  # Enhance contrast

            # Detect blobs and bright spots in the image
            diagnosis, doctors = detect_blobs(enhanced_image)

            # Display the result with detailed explanation
            result_text = ""
            if diagnosis == 'No_DR':
                result_text = "No diabetic retinopathy detected. However, it's important to continue regular eye check-ups as prevention is key. Maintain a healthy lifestyle and monitor blood sugar levels closely."
                bg_color = "#b3ffb3"  # Light green background
            elif diagnosis == 'Mild_DR':
                result_text = "Mild diabetic retinopathy detected. It's crucial to manage blood sugar levels effectively through diet, exercise, and medication as prescribed by your healthcare provider. Regular eye exams are essential to monitor progression."
                bg_color = "#ffffcc"  # Light yellow background
            elif diagnosis == 'Moderate_DR':
                result_text = "Moderate diabetic retinopathy detected. Consult with your healthcare provider immediately for further evaluation and treatment options. Strict blood sugar control and possibly laser treatment may be necessary to prevent vision loss."
                bg_color = "#ffd699"  # Light orange background
            elif diagnosis == 'Severe_DR':
                result_text = "Severe diabetic retinopathy detected. Urgent intervention is required to prevent vision loss. Seek immediate medical attention from an eye specialist for advanced treatment options such as injections or surgery."
                bg_color = "#ff9999"  # Light red background
            elif diagnosis == 'Proliferative_DR':
                result_text = "Proliferative diabetic retinopathy detected. This is a serious condition that requires prompt and aggressive treatment to prevent blindness. Contact your healthcare provider immediately for specialized care."
                bg_color = "#ff6666"  # Red background
            else:
                result_text = "Unknown result. Please consult with a healthcare professional for further evaluation."
                bg_color = "#f0f0f0"  # Light gray background

            # Display the result in a styled text box with colored background
            st.markdown(
                f"<div style='background-color:{bg_color}; padding:20px; border-radius:5px;'><h3>Result: {diagnosis}</h3><p>{result_text}</p></div>",
                unsafe_allow_html=True
            )

            # Display doctor suggestions if available
            if doctors:
                st.write("Suggested doctors in Bengaluru:")
                for doctor in doctors:
                    if isinstance(doctor, list):
                        st.write("- ", doctor[0])
                        for i in range(1, len(doctor)):
                            st.write(f"   {doctor[i]}")
                    else:
                        st.write("- ", doctor)

if __name__ == "__main__":
    main()
