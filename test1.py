import cv2
import base64
import os
import time
import threading
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from picamera2 import Picamera2
picam2 = Picamera2()
picam2.preview_configuration.main.size = (800,600)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()


# ✅ Set up Google API Key
os.environ["GOOGLE_API_KEY"] = ""

# ✅ Initialize the Gemini model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# ✅ Function to send the captured image to Gemini for analysis
def analyze_image_with_gemini(image):
    if image is None:
        return "No image to analyze."

    # Convert the captured image to base64
    _, img_buffer = cv2.imencode('.jpg', image)
    image_data = base64.b64encode(img_buffer).decode('utf-8')

    # Create the message with the image
    message = HumanMessage(
    content=[ 
        {"type": "text", "text": "The agent's task is to list object"},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}} 
    ]
)
    

    # Send the message to Gemini and get the response
    response = model.invoke([message])

    return response.content

# Function to continuously capture images every 5 seconds and analyze them
def background_capture(cap):
    while True:
        time.sleep(2)  # Wait for 5 seconds before capturing the next image
        
        im= picam2.capture_array()
        im=cv2.flip(im,-1)
        
        print("Sending the image for analysis...")
        response_content = analyze_image_with_gemini(im)  # Analyze the image with Gemini
        print("Gemini Response: ", response_content)  # Print the response from Gemini

# Main function to continuously show the live webcam feed
def main():
    im= picam2.capture_array()
    im=cv2.flip(im,-1)

#    if not cap.isOpened():
#        print("Error: Unable to access the camera.")
#        return

    # Start a background thread to capture and analyze images every 5 seconds
    capture_thread = threading.Thread(target=background_capture, args=(im,))
    capture_thread.daemon = True  # Daemonize the thread to ensure it exits when the main program exits
    capture_thread.start()

    # Continuously show the live webcam feed
    while True:
        im= picam2.capture_array()
        im=cv2.flip(im,-1)
        frame = cv2.resize(im, (800,600))  # Resize the frame for better display

       

        # Display the webcam feed
        cv2.imshow("Webcam Feed", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
