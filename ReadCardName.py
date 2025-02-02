import cv2
import easyocr
import numpy as np

# Initialize EasyOCR
reader = easyocr.Reader(['en'])

def preprocess_image(frame):
    """ Enhance text clarity for OCR. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    # Apply CLAHE for better contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Resize the image for better OCR detection
    enhanced = cv2.resize(enhanced, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    return enhanced

def extract_card_name(image):
    """ Use EasyOCR to detect text in the image. """
    processed = preprocess_image(image)

    # Run EasyOCR
    results = reader.readtext(processed, detail=0)  # Set detail=0 to return only text

    if results:
        return results[0].strip()
    return "No text detected"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    height, width, _ = frame.shape

    # Define a smaller ROI (center-top section for card name)
    x1, y1, x2, y2 = int(width * 0.35), int(height * 0.05), int(width * 0.70), int(height * 0.12)

    # Draw ROI boundary on the live feed
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Show the live feed with the overlay
    cv2.imshow("MTG Card Reader", frame)

    # Handle key events
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):  # Capture and process image on spacebar press
        roi = frame[y1:y2, x1:x2]  # Extract the ROI
        detected_card_name = extract_card_name(roi)  # Process the image
        print(f"Detected Card Name: {detected_card_name}")  # Print the detected name

        # Overlay detected card name on the live feed (for next frame)
        cv2.putText(frame, f"Detected: {detected_card_name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
 