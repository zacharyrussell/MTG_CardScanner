import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows path


import cv2
import pytesseract
import numpy as np

def preprocess_image(frame):
    """ Converts the frame to grayscale and applies thresholding to improve OCR accuracy. """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh

def extract_card_name(frame):
    """ Extracts text from the image using Tesseract OCR. """
    processed = preprocess_image(frame)
    text = pytesseract.image_to_string(processed, config='--psm 6')  # PSM 6: Assume a single block of text
    return text.strip()

def main():
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Define the region where the card name is expected (adjust as needed)
        height, width, _ = frame.shape
        roi = frame[int(height * 0.05):int(height * 0.15), int(width * 0.2):int(width * 0.8)]  # Crop top portion

        card_name = extract_card_name(roi)

        # Display extracted text
        cv2.putText(frame, f"Card Name: {card_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("MTG Card Scanner", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
