import threading
import cv2
from deepface import DeepFace

# Initialize video capture
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Global variables
counter = 0
face_match = False
lock = threading.Lock()  # Ensure thread-safe updates
ref_img = cv2.imread("reference.jpg")

def check_face(frame):
    """
    Verifies if the captured frame matches the reference image.
    Updates the global `face_match` flag in a thread-safe manner.
    """
    global face_match
    try:
        # Use DeepFace's Face Recognition with less resource consumption
        result = DeepFace.verify(frame, ref_img.copy(), model_name='VGG-Face')  # Specify model to reduce load
        with lock:
            face_match = result['verified']
    except Exception as e:
        print(f"Error during face verification: {e}")
        with lock:
            face_match = False

def process_frame(frame):
    """
    Adds a label to the frame indicating whether the face matches or not.
    """
    with lock:
        label = "MATCH!" if face_match else "NO MATCH!"
        color = (0, 255, 0) if face_match else (0, 0, 255)
    cv2.putText(frame, label, (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 3)
    return frame

def main():
    global counter
    while True:
        ret, frame = cap.read()
        if ret:
            # Process every 30 frames to reduce the processing load
            if counter % 30 == 0:
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            counter += 1

            # Add match/no match label
            frame = process_frame(frame)
            cv2.imshow("video", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
