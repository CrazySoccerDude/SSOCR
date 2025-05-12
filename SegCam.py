import cv2
import numpy as np
from ssocr_lib import preprocess, find_digits_positions, recognize_digits_line_method  # Import the recognition method

# Global variables
ref_point = []
cropping = False
roi_selected = False
selected_roi_img = None




def select_roi(event, x, y, flags, param):
    """
    Mouse callback function to select ROI.
    """
    global ref_point, cropping, roi_selected, selected_roi_img, frame_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True
        roi_selected = False
        selected_roi_img = None

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            temp_frame = frame_copy.copy()
            cv2.rectangle(temp_frame, ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("Frame", temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        roi_selected = True

        # Ensure the points are in the correct order (top-left, bottom-right)
        x1, y1 = ref_point[0]
        x2, y2 = ref_point[1]
        roi_x = min(x1, x2)
        roi_y = min(y1, y2)
        roi_w = abs(x1 - x2)
        roi_h = abs(y1 - y2)

        # Ensure ROI coordinates are within frame boundaries
        h, w = frame_copy.shape[:2]
        roi_x = max(0, roi_x)
        roi_y = max(0, roi_y)
        roi_w = min(w - roi_x, roi_w)
        roi_h = min(h - roi_y, roi_h)
        
        ref_point = [(roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h)]

        if roi_w > 0 and roi_h > 0:
            selected_roi_img = frame_copy[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w].copy()
            
            # Magnify the selected_roi_img itself
            zoom_factor = 5
            zoomed_width = int(selected_roi_img.shape[1] * zoom_factor)
            zoomed_height = int(selected_roi_img.shape[0] * zoom_factor)

            if zoomed_width > 0 and zoomed_height > 0:
                selected_roi_img = cv2.resize(selected_roi_img, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)
            else:
                print("Cannot magnify ROI to zero dimensions. Using original size.")

            cv2.imshow("Selected ROI", selected_roi_img) # Now shows magnified ROI

            # Process the ROI using ssocr_lib and display results
            # gray_roi will also be based on the magnified selected_roi_img
            gray_roi = cv2.cvtColor(selected_roi_img, cv2.COLOR_BGR2GRAY)
            
            # Preprocess the ROI and show intermediate steps
            # The threshold value (e.g., 10) might need tuning
            # processed_roi will be magnified
            processed_roi = preprocess(gray_roi, threshold=10, show=True) 
            
            # Find digit positions in the processed ROI
            # The reserved_threshold value (e.g., 5) might need tuning
            # Note: find_digits_positions may raise an AssertionError if no digits are found
            # Positions will be relative to the magnified processed_roi
            try:
                digits_positions = find_digits_positions(processed_roi.copy(), reserved_threshold=5)
            except AssertionError as e:
                print(f"Error finding digits: {e}")
                digits_positions = [] # Continue without drawing boxes if assertion fails

            # Draw bounding boxes on a copy of the selected ROI
            # roi_with_boxes will be a magnified image
            roi_with_boxes = selected_roi_img.copy()
            if digits_positions:
                for pos in digits_positions:
                    # pos is expected to be [(x1, y1), (x2, y2)]
                    # These coordinates are for the magnified image
                    pt1 = (pos[0][0], pos[0][1])
                    pt2 = (pos[1][0], pos[1][1])
                    cv2.rectangle(roi_with_boxes, pt1, pt2, (0, 255, 0), 1) # Thickness might need adjustment for magnified view
            
            cv2.imshow("Detected Digits in ROI", roi_with_boxes) # Now shows magnified ROI with boxes
        else:
            roi_selected = False # Invalid ROI
            print("Invalid ROI selected (width or height is zero). Please try again.")

        # Draw the final ROI on the main frame
        final_frame = frame_copy.copy()
        cv2.rectangle(final_frame, ref_point[0], ref_point[1], (0, 0, 255), 2)
        cv2.imshow("Frame", final_frame)


def main():
    global frame_copy, roi_selected, selected_roi_img, ref_point

    cap = cv2.VideoCapture(0)  # 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", select_roi)

    print("Drag mouse to select ROI. Press 'r' to reset, 's' to save ROI, 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        frame_copy = frame.copy()  # Keep a copy for drawing and ROI extraction

        if roi_selected and len(ref_point) == 2:
            # Extract the current ROI from the live frame
            x1, y1 = ref_point[0]
            x2, y2 = ref_point[1]
            current_roi = frame[y1:y2, x1:x2]

            if current_roi.size > 0:
                # Magnify the ROI
                zoom_factor = 5
                zoomed_width = int(current_roi.shape[1] * zoom_factor)
                zoomed_height = int(current_roi.shape[0] * zoom_factor)
                magnified_roi = cv2.resize(current_roi, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)

                # Convert to grayscale and process
                gray_roi = cv2.cvtColor(magnified_roi, cv2.COLOR_BGR2GRAY)
                processed_roi = preprocess(gray_roi, threshold=10, show=True)

                # Detect digits and draw bounding boxes
                try:
                    digits_positions = find_digits_positions(processed_roi.copy(), reserved_threshold=5)
                except AssertionError:
                    digits_positions = []

                roi_with_boxes = magnified_roi.copy()
                for pos in digits_positions:
                    pt1 = (pos[0][0], pos[0][1])
                    pt2 = (pos[1][0], pos[1][1])
                    cv2.rectangle(roi_with_boxes, pt1, pt2, (0, 255, 0), 1)

                # Recognize digits using line method
                recognized_digits = recognize_digits_line_method(digits_positions, roi_with_boxes, processed_roi)
                print(f"Recognized Digits: {''.join(map(str, recognized_digits))}")

                # Display the processed ROI with recognized digits
                cv2.imshow("Selected ROI", roi_with_boxes)

            # Draw the ROI rectangle on the main frame
            cv2.rectangle(frame, ref_point[0], ref_point[1], (0, 0, 255), 2)

        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('r'):  # Reset ROI
            roi_selected = False
            ref_point = []
            selected_roi_img = None
            if cv2.getWindowProperty("Selected ROI", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Selected ROI")
            if cv2.getWindowProperty("equlizeHist", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("equlizeHist")
            if cv2.getWindowProperty("threshold", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("threshold")
            print("ROI reset.")
        elif key == ord('s'):  # Save ROI
            if roi_selected and selected_roi_img is not None:
                cv2.imwrite("selected_roi.png", selected_roi_img)
                print(f"ROI saved as selected_roi.png. Coordinates: {ref_point}")
            else:
                print("No ROI selected to save.")

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()