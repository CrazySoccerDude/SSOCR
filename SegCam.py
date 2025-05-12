import cv2
import numpy as np
from ssocr_lib import preprocess, find_digits_positions, recognize_digits_area_method, recognize_digits_line_method  # Use area method

# Global variables
ref_point = []
cropping = False
roi_selected = False
selected_roi_img = None

# Global variables for thresholds
H_threshold = 35
V_threshold = 120
# Global variables for time-averaging filter
frame_buffer = []  # Buffer to store the last N frames
average_frame_count = 5  # Number of frames to average

# Global variable for binarization threshold
binarization_threshold = 10  # Default threshold for binarization

# Global variable for height-to-width ratio
H_W_Ratio = 4  # Default ratio




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
                digits_positions = find_digits_positions(processed_roi.copy(), H_threshold = 20, V_threshold = 20)
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


def apply_time_averaging_filter(processed_roi):
    """
    Apply a time-averaging filter to the processed ROI.
    """
    global frame_buffer, average_frame_count

    # Add the current frame to the buffer
    frame_buffer.append(processed_roi)

    # Keep only the last `average_frame_count` frames
    if len(frame_buffer) > average_frame_count:
        frame_buffer.pop(0)

    # Compute the average of the frames in the buffer
    averaged_frame = np.mean(frame_buffer, axis=0).astype("uint8")
    return averaged_frame

def main():
    global frame_copy, roi_selected, selected_roi_img, ref_point, H_threshold, V_threshold, frame_buffer, average_frame_count, binarization_threshold, H_W_Ratio

    cap = cv2.VideoCapture(0)  # 0 for default camera

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", select_roi)

    print("Drag mouse to select ROI. Press 'r' to reset, 's' to save ROI, 'q' to quit.")
    print("Use 'h/H' to decrease/increase H_threshold, 'v/V' to decrease/increase V_threshold.")
    print("Use 'n/N' to decrease/increase the number of frames for averaging.")
    print("Use 't/T' to decrease/increase the binarization threshold.")
    print("Use 'r/R' to decrease/increase H_W_Ratio.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        frame_copy = frame.copy()  # Keep a copy for drawing and ROI extraction

        # Display the current threshold values and averaging frame count on the frame
        cv2.putText(frame, f"H_threshold: {H_threshold}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"V_threshold: {V_threshold}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Averaging Frames: {average_frame_count}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Binarization Threshold: {binarization_threshold}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"H_W_Ratio: {H_W_Ratio:.1f}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if roi_selected and len(ref_point) == 2:
            # Extract the current ROI from the live frame
            x1, y1 = ref_point[0]
            x2, y2 = ref_point[1]
            current_roi = frame[y1:y2, x1:x2]

            if current_roi.size > 0:
                # Apply time-averaging filter first
                gray_roi = cv2.cvtColor(current_roi, cv2.COLOR_BGR2GRAY)
                averaged_roi = apply_time_averaging_filter(gray_roi)

                # Magnify the averaged ROI
                zoom_factor = 5
                zoomed_width = int(averaged_roi.shape[1] * zoom_factor)
                zoomed_height = int(averaged_roi.shape[0] * zoom_factor)
                magnified_roi = cv2.resize(averaged_roi, (zoomed_width, zoomed_height), interpolation=cv2.INTER_LINEAR)

                # Process the averaged ROI with the adjustable binarization threshold
                processed_roi = preprocess(magnified_roi, threshold=binarization_threshold, show=False)

                # Detect digits and draw bounding boxes
                try:
                    digits_positions = find_digits_positions(processed_roi.copy(), H_threshold=H_threshold, V_threshold=V_threshold)
                except AssertionError:
                    digits_positions = []

                roi_with_boxes = cv2.cvtColor(magnified_roi, cv2.COLOR_GRAY2BGR)  # Convert to BGR for visualization

                # Recognize digits using area method
                #recognized_digits = recognize_digits_area_method(digits_positions, roi_with_boxes, processed_roi)
                #print(f"Recognized Digits: {''.join(map(str, recognized_digits))}")

                # Recognize digits using line method and display results
                try:
                    line_digits_positions = find_digits_positions(processed_roi.copy(), H_threshold=H_threshold, V_threshold=V_threshold)
                    line_recognized_digits = recognize_digits_line_method(line_digits_positions, roi_with_boxes, processed_roi)
                    print(f"Line Method Recognized Digits: {''.join(map(str, line_recognized_digits))}")
                except AssertionError:
                    line_recognized_digits = []
                    print("Line Method: No digits detected.")

                # Display the processed ROI with recognized digits
                cv2.imshow("Selected ROI", roi_with_boxes)

                # Display intermediate steps for debugging
                cv2.imshow("Averaged ROI", averaged_roi)
                cv2.imshow("Processed ROI", processed_roi)

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
            frame_buffer = []  # Reset the frame buffer
            if cv2.getWindowProperty("Selected ROI", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Selected ROI")
            if cv2.getWindowProperty("Averaged ROI", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Averaged ROI")
            if cv2.getWindowProperty("Processed ROI", cv2.WND_PROP_VISIBLE) >= 1:
                cv2.destroyWindow("Processed ROI")
            print("ROI reset.")
        elif key == ord('s'):  # Save ROI
            if roi_selected and selected_roi_img is not None:
                cv2.imwrite("selected_roi.png", selected_roi_img)
                print(f"ROI saved as selected_roi.png. Coordinates: {ref_point}")
            else:
                print("No ROI selected to save.")
        elif key == ord('h'):  # Decrease H_threshold
            H_threshold = max(1, H_threshold - 1)
            print(f"H_threshold decreased to {H_threshold}")
        elif key == ord('H'):  # Increase H_threshold
            H_threshold += 1
            print(f"H_threshold increased to {H_threshold}")
        elif key == ord('v'):  # Decrease V_threshold
            V_threshold = max(1, V_threshold - 1)
            print(f"V_threshold decreased to {V_threshold}")
        elif key == ord('V'):  # Increase V_threshold
            V_threshold += 1
            print(f"V_threshold increased to {V_threshold}")
        elif key == ord('n'):  # Decrease averaging frame count
            average_frame_count = max(1, average_frame_count - 1)
            print(f"Averaging Frames decreased to {average_frame_count}")
        elif key == ord('N'):  # Increase averaging frame count
            average_frame_count += 1
            print(f"Averaging Frames increased to {average_frame_count}")
        elif key == ord('t'):  # Decrease binarization threshold
            binarization_threshold = max(1, binarization_threshold - 1)
            print(f"Binarization Threshold decreased to {binarization_threshold}")
        elif key == ord('T'):  # Increase binarization threshold
            binarization_threshold += 1
            print(f"Binarization Threshold increased to {binarization_threshold}")
        elif key == ord('w'):  # Decrease H_W_Ratio
            H_W_Ratio = max(0.1, H_W_Ratio - 0.1)
            print(f"H_W_Ratio decreased to {H_W_Ratio:.1f}")
        elif key == ord('W'):  # Increase H_W_Ratio
            H_W_Ratio += 0.1
            print(f"H_W_Ratio increased to {H_W_Ratio:.1f}")

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()