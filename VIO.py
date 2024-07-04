import cv2
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import glob

#Storable parameters
gridx = 7
gridy = 10
grid_mm = 25
mtx = None
dist = None
selected_exposure_value = -4
selected_camera_index = 1
binarize_threshold = 100
dpi = 70

# Global frames
grid_frame = None
background_frame = None
scan_frame = None
difference_frame = None
sum_frame = None
backgroundSubtract_frame = None
binarized_frame = None
calibrated_frame = None


def capture_and_average_frames(num_frames=10):
    """
    Webcams often have very noisy image capture.  This function captures multiple frames and 
    averages them to reduce noise.
    Auto white balance is turned off and exposure set to a fixed value, but some cameras
    don't actually allow turning white balance off.

    Args:
        num_frames (int, optional): The number of frames to capture. Defaults to 10.

    Returns:
        numpy.ndarray: The averaged frame as a numpy array.
    """
    cap = cv2.VideoCapture(selected_camera_index)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Turn off auto white balance
    cap.set(cv2.CAP_PROP_EXPOSURE, selected_exposure_value)
    
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(frame.astype(np.float32))
    cap.release()
    
    if frames:
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
    else:
        avg_frame = None
    return avg_frame

def update_image(label, capture_function, storage_variable_name, scale=1.0):
    """
    Capture an image using the provided function, convert it for Tkinter display, and update the label.

    Args:
        label (Tkinter.Label): The label widget to display the image on.
        capture_function (function): Function to capture the image.
        storage_variable_name (str): Global variable name to store the captured image.
        scale (float, optional): Scale factor for resizing the image. Defaults to 1.0 (no resizing).
    """
    frame = capture_function()
    if frame is not None:
        globals()[storage_variable_name] = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

def calculate_difference(frame1, frame2):
    """
    Calculate the absolute difference between two frames

    Args:
        frame1 (numpy.ndarray): First frame
        frame2 (numpy.ndarray): Second frame

    Returns:
        numpy.ndarray: The difference frame
    """
    global difference_frame
    difference_frame = cv2.absdiff(frame1, frame2)
    return difference_frame

def color_sum_to_bw(frame):
    """
    Take a color image, sum the 3 colors together for each pixel, and convert that to black and white.

    Args:
        frame (numpy.ndarray): Input color frame

    Returns:
        numpy.ndarray: Black and white frame
    """
    global sum_frame
    if frame is not None:
        sum_color = np.sum(frame.astype(np.float32), axis=2)
        normalized = cv2.normalize(sum_color, None, 0, 255, cv2.NORM_MINMAX)
        sum_frame = normalized.astype(np.uint8)
        return sum_frame
    return None

def pause_for_debug():
    """
    Pause the program execution to allow for debugging.
    """
    pass

def display_histogram(frame):
    """
    Display the histogram of an image which can be either color or black and white.

    Args:
        frame (numpy.ndarray): Input frame
    """
    if frame is not None:
        if len(frame.shape) == 3 and frame.shape[2] == 3:  # Color image
            # Calculate histogram for each color channel
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
                plt.plot(hist, color=color)
                plt.xlim([0, 256])
            plt.title('Histogram for color image')
        elif len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):  # Black and white image
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            plt.plot(hist, color='gray')
            plt.xlim([0, 256])
            plt.title('Histogram for black and white image')
        plt.show()
    else:
        print("No frame data to display histogram.")

def background_subtraction(background_frame, target_frame):
    """
    Apply OpenCV background subtraction method to two frames.

    Args:
        background_frame (numpy.ndarray): The background frame.
        target_frame (numpy.ndarray): The target frame to apply background subtraction.

    Returns:
        numpy.ndarray: The mask of the target frame after background subtraction.
    """
    # Create a background subtractor object
    backSub = cv2.createBackgroundSubtractorMOG2()
    
    # Apply the background subtractor to the background frame to train the model
    backSub.apply(background_frame)
    
    # Apply the background subtractor to the target frame to get the foreground mask
    fg_mask = backSub.apply(target_frame)
    
    return fg_mask

def binarize_image(frame, threshold):
    """
    Binarize a black and white image using a specified threshold.

    Args:
        frame (numpy.ndarray): Input black and white frame.
        threshold (int): Threshold value for binarization.

    Returns:
        numpy.ndarray: Binarized frame.
    """
    if frame is not None:
        # Check if the image is indeed black and white
        if len(frame.shape) == 2 or (len(frame.shape) == 3 and frame.shape[2] == 1):
            # Apply threshold to binarize the image
            _, binarized_frame = cv2.threshold(frame, threshold, 255, cv2.THRESH_BINARY)
            return binarized_frame
        else:
            print("Error: The provided frame is not in black and white format.")
            return None
    else:
        print("No frame data provided.")
        return None

def save_parameters():
    # Save the parameters to a file
    with open('parameters.txt', 'w') as file:
        file.write(f"Grid X: {gridx}\n")
        file.write(f"Grid Y: {gridy}\n")
        file.write(f"Grid mm: {grid_mm}\n")
        file.write(f"Binarize Threshold: {binarize_threshold}\n")
        file.write(f"DPI: {dpi}\n")
        file.write(f"Exposure: {selected_exposure_value}\n")
        file.write(f"Camera Index: {selected_camera_index}\n")
        file.write(f"Calibration Matrix: {mtx}\n")
        file.write(f"Distortion Coefficients: {dist}\n")

def load_parameters():
    global gridx, gridy, grid_mm, binarize_threshold, dpi, selected_exposure_value, selected_camera_index, mtx, dist
    # Load the parameters from a file
    with open('parameters.txt', 'r') as file:
        gridx = int(file.readline().split(': ')[1])
        gridy = int(file.readline().split(': ')[1])
        grid_mm = int(file.readline().split(': ')[1])
        binarize_threshold = int(file.readline().split(': ')[1])
        dpi = int(file.readline().split(': ')[1])
        selected_exposure_value = int(file.readline().split(': ')[1])
        selected_camera_index = int(file.readline().split(': ')[1])
        mtx = np.array([[float(x) for x in line.split()] for line in file.readline().split(': ')[1].split('\n')])
        dist = np.array([float(x) for x in file.readline().split(': ')[1].split()])

def save_image():
    global calibrated_frame, binarized_frame
    if background_frame is not None and scan_frame is not None and mtx is not None and dist is not None:
        if difference_frame is None:
            calculate_difference(background_frame, scan_frame)
        if sum_frame is None:
            color_sum_to_bw(difference_frame)
        if binarized_frame is None:
            binarized_frame = binarize_image(sum_frame, binarize_threshold)
        if calibrated_frame is None:
            calibrated_frame = undistort(binarized_frame)

        # Define the figure size in inches
        calc_dpi()
        fig_width = calibrated_frame.shape[1] / dpi
        fig_height = calibrated_frame.shape[0] / dpi

        # Save the image to a pdf, scaled, and black on white
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.axis('off')  # Hide axes
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.imshow(255 - calibrated_frame, cmap='gray', aspect='auto')
        plt.savefig('image.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
        plt.close(fig)

def create_gui():
    root = Tk()
    root.title("Webcam Frame Capture")

    # Menu setup
    menubar = Menu(root)
    
    # File menu
    file_menu = Menu(menubar, tearoff=0)
    file_menu.add_command(label="Save Parameters", command=save_parameters)
    file_menu.add_command(label="Load Parameters", command=load_parameters)
    file_menu.add_command(label="Save Image", command=save_image)
    menubar.add_cascade(label="File", menu=file_menu)
    
    # Settings menu
    settings_menu = Menu(menubar, tearoff=0)
    settings_menu.add_command(label="Camera Setup", command=camera_setup)
    settings_menu.add_command(label="Pause for Debug", command=pause_for_debug)  # Placeholder for debug breakpoint
    settings_menu.add_command(label="Calibration", command=lambda: calibration(root))  # Opens a new window for calibration
    menubar.add_cascade(label="Settings", menu=settings_menu)
    
    root.config(menu=menubar)
    tab_control = ttk.Notebook(root)

    # Tab 1: Background and Scan
    tab1 = ttk.Frame(tab_control)
    tab_control.add(tab1, text='Background and Scan')
    background = Frame(tab1)
    background.grid(row=0, column=0, padx=10, pady=10)
    label_background = Label(background)
    label_background.pack()
    btn_background = Button(background, text="Capture Background Frame", command=lambda: update_image(label_background, capture_and_average_frames, 'background_frame', .5))
    btn_background.pack()
    btn_hist_background = Button(background, text="Show Histogram", command=lambda: display_histogram(background_frame))
    btn_hist_background.pack()

    scan = Frame(tab1)
    scan.grid(row=0, column=1, padx=10, pady=10)
    label_scan = Label(scan)
    label_scan.pack()
    btn_scan = Button(scan, text="Capture Scan Frame", command=lambda: update_image(label_scan, lambda: capture_and_average_frames(), 'scan_frame', .5))
    btn_scan.pack()
    btn_hist_scan = Button(scan, text="Show Histogram", command=lambda: display_histogram(scan_frame))
    btn_hist_scan.pack()

    # Tab 2: Difference and Sum to BW
    tab2 = ttk.Frame(tab_control)
    tab_control.add(tab2, text='Difference and Sum to BW')
    difference = Frame(tab2)
    difference.grid(row=0, column=0, padx=10, pady=10)
    label_difference = Label(difference)
    label_difference.pack()
    btn_difference = Button(difference, text="Show Difference", command=lambda: update_image(label_difference, lambda: calculate_difference(background_frame, scan_frame), 'difference_frame', .5))
    btn_difference.pack()
    btn_hist_difference = Button(difference, text="Show Histogram", command=lambda: display_histogram(difference_frame))
    btn_hist_difference.pack()

    sum_bw = Frame(tab2)
    sum_bw.grid(row=0, column=1, padx=10, pady=10)
    label_sum_bw = Label(sum_bw)
    label_sum_bw.pack()
    btn_sum_bw = Button(sum_bw, text="Show Sum to BW", command=lambda: update_image(label_sum_bw, lambda: color_sum_to_bw(difference_frame), 'sum_frame', .5))
    btn_sum_bw.pack()
    btn_hist_sum_bw = Button(sum_bw, text="Show Histogram", command=lambda: display_histogram(sum_frame))
    btn_hist_sum_bw.pack()

    # Tab 3: Binarized Image
    tab3 = ttk.Frame(tab_control)
    tab_control.add(tab3, text='Binarized Image')
    binarized = Frame(tab3)
    binarized.pack(fill=BOTH, expand=True)

    # Create a frame for the histogram and binarized image display
    frame_histogram = Frame(binarized)
    frame_histogram.pack(side=LEFT, fill=BOTH, expand=True)

    frame_binarized = Frame(binarized)
    frame_binarized.pack(side=RIGHT, fill=BOTH, expand=True)

    # Label for displaying the binarized image
    label_binarized = Label(frame_binarized)
    label_binarized.pack(fill=BOTH, expand=True)

    # Checkbox for filtering option
    filter_var = BooleanVar()
    filter_checkbox = Checkbutton(frame_binarized, text="Filter first", variable=filter_var, command=lambda: update_binarized_image(binarize_threshold))
    filter_checkbox.pack()

    # Checkbox for gap filling option
    gap_fill_var = BooleanVar()
    gap_fill_checkbox = Checkbutton(frame_binarized, text="Gap Filling", variable=gap_fill_var, command=lambda: update_binarized_image(binarize_threshold))
    gap_fill_checkbox.pack()

    # Create the histogram plot
    fig = Figure(figsize=(6, 4), dpi=100)
    plot = fig.add_subplot(111)
    hist = np.zeros((256, 1))

    # Create a canvas to embed the histogram plot in the GUI
    canvas = FigureCanvasTkAgg(fig, master=frame_histogram)
    canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
    plot.plot(hist, color='gray')
    plot.axvline(x=binarize_threshold, color='r')
    plot.set_title('Click to set threshold')
    plot.set_xlim([0, 256])
    canvas.draw()

    # Tab 4: Calibration
    tab4 = ttk.Frame(tab_control)
    tab_control.add(tab4, text='Calibration')
    calibration_frame = Frame(tab4)
    calibration_frame.pack(fill=BOTH, expand=True)

    # Label for displaying the binarized image
    label_binarized_calibration = Label(calibration_frame)
    label_binarized_calibration.pack(side=LEFT, fill=BOTH, expand=True)

    # Label for displaying the undistorted binarized image
    label_undistorted_binarized = Label(calibration_frame)
    label_undistorted_binarized.pack(side=RIGHT, fill=BOTH, expand=True)

    # Function to update the binarized image based on the selected threshold
    def update_binarized_image(threshold):
        global binarized_frame
        if filter_var.get():
            frame_filtered = cv2.GaussianBlur(sum_frame, (3, 3), 0)
            update_histogram(frame_filtered)
        else:
            frame_filtered = sum_frame
            update_histogram(sum_frame)

        binarized_frame = binarize_image(frame_filtered, threshold)

        if gap_fill_var.get():
            # Apply gap filling if checkbox is selected
            kernel = np.ones((3,3), np.uint8)
            binarized_frame = cv2.morphologyEx(binarized_frame, cv2.MORPH_CLOSE, kernel)

        img = Image.fromarray(binarized_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label_binarized.imgtk = imgtk
        label_binarized.configure(image=imgtk)

    # Update images if necessary
    def update_difference_and_sum():
        if background_frame is not None and scan_frame is not None:
            update_image(label_difference, lambda: calculate_difference(background_frame, scan_frame), 'difference_frame', .5)
            update_image(label_sum_bw, lambda: color_sum_to_bw(difference_frame), 'sum_frame', .5)

    def update_histogram(image):
        global hist
        if image is not None:
            hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            plot.clear()
            plot.plot(hist, color='gray')
            plot.axvline(x=binarize_threshold, color='r')
            plot.set_title('Click to set threshold')
            plot.set_xlim([0, 256])
            canvas.draw()

    def update_calibration():
        label_binarized_calibration.imgtk = label_binarized.imgtk
        label_binarized_calibration.configure(image=label_binarized.imgtk)
        calibrated_frame = undistort(binarized_frame)
        img_undistorted = Image.fromarray(calibrated_frame)
        imgtk_undistorted = ImageTk.PhotoImage(image=img_undistorted)
        label_undistorted_binarized.imgtk = imgtk_undistorted
        label_undistorted_binarized.configure(image=imgtk_undistorted)

    # Event handler for mouse clicks on the histogram
    def on_click(event):
        global binarize_threshold
        if event.xdata:  # Check if click is on the plot
            binarize_threshold = int(event.xdata)
            update_binarized_image(binarize_threshold)
            
    # Update things when tab changes
    def on_tab_selected(event):
        selected_tab = event.widget.select()
        tab_text = event.widget.tab(selected_tab, "text")
        if tab_text == "Difference and Sum to BW" and background_frame is not None and scan_frame is not None:
            update_difference_and_sum()
        elif tab_text == "Binarized Image" and sum_frame is not None:
            update_binarized_image(binarize_threshold)
        elif tab_text == "Calibration" and binarized_frame is not None:
            update_calibration()

    tab_control.bind("<<NotebookTabChanged>>", on_tab_selected)

    canvas.mpl_connect('button_press_event', on_click)

    tab_control.pack(expand=1, fill='both')
    root.mainloop()

def camera_setup():
    """
    Open a new window to display the live webcam feed with exposure settings.
    """
    global selected_exposure_value
    global selected_camera_index
    window = Toplevel()
    window.title("Live Webcam Feed")

    # Detect available cameras
    camera_indices = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            camera_indices.append(index)
        cap.release()
        index += 1

    # Default camera setup
    cap = cv2.VideoCapture(selected_camera_index)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    # Mapping of CAP_PROP_EXPOSURE values to actual exposure times
    exposure_values = {
        0: "1s",
        -1: "500ms",
        -2: "250ms",
        -3: "125ms",
        -4: "62.5ms",
        -5: "31.3ms",
        -6: "15.6ms",
        -7: "7.8ms",
        -8: "3.9ms",
        -9: "2ms",
        -10: "976.6µs",
        -11: "488.3µs",
        -12: "244.1µs",
        -13: "122.1µs"
    }

    def update_exposure(new_exposure):
        global selected_exposure_value
        cap.set(cv2.CAP_PROP_EXPOSURE, int(new_exposure))
        selected_exposure_value = int(new_exposure)

    def update_camera(event):
        global selected_camera_index
        nonlocal cap
        new_index = int(camera_dropdown.get())
        if new_index != selected_camera_index:
            cap.release()
            selected_camera_index = new_index
            cap = cv2.VideoCapture(selected_camera_index)
            cap.set(cv2.CAP_PROP_AUTO_WB, 0)
            cap.set(cv2.CAP_PROP_EXPOSURE, selected_exposure_value)

    def show_frame():
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        lbl_video.imgtk = imgtk
        lbl_video.configure(image=imgtk)
        lbl_video.after(10, show_frame)

    lbl_video = Label(window)
    lbl_video.pack()

    exposure_label = Label(window, text="Select Exposure:")
    exposure_label.pack(side=LEFT)
    exposure_dropdown = ttk.Combobox(window, values=list(exposure_values.values()), state="readonly")
    exposure_dropdown.pack(side=LEFT)
    exposure_dropdown.bind("<<ComboboxSelected>>", lambda event: update_exposure(list(exposure_values.keys())[list(exposure_values.values()).index(exposure_dropdown.get())]))

    camera_label = Label(window, text="Select Camera:")
    camera_label.pack(side=LEFT)
    camera_dropdown = ttk.Combobox(window, values=camera_indices, state="readonly")
    camera_dropdown.pack(side=LEFT)
    camera_dropdown.bind("<<ComboboxSelected>>", update_camera)

    show_frame()

def grid_calibration(frame):
    global mtx, dist, gridy, gridx
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, (gridy, gridx), None)
    if ret_corners:
        # Refining corner locations for higher accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Define the 3D coordinates of chessboard corners in mm (z always 0)
        objp = np.zeros((gridy*gridx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:gridy*grid_mm:grid_mm, 0:gridx*grid_mm:grid_mm].T.reshape(-1, 2)

        # Arrays to store object points and image points from all images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        objpoints.append(objp)
        imgpoints.append(corners)

        # Calibrate the camera
        ret_calibrate, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        if not ret_calibrate:
            return False, "Calibration failed."
    else:
        return False, "Chessboard corners not found."
    return True, "Calibration successful.", corners, mtx, dist

def calibration(root):
    calibration_window = Toplevel(root)
    calibration_window.title("Calibration")
    lbl_calibration = Label(calibration_window)
    lbl_calibration.pack()

    # Default camera setup
    cap = cv2.VideoCapture(selected_camera_index)
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)

    def show_calibration_frame():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_calibration.imgtk = imgtk
            lbl_calibration.configure(image=imgtk)
        lbl_calibration.after(10, show_calibration_frame)
    show_calibration_frame()
    
    def capture_and_calibrate():
        global grid_frame, gridx, gridy, grid_mm, mtx, dist
        lbl_error.config(text="")  # Clear previous errors

        try:
            gridx = int(entry_horizontal.get())
            gridy = int(entry_vertical.get())
            grid_mm = int(entry_grid_size.get())
        except ValueError:
            lbl_error.config(text="Please enter valid integers for vertices.")
            return

        cap.release()
        grid_frame = capture_and_average_frames()
        try:
            success, message, corners, mtx, dist = grid_calibration(grid_frame)
        except Exception as e:
            lbl_error.config(text=str(e))
            return
        if success:
            grid_frame_draw = cv2.cvtColor(grid_frame, cv2.COLOR_RGB2BGR)
            cv2.drawChessboardCorners(grid_frame_draw, (gridx, gridy), corners, success)
            grid_frame_draw = cv2.cvtColor(grid_frame_draw, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(grid_frame_draw)
            imgtk = ImageTk.PhotoImage(image=img)
            lbl_calibration.imgtk = imgtk
            lbl_calibration.configure(image=lbl_calibration.imgtk)
            lbl_calibration.update()
        lbl_error.config(text=message)
    
    frame_vertical = Frame(calibration_window)
    frame_vertical.pack()
    lbl_vertical = Label(frame_vertical, text="Horiz Vertices")
    lbl_vertical.pack(side=LEFT)
    entry_vertical = Entry(frame_vertical)
    entry_vertical.pack(side=LEFT)
    entry_vertical.insert(0, str(gridx))

    frame_horizontal = Frame(calibration_window)
    frame_horizontal.pack()
    lbl_horizontal = Label(frame_horizontal, text="Vertical Vertices")
    lbl_horizontal.pack(side=LEFT)
    entry_horizontal = Entry(frame_horizontal)
    entry_horizontal.pack(side=LEFT)
    entry_horizontal.insert(0, str(gridy))

    frame_grid_size = Frame(calibration_window)
    frame_grid_size.pack()
    lbl_grid_size = Label(frame_grid_size, text="Grid size (mm)")
    lbl_grid_size.pack(side=LEFT)
    entry_grid_size = Entry(frame_grid_size)
    entry_grid_size.pack(side=LEFT)
    entry_grid_size.insert(0, str(grid_mm))

    btn_calibrate = Button(calibration_window, text="Calibrate", command=capture_and_calibrate)
    btn_calibrate.pack()

    lbl_error = Label(calibration_window, text="", fg="red")
    lbl_error.pack(side=LEFT, anchor=CENTER)

def undistort(frame):
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    # undistort
    undist = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    return undist[y:y+h, x:x+w]

def calc_dpi():
    global dpi

    undist_grid = undistort(grid_frame)
    gray = cv2.cvtColor(undist_grid, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (gridy, gridx), None)
    if ret:
        # Refining corner locations for higher accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        # Calculate the distances between each consecutive corner
        distances = []
        for i in range(len(corners) - 1):
            # Calculate Euclidean distance between consecutive corners
            dist = np.linalg.norm(corners[i] - corners[i+1])
            distances.append(dist)

        # The distances include some large values when moving between row/columns.  Remove the large ones before averaging
        small_distances = [d for d in distances if d < np.average(distances)]
        avg_square_pixels = np.average(small_distances)
        print("Average pixels per calibration square:", avg_square_pixels)

        dpi = avg_square_pixels * 25.4 / grid_mm
        print("DPI:", dpi)


create_gui()
