#this needs some cleaning
import statsmodels.api as sm
import numpy as np
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt
import cv2
import time
from scipy.integrate import simps
import os
import tkinter
from scipy.signal import find_peaks
import csv
import tkinter.ttk as ttk
import math
from matplotlib.ticker import (AutoMinorLocator)
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
from tkinter import Radiobutton, Entry, Frame, Button, StringVar, filedialog, Scale, Canvas, PhotoImage, Label
import random
from shutil import rmtree

#make GUI
root = tkinter.Tk()
root.title("Intensity Grapher")
smooth_val = 0
h_shift_val = 0
v_shift_val = 0

#ratio is 3:2
plot_disp_size = (int(370*1.5), 370)

text_entry_arr = []
text_box_arr = []

bounds = []

#creates resource folder in path
if 'temp_resources' not in os.listdir('../'):
	os.mkdir('../temp_resources')

if 'cropped' not in os.listdir('../temp_resources'):
	os.mkdir('../temp_resources/cropped')

#for exiting the program
def on_closing():
	if tkinter.messagebox.askokcancel("Quit", "Are you sure you want to quit (unsaved data will be discarded)?"):
		print("Exited")
		root.quit()
		root.destroy()
		rmtree('../temp_resources')

#presents a help window with documentation on how to use our program, will make it read from the README.md file later
def help_window():
	window = tkinter.Toplevel(root)
	window.title("New Window") 
	window.geometry("200x200")
	text = "This is \na test"
	label = Label(window, text=text).pack(anchor='nw') 

#opens dialog to select image
def select_file():
	root.filename = filedialog.askopenfilename(initialdir="../", title="Select image file", filetypes=(("Image files (.jpg, .jpeg, .png)", "*.jpg *.jpeg *.png"), ("all files","*.*")))

#threshold slider
def update_thresh(val):
	global thresh_val
	thresh_val = val
	thresh_and_crop()
	print("Thresh: " + thresh_val)

#image processing
def thresh_and_crop():
	try:
		img_path = root.filename
	except:
		return

	#thresholding
	img = cv2.imread(img_path)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	_, img_thresh = cv2.threshold(img_gray, 255*(float(thresh_val)/100), 255, cv2.THRESH_TOZERO)

	#cropping
	cnt, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
	cnt_sort = sorted(cnt, key=cv2.contourArea)

	'''
	KEEP might use pointpolytest later
	print(cnt_sort, cv2.pointPolygonTest(cnt_sort[-1], (0, 0), False))
	'''

	cv2.drawContours(img_thresh, cnt_sort[:-2], -1, 0, -1)
	cnt_sort = cnt_sort[-2:]

	xmin = cnt_sort[-1][0][0][0]
	xmax = 0	
	ymin = cnt_sort[-1][0][0][1]
	ymax = 0

	#finding lowest x val and highest x val
	for i in range(len(cnt_sort)):
		for j in range(len(cnt_sort[i])):
			for z in range(len(cnt_sort[i][j])):
				f = cnt_sort[i][j]
				if f[z][0] < xmin:
					xmin = f[z][0]
				if f[z][0] > xmax:
					xmax = f[z][0]
				if f[z][1] < ymin:
					ymin = f[z][1]
				if f[z][1] > ymax:
					ymax = f[z][1]
	
	print((ymax, ymin, xmin, xmax))
	
	img_crop = img_thresh[ymin:ymax, xmin:xmax]

	#saves cropped image in cropped folder
	cv2.imwrite('../temp_resources/cropped/' + os.path.split(img_path)[1], img_crop)

	global im
	imtemp = Image.open('../temp_resources/cropped/' + os.path.split(img_path)[1]).resize(plot_disp_size)
	im = ImageTk.PhotoImage(imtemp)
	image_canvas.itemconfigure(imload, image=im)

#finding regions of interest
def find_roi():
	try:
		img_path = '../temp_resources/cropped/' + os.path.split(root.filename)[1]
	except:
		return
	
	img_raw = cv2.imread(img_path)
	
	#select ROI function 1 (top strip)
	roi = cv2.selectROI(img_raw)
	roi_cropped1 = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
	
	#select ROI function 2 (bottom strip)
	roi = cv2.selectROI(img_raw)
	roi_cropped2 = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] 
	
<<<<<<< HEAD
	try:
		cv2.imwrite("../resources/topstrip.jpeg", roi_cropped1)
		cv2.imwrite('../resources/bottomstrip.jpeg', roi_cropped2)
	except:
		print("No ROI selected")
=======
	cv2.imwrite("../temp_resources/topstrip.jpeg", roi_cropped1)
	cv2.imwrite('../temp_resources/bottomstrip.jpeg', roi_cropped2)
>>>>>>> i_hate_you

	cv2.destroyAllWindows()

#smoothing filter slider
def update_smooth(val):
	global smooth_val
	smooth_val = val
	print("Smooth: " + smooth_val)
	make_graph()
	os.remove('../temp_resources/temp.png')

#curve smoothing
def smooth(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	print("window {}".format(window))
	return np.convolve(interval, window, mode='valid')

#updates after baseline selection
def update_choice():
	preview_button["state"] = "normal"

	global baseline_grabbed
	baseline_grabbed = baseline_choice.get()

	try:
		_ = root.filename
	except:
		print("No file chosen")

#choosing peak bounds for integration step 
#NEEDS TO ADD AREA and do all the calculation here so it can iterate regardless of the curve bc some only have one peak
def choose_peak_bounds():
	export_button["state"] = "normal"

	global bounds

	return bounds

def update_h_shift(val):
	global h_shift_val
	h_shift_val = val
	print("Horizontal Shift: " + h_shift_val)
	make_graph()
	os.remove('../temp_resources/temp.png')

def update_v_shift(val):
	global v_shift_val
	v_shift_val = val
	print("Vertical Shift: " + v_shift_val)
	make_graph()
	os.remove('../temp_resources/temp.png')

def preview_graph():
	curve_smoothing_slider['state'] = 'normal'
	horizontal_shift_slider['state'] = 'normal'
	vertical_shift_slider['state'] = 'normal'
	make_graph()
	os.remove('../temp_resources/temp.png')	
	bounds_button['state'] = 'normal'
	curve_smoothing_slider['state'] = 'normal'
	horizontal_shift_slider['state'] = 'normal'
	vertical_shift_slider['state'] = 'normal'

#previews graph
def make_graph():
	
	'''
	UNCOMMENT LATER
	folder_selected = filedialog.askdirectory(title='Choose Location to Save Data')
	'''

	#in case matplotlib crashes
	try:
		plt.clf()
	except:
		print('You should be impressed you managed to get this error')
		
	top_line = Image.open('../temp_resources/topstrip.jpeg').convert("L")
	bottom_line = Image.open('../temp_resources/bottomstrip.jpeg').convert("L")

	#convert to numpy array
	np_top = np.array(top_line)
	top_line_array = []
	for elem in np_top:
		if elem.sum() != 0:
			top_line_array.append(elem)
			
	np_bottom = np.array(bottom_line)
	bottom_line_array = []
	for elem in np_bottom:
		if elem.sum() != 0:
			bottom_line_array.append(elem)

	x1 = [float(sum(l))/len(l) for l in zip(*top_line_array)]
	x2 = [float(sum(l))/len(l) for l in zip(*bottom_line_array)]

	#smoothing
	if int(smooth_val) > 0:
		print(smooth_val)
		print("init x1 {}".format(x1))
		print(float(len(x1)/3 * (float(float(smooth_val)/100.000))))

		x1 = smooth(x1, int(len(x1)/3 * (float(float(smooth_val))/100.000)))
		x2 = smooth(x2, int(len(x2)/3 * (float(float(smooth_val))/100.000)))

		print("smoothed x1 {}".format(x1))
		
		x1 = x1[1:(len(x1) - 1)]
		x2 = x2[1:(len(x2) - 1)]

	#baseline adjustment
	if baseline_grabbed == 101: #midpoint
		x1_mid = x1[int(len(x1)/2)]
		x2_mid = x2[int(len(x2)/2)]

		x1 = [i - x1_mid for i in x1]
		x2 = [i - x2_mid for i in x2]

		minimum = min(x1[np.argmin(np.array(x1))], x2[np.argmin(np.array(x2))])

		print(minimum, np.argmin(np.array(x1)))

		x1 = [i - minimum for i in x1]
		x2 = [i - minimum for i in x2]	
	
	if baseline_grabbed == 102: #lowest value
		x1_min = x1[np.argmin(np.array(x1))]
		x2_min = x2[np.argmin(np.array(x2))]
		
		x1 = [i - x1_min for i in x1]
		x2 = [i - x2_min for i in x2]
	
	#converts values to percentages of max intensity to nearest hundredth (to make uniform across pictures)
	highest_intensity = max(x1[np.argmax(np.array(x1))], x2[np.argmax(np.array(x2))])

	for i in range(len(x1)):
		x1[i] = round((float(x1[i]) / float(highest_intensity)) * 100.00000, 2)
	for i in range(len(x2)):
		x2[i] = round((float(x2[i]) / float(highest_intensity)) * 100.00000, 2)

	print("scaled intensity: {}".format(x1))

	#new auto peak detector for initial horizontal adjustment
	x1_peaks, _ = find_peaks(np.array(x1), height=15, distance=10, width=10)
	x2_peaks, _ = find_peaks(np.array(x2), height=15, distance=10, width=10)

	x1_peak = 0
	x2_peak = 0

	for i in x1_peaks:
		if x1[i] > x1[x1_peak]:
			x1_peak = i

	for i in x2_peaks:
		if x2[i] > x2[x2_peak]:
			x2_peak = i

	print("peak 1 index: {}".format(x1_peak))
	print("peak 2 index: {}".format(x2_peak))

	t1 = np.arange(len(x1))
	t2 = np.arange(len(x2))

	if x1_peak < x2_peak:
		t1 = [i+x2_peak-x1_peak for i in t1]
	
	if x2_peak < x1_peak:
		t2 = [i+x1_peak-x2_peak for i in t2]

	#manual h and v shift 
	t1 = [i+int(h_shift_val) for i in t1]
	x1 = [i+int(v_shift_val) for i in x1]

	'''
	plt.clf()
	plt.title("CLICK LEFT AND RIGHT OF THE RIGHTMOST PEAK (Bounds Selection)")
	plt.plot(x)

	clicked = plt.ginput(2)
	print(clicked)
	right_peak = [float(str(clicked).split(', ')[0]), float(str(clicked).split(', ')[1])]
	'''

	'''
	TESTPLOT AREA MODEL
	t = np.arange(0, 10, 0.01) 
	y = np.sin(t)+1
	plt.plot(t, y) 
	plt.title('matplotlib.pyplot.ginput() function Example', fontweight ="bold") 

	print("After 2 clicks :") 
	x = plt.ginput(2) 
	print(x) 
	
	plt.clf()

	endpoint1 = round(float(str(x[0]).split(', ')[0][1:]),2)
	endpoint2 = round(float(str(x[1]).split(', ')[0][1:]),2)

	print(endpoint1, endpoint2, t)

	t2 = np.arange(endpoint1, endpoint2, 0.01)

	print(t2)

	plt.close()

	y2 = np.sin(t2)+1
	plt.plot(t2, y2) 

	area = simps(y2, dx = 0.01)
	print(area)

	plt.show() 
	'''

	#matplot plotting
	hfont = {'fontname': 'Arial', 'weight': 'bold', 'size': 45}
	ax = plt.subplot(111)

	plt.plot(t1, x1)
	plt.plot(t2, x2)
	ax.tick_params(width=1)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax.yaxis.set_minor_locator(AutoMinorLocator(2))
	ax.legend()
	plt.setp(ax.spines.values(), linewidth=1.5)
	ax.tick_params(which='minor', width=1, length=5, labelsize=14)
	ax.tick_params(which='major', width=1.5, length=15, labelsize=32)

	plt.ylabel('Intensity (a.u.)', **hfont)
	plt.xlabel('Pixel distance', **hfont)

	plt.setp(ax.get_yticklabels(), fontweight="bold", fontname="Arial")
	plt.setp(ax.get_xticklabels(), fontweight="bold", fontname="Arial")

	plt.legend(['Top Line', 'Bottom Line'], frameon=False, prop={'family': 'Arial', 'weight': 'bold', 'size': 32})
	
	'''
	PEAK ANNOTATION
	for i in peaks_x:
		plt.annotate('Peak: {}'.format(x1[i]), xy = (i, x1[i]))
	for i in peaks_x2:
		plt.annotate('Peak: {}'.format(x2[i]), xy = (i, x2[i]))

	print(peaks_x, peaks_x2)
	'''

	#resizing
	figure = plt.gcf()
	figure.set_size_inches(15, 10)

	global im
	plt.savefig('../temp_resources/temp.png', bbox_inches='tight')
	im = ImageTk.PhotoImage(Image.open('../temp_resources/temp.png').resize(plot_disp_size))
	image_canvas.itemconfigure(imload, image=im)

#saves graph
#NEEDS TO ALSO EXPORT EXCEL DATA
def save_graph():
	f = filedialog.asksaveasfilename(defaultextension='.png')
	if f:
		plt.savefig(f, bbox_inches='tight')
	elif f is None:
		return


#makes sure things inputted into the v_shift and h_shift text areas are strictly numbers of 8 characters or less (i.e. -5.2, 5, 195.925)
def character_limit(p):
	if len(p.get()) > 8 or is_number(p.get()) == False:
		p.set(p.get()[:-1])

#checks if value is a number
def is_number(n):
	try:
		float(n)
		return True
	except:
		if n == "-":
			return True
		return False

'''
MIGHT USE LATER??
def make_ordinal(num):
	if num > 3:
		return str(num)+"th"
	elif num == 3:
		return "3rd"
	elif num == 2:
		return "2nd"
	else:
		return "1st"
'''

#initializes tkinter GUI
def init():
	#setting variables to global scope that need to be accessed outside of init()
	global curve_smoothing_slider, horizontal_shift_slider, vertical_shift_slider, image_canvas, bounds_button, preview_button, export_button, baseline_choice, im, imload

	left_frame = Frame(root)
	left_frame.pack(side="left")

	middle_frame = Frame(root)
	middle_frame.pack(side="right")

	right_frame = Frame(root)
	right_frame.pack(side="right")

	sub_middle_frame = Frame(middle_frame)
	sub_middle_frame.pack(side="bottom", pady=(0,10))

	#left side inputs
	Button(left_frame, text="Help", command=help_window).pack(anchor='nw', padx=(10,0),pady=(10,10))

	Button(left_frame, text="Select a file", command=select_file).pack(pady=(0,10))

	Label(left_frame, text="Threshold Slider", justify="center").pack(pady=(0,5))
	threshold_slider = Scale(left_frame, orient="horizontal", length=200, from_=1.0, to=50.0, command=update_thresh)
	threshold_slider.pack(padx=20, pady=(0,10))

	Button(left_frame, text="Select a ROI", command=find_roi).pack(pady=(0,15))

	Label(left_frame, text="Curve Smoothing", justify="center", padx=20).pack()
	curve_smoothing_slider = Scale(left_frame, orient="horizontal", length=200, from_=0.0, to=100.0, command=update_smooth)
	curve_smoothing_slider.pack(padx=20, pady=(0,20))
	curve_smoothing_slider['state'] = 'disable'

	baseline_choice = tkinter.IntVar()
	baseline_choice.set(1)
	modes = [("One Peak", 101), ("Two Peaks", 102)]
	Label(left_frame, text="Number of bands present on strip:", justify="left", padx=20).pack()
	i=0
	for mode, val in modes:
		Radiobutton(left_frame, text=mode, indicatoron=1, command=update_choice, justify="left", padx=20,  variable=baseline_choice, value=val).pack(anchor='w')
		i+=1

	#bottom row inputs
	bounds_button = Button(left_frame, text="Choose Bounds", command=choose_peak_bounds)
	bounds_button.pack(side="left", padx=(15,10), pady=(30,10))
	bounds_button["state"] = "disable"

	preview_button = Button(left_frame, text="Preview", command=preview_graph)
	preview_button.pack(side="left", padx=(10,10), pady=(30,10))
	preview_button["state"] = "disable"

	export_button = Button(left_frame, text="Export", command=save_graph)
	export_button.pack(side="left", padx=(10,0), pady=(30,10))
	export_button["state"] = "disable"

	'''
	MIGRATING TO A SLIDER
	Label(sub_middle_frame, text="Horizontal shift lines value (ex: -10.5): ").grid(column=0, row=0, pady=(10,0))
	h_shift = StringVar()
	h_shift_box = Entry(sub_middle_frame, textvariable=h_shift, width=8)
	h_shift_box.grid(column=1, row=0, pady=(10,0))
	h_shift.trace("w", lambda *args:character_limit(h_shift))

	Label(sub_middle_frame, text="Vertical shift lines value (ex: -10.5): ").grid(column=0, row=1, pady=(10,0))
	v_shift = StringVar()
	v_shift_box = Entry(sub_middle_frame, textvariable=v_shift, width=8)
	v_shift_box.grid(column=1, row=1, pady=(10,0))
	v_shift.trace("w", lambda *args:character_limit(v_shift))
	'''

	Label(sub_middle_frame, text="Horizontal Shift").grid(column=0, row=1, pady=(0,20))
	horizontal_shift_slider = Scale(sub_middle_frame, orient="horizontal", length=200, from_=-50.0, to=50.0, command=update_h_shift)
	horizontal_shift_slider.grid(column=0, row=0, padx=(0,20))
	horizontal_shift_slider['state'] = 'disable'

	Label(sub_middle_frame, text="Vertical Shift").grid(column=1, row=1, pady=(0,0))
	vertical_shift_slider = Scale(sub_middle_frame, orient="horizontal", length=200, from_=-50.0, to=50.0, command=update_v_shift)
	vertical_shift_slider.grid(column=1, row=0)
	vertical_shift_slider['state'] = 'disable'

	#vertical_shift_slider['length'] = 300

	#graph on right
	width, height = plot_disp_size
	image_canvas = Canvas(middle_frame, width=width, height=height)
	image_canvas.pack(padx=(20,0), pady=(0,0))

	im = ImageTk.PhotoImage(Image.new("RGB", plot_disp_size, (255, 255, 255)))  #PIL solution
	imload = image_canvas.create_image(0, 0, image=im, anchor='nw')

#__name__ is a preset python variable where if you're running this as the main file and not as some imported library, then __name__ is set to __main__
if __name__ == '__main__':
	init() #builds all the buttons and frames
	
	root.protocol("WM_DELETE_WINDOW", on_closing) #when the "x" is hit to close the window, tkinter needs to handle it in a special way
	root.mainloop() #starts the instance of tkinter (the GUI framework)
