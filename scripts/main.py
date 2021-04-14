import numpy as np
from PIL import Image, ImageTk, ImageOps
import matplotlib.pyplot as plt
import cv2
from scipy.integrate import simps
import os
import tkinter
from scipy.signal import find_peaks
import math
from tkinter import messagebox
from matplotlib.ticker import (AutoMinorLocator)
from tkinter import Radiobutton, Frame, Button, filedialog, Scale, Canvas, PhotoImage, Label, Scale, Entry, StringVar
from shutil import rmtree
import xlsxwriter

#make GUI
root = tkinter.Tk()
root.title("Intensity Grapher")
smooth_val = 0
h_shift_val = 0
v_shift_val = 0

#ratio is 3:2
plot_disp_size = (int(430*1.5), 430)

# text_entry_arr = []
text_box_arr = []

bounds = []

#creates resource folder in path
if 'temp_resources' not in os.listdir('../'):
	os.mkdir('../temp_resources')

if 'cropped' not in os.listdir('../temp_resources'):
	os.mkdir('../temp_resources/cropped')

#for exiting the program
def on_closing():
	if messagebox.askokcancel("Quit", "Are you sure you want to quit (unsaved data will be discarded)?"):
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

#image processing
def thresh_and_crop():
	global init_vals
	init_vals = []

	try:
		img_path = root.filename
	except:
		print("Root Filename not compatible with image path")
		return

	#thresholding
	img = cv2.imread(img_path)
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	_, img_thresh = cv2.threshold(img_gray, 255*(float(thresh_val)/100), 255, cv2.THRESH_TOZERO)

	#cropping
	cnt, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 
	cnt_sort = sorted(cnt, key=cv2.contourArea)

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
		global img_path
		img_path = '../temp_resources/cropped/' + os.path.split(root.filename)[1]
	except:
		print("Image path not defined")
		return
	
	if os.path.exists(img_path) == False:
		print("Must threshold image first")
		return

	img_raw = cv2.imread(img_path)
	
	#select ROI function 1 (top strip)
	roi = cv2.selectROI(img_raw)
	roi_cropped1 = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

	#select ROI function 2 (bottom strip)
	roi = cv2.selectROI(img_raw)
	roi_cropped2 = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] 

	try:
		cv2.imwrite("../temp_resources/topstrip.jpeg", roi_cropped1)
		cv2.imwrite('../temp_resources/bottomstrip.jpeg', roi_cropped2)
	except:
		print("No ROI selected")

	cv2.destroyAllWindows()

#smoothing filter slider
def update_smooth(val):
	global smooth_val
	smooth_val = val
	make_graph()
	os.remove('../temp_resources/temp.png')

#curve smoothing
def smooth(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, mode='valid')

#updates after baseline selection
def update_baseline():
	preview_button["state"] = "normal"

	global baseline_grabbed
	baseline_grabbed = baseline_choice.get()

#updates after selecting number of peak bounds
def update_peaks():
	bounds_button['state'] = 'normal'

	global peaks_num_grabbed
	peaks_num_grabbed = peak_num_choice.get()

#choosing peak bounds for integration step 
def choose_peak_bounds():
	export_button["state"] = "normal"

	global bounds

	make_graph(bounds = True)

	return bounds

def update_h_shift(val):
	global h_shift_val
	h_shift_val = val
	make_graph()
	os.remove('../temp_resources/temp.png')

def update_v_shift(val):
	global v_shift_val
	v_shift_val = val
	make_graph()
	os.remove('../temp_resources/temp.png')

def preview_graph():
	make_graph()
	os.remove('../temp_resources/temp.png')	
	export_button['state'] = 'normal'
	curve_smoothing_slider['state'] = 'normal'
	horizontal_shift_slider['state'] = 'normal'
	vertical_shift_slider['state'] = 'normal'

#previews graph
def make_graph(bounds = False):
	global vals
	vals = []
	
	'''
	UNCOMMENT LATER?
	folder_selected = filedialog.askdirectory(title='Choose Location to Save Data')
	'''

	#in case matplotlib crashes
	plt.clf()
		
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

	#initial vals
	if len(init_vals) == 0:
		t1 = np.arange(len(x1))
		t2 = np.arange(len(x2))
		init_vals.extend([t1, x1, t2, x2])

	#smoothing
	if int(smooth_val) > 0:
		x1 = smooth(x1, int(len(x1)/3 * (float(float(smooth_val))/100.000)))
		x2 = smooth(x2, int(len(x2)/3 * (float(float(smooth_val))/100.000)))
		
		x1 = x1[1:(len(x1) - 1)]
		x2 = x2[1:(len(x2) - 1)]

	#baseline adjustment
	if baseline_grabbed == 101: #midpoint
		x1_mid = x1[int(len(x1)/2)]
		x2_mid = x2[int(len(x2)/2)]

		x1 = [i - x1_mid for i in x1]
		x2 = [i - x2_mid for i in x2]

	#low val (shifts all to y=0 for standard axis)
	low_val = min(list(np.append(x1, x2)))

	x1 = [i-low_val for i in x1]
	x2 = [i-low_val for i in x2]
	
	#converts values to percentages of max intensity to nearest hundredth (to make uniform across pictures)
	highest_intensity = max(list(np.append(x1, x2)))

	for i in range(len(x1)):
		x1[i] = round((float(x1[i]) / float(highest_intensity)) * 100.00000, 2)
	for i in range(len(x2)):
		x2[i] = round((float(x2[i]) / float(highest_intensity)) * 100.00000, 2)

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

	t1 = np.arange(len(x1))
	t2 = np.arange(len(x2))

	if x1_peak < x2_peak:
		t1 = [i+x2_peak-x1_peak for i in t1]
	
	if x2_peak < x1_peak:
		t2 = [i+x1_peak-x2_peak for i in t2]

	#manual h and v shift 
	t1 = [i+int(h_shift_val)*0.75 for i in t1]
	x1 = [i+int(v_shift_val)*0.75 for i in x1]

	'''
	if h_shift_val == 0:
		horizontal_shift_slider['from'] = t1[-1]/2 * -1
		horizontal_shift_slider['to'] = t1[-1]/2
	if v_shift_val == 0:
		vertical_shift_slider['from'] = highest_intensity/2 * -1
		vertical_shift_slider['to'] = highest_intensity/2
	'''

	#NO THIS IS THE FISHY PART TOO
	if bounds == True:
		plt.clf()
		plt.title("Select LEFT and RIGHT BOUNDS of CONTROL PEAK (right)")
		plt.plot(t1, x1)
		plt.plot(t2, x2)
		clicked = plt.ginput(2)
		plt.close()
		control_peak = [math.floor(float(str(clicked).split(', ')[0][2:])), math.ceil(float(str(clicked).split(', ')[2][1:]))]
		left_point = min(range(len(t1)), key=lambda i: abs(t1[i]-control_peak[0]))
		right_point = min(range(len(t1)), key=lambda i: abs(t1[i]-control_peak[1]))
		points_right_peak = [left_point, right_point]
		plt.clf()

		print('right {} {} {}'.format(clicked, control_peak, points_right_peak))

		if peaks_num_grabbed == 102:
			plt.clf()
			plt.title("Select LEFT and RIGHT BOUNDS of TEST PEAK (left)")
			plt.plot(t1, x1)
			plt.plot(t2, x2)
			clicked = plt.ginput(2)
			plt.close()
			test_peak = [math.floor(float(str(clicked).split(', ')[0][2:])), math.ceil(float(str(clicked).split(', ')[2][1:]))]
			left_point = min(range(len(t1)), key=lambda i: abs(t1[i]-test_peak[0]))
			right_point = min(range(len(t1)), key=lambda i: abs(t1[i]-test_peak[1]))
			points_left_peak = [left_point, right_point]
			plt.clf()
			
			print('left {} {} {}'.format(clicked, test_peak, points_left_peak))


	#matplot plotting
	hfont = {'fontname': 'Arial', 'weight': 'bold', 'size': 45}
	ax = plt.subplot(111)

	plt.plot(t1, x1, linewidth=2)
	plt.plot(t2, x2, linewidth=2)
	ax.tick_params(width=1)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax.yaxis.set_minor_locator(AutoMinorLocator(2))
	plt.setp(ax.spines.values(), linewidth=1.5)
	ax.tick_params(which='minor', width=1, length=5, labelsize=14)
	ax.tick_params(which='major', width=1.5, length=15, labelsize=32)

	plt.ylabel('Intensity (a.u.)', **hfont)
	plt.xlabel('Pixel distance', **hfont)

	plt.setp(ax.get_yticklabels(), fontweight="bold", fontname="Arial")
	plt.setp(ax.get_xticklabels(), fontweight="bold", fontname="Arial")

	vals.extend([t1, x1, t2, x2])

	plt.legend(['Top Line', 'Bottom Line'], frameon=False, prop={'family': 'Arial', 'weight': 'bold', 'size': 32})

	#resizing
	figure = plt.gcf()
	figure.set_size_inches(15, 10)

	#fill + area + peak
	#bounds kinda work??
	if bounds == True:
		plt.fill_between(t1, x1, 0, where = (t1 > (t1[points_right_peak[0]]+t2[0]-t1[0] if t2[0] > t1[0] else t1[0]-t2[0])) & (t1 <= (t1[points_right_peak[1]]+t2[0]-t1[0] if t2[0] > t1[0] else t1[0]-t2[0])), color = (1, 0, 0, 0.2))
		plt.fill_between(t2, x2, 0, where = (t2 > t2[points_right_peak[0]]) & (t2 <= t2[points_right_peak[1]]), color = (0, 0, 1, 0.2))

		vals.extend([simps(x1[points_right_peak[0]:points_right_peak[1]], t1[points_right_peak[0]:points_right_peak[1]], dx=0.01)])
		vals.extend([simps(x2[points_right_peak[0]:points_right_peak[1]], t2[points_right_peak[0]:points_right_peak[1]], dx=0.01)])
		vals.extend([max(x1[points_right_peak[0]:points_right_peak[1]])])
		vals.extend([max(x2[points_right_peak[0]:points_right_peak[1]])])
		
		if peaks_num_grabbed == 102:
			plt.fill_between(t1, x1, 0, where = (t1 > (t1[points_left_peak[0]]+t2[0]-t1[0] if t2[0] > t1[0] else t1[0]-t2[0])) & (t1 <= (t1[points_left_peak[1]]+t2[0]-t1[0] if t2[0] > t1[0] else t1[0]-t2[0])), color = (1, 0, 0, 0.2))
			plt.fill_between(t2, x2, 0, where = (t2 > t2[points_left_peak[0]]) & (t2 <= t2[points_left_peak[1]]), color = (0, 0, 1, 0.2))
			
			vals.extend([simps(x1[points_left_peak[0]:points_left_peak[1]], t1[points_left_peak[0]:points_left_peak[1]], dx=0.01)])
			vals.extend([simps(x2[points_left_peak[0]:points_left_peak[1]], t2[points_left_peak[0]:points_left_peak[1]], dx=0.01)])
			vals.extend([max(x1[points_left_peak[0]:points_left_peak[1]])])
			vals.extend([max(x2[points_left_peak[0]:points_left_peak[1]])])

	global im
	plt.savefig('../temp_resources/temp.png', bbox_inches='tight')
	im = ImageTk.PhotoImage(Image.open('../temp_resources/temp.png').resize(plot_disp_size))
	image_canvas.itemconfigure(imload, image=im)

#saves graph
#except needs more data to save 
def save_graph():
	f = filedialog.asksaveasfilename(defaultextension='.xlsx')
	if f:
		plt.savefig(f.split('.xlsx')[0] + '.png', bbox_inches='tight')
		workbook = xlsxwriter.Workbook(f)
		worksheet = workbook.add_worksheet()
		# Add a bold format to use to highlight cells.
		bold = workbook.add_format({'bold': True})
		#vals = [t1, x1, t2, x2, area_peak_left_x1, area_padfax2, prekarightx1, same x2]
		worksheet.write('A1', 'Top Strip X-values (initial)', bold)
		worksheet.write('B1', 'Top Strip Y-values (initial)', bold)
		worksheet.write('C1', 'Bottom Strip X-values (initial)', bold)
		worksheet.write('D1', 'Bottom Strip Y-Values (initial)', bold)
		worksheet.write('E1', 'Top Strip X-values (adjusted)', bold)
		worksheet.write('F1', 'Top Strip Y-values (adjusted)', bold)
		worksheet.write('G1', 'Bottom Strip X-values (adjusted)', bold)
		worksheet.write('H1', 'Bottom Strip Y-Values (adjusted)', bold)
		worksheet.write('I1', 'Area of control (right) peak - Top Line', bold)
		worksheet.write('J1', 'Area of control (right) peak - Bottom Line', bold)
		worksheet.write('K1', 'Area of test (left) peak - Top Line', bold)
		worksheet.write('L1', 'Area of test (left) peak - Bottom Line', bold)
		worksheet.write('I3', 'Max of control (right) peak - Top Line', bold)
		worksheet.write('J3', 'Max of control (right) peak - Bottom Line', bold)
		worksheet.write('K3', 'Max of test (left) peak - Top Line', bold)
		worksheet.write('L3', 'Max of test (left) peak - Bottom Line', bold)

		worksheet.set_column('A:A', 22) #these are widths of columns in cm of excel, just to make it more readable
		worksheet.set_column('B:B', 22)
		worksheet.set_column('C:C', 25)
		worksheet.set_column('D:D', 25)
		worksheet.set_column('E:E', 25)
		worksheet.set_column('F:F', 25)
		worksheet.set_column('G:G', 28)
		worksheet.set_column('H:H', 28)
		worksheet.set_column('I:I', 32)
		worksheet.set_column('J:J', 36)
		worksheet.set_column('K:K', 30)
		worksheet.set_column('L:L', 34)

		for i in range(len(init_vals[0])):
			worksheet.write('A'+str(i+2), init_vals[0][i])
			worksheet.write('B'+str(i+2), init_vals[1][i])

		for i in range(len(init_vals[2])):
			worksheet.write('C'+str(i+2), init_vals[2][i])
			worksheet.write('D'+str(i+2), init_vals[3][i])

		for i in range(len(vals[0])):
			worksheet.write('E'+str(i+2), vals[0][i])
			worksheet.write('F'+str(i+2), vals[1][i])

		for i in range(len(vals[2])):
			worksheet.write('G'+str(i+2), vals[2][i])
			worksheet.write('H'+str(i+2), vals[3][i])

		if len(vals) >= 6:
			worksheet.write('I2', vals[4])
			worksheet.write('J2', vals[5])
			worksheet.write('I4', vals[6])
			worksheet.write('J4', vals[7])

		if len(vals) >= 10:
			worksheet.write('K2', vals[8])
			worksheet.write('L2', vals[9])
			worksheet.write('K4', vals[10])
			worksheet.write('L4', vals[11])

		# Insert an image.
		worksheet.insert_image('J6', f.split('.xlsx')[0] + '.png', {'x_scale': 0.40, 'y_scale': 0.40})	
		worksheet.insert_image('J25', img_path, {'x_scale': 0.40, 'y_scale': 0.40})	

		workbook.close()

	elif f is None:
		return

#makes sure things inputted into the v_shift and h_shift text areas are strictly numbers of 8 characters or less (i.e. -5.2, 5, 195.925)
def character_limit(p):
	if len(p.get()) > 6 or is_number(p.get()) == False:
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

#initializes tkinter GUI
def init():
	#setting variables to global scope that need to be accessed outside of init()
	global curve_smoothing_slider, horizontal_shift_slider, vertical_shift_slider, image_canvas, bounds_button, preview_button, export_button, baseline_choice, im, imload, peak_num_choice

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

	Button(left_frame, text="Select a file", command=select_file).pack(anchor= 'n',pady=(0,10))

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
	modes = [("Midpoint", 101), ("Lowest Value", 102)]
	Label(left_frame, text="Baseline from:", justify="left", padx=20).pack()
	i=0
	for mode, val in modes:
		Radiobutton(left_frame, text=mode, indicatoron=1, command=update_baseline, justify="left", padx=20,  variable=baseline_choice, value=val).pack(anchor='w')
		i+=1

	peak_num_choice = tkinter.IntVar()
	peak_num_choice.set(1)
	modes = [("One Peak", 101), ("Two Peaks", 102)]
	Label(left_frame, text="How many peaks to compare:", justify="left", padx=20).pack(pady=(20,0))
	i=0
	for mode, val in modes:
		Radiobutton(left_frame, text=mode, indicatoron=1, command=update_peaks, justify="left", padx=20,  variable=peak_num_choice, value=val).pack(anchor='w')
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

	Label(sub_middle_frame, text="Horizontal Shift").grid(column=0, row=1, padx=(0,20))
	horizontal_shift_slider = Scale(sub_middle_frame, orient="horizontal", length=300, from_=-10.0, to=10.0, command=update_h_shift)
	horizontal_shift_slider.grid(column=0, row=0, padx=(0,20))
	horizontal_shift_slider['state'] = 'disable'
	'''
	h_shift = StringVar()
	horizontal_shift_box = Entry(sub_middle_frame, textvariable=h_shift, width=8)
	horizontal_shift_box.grid(column=0, row=2, padx=(0,20), pady=(0,5))
	h_shift.trace('w', lambda *args:character_limit(h_shift))
	'''

	Label(sub_middle_frame, text="Vertical Shift").grid(column=1, row=1)
	vertical_shift_slider = Scale(sub_middle_frame, orient="horizontal", length=300, from_=-10.0, to=10.0, command=update_v_shift)
	vertical_shift_slider.grid(column=1, row=0)
	vertical_shift_slider['state'] = 'disable'
	'''
	v_shift = StringVar()
	vertical_shift_box = Entry(sub_middle_frame, textvariable=v_shift, width=8)
	vertical_shift_box.grid(column=1, row=2, pady=(0,5))
	v_shift.trace('w', lambda *args:character_limit(v_shift))
	'''

	#right side graph
	width, height = plot_disp_size
	image_canvas = Canvas(middle_frame, width=width, height=height)
	image_canvas.pack(padx=(20,0), pady=(0,0))

	im = ImageTk.PhotoImage(Image.new("RGB", plot_disp_size, (255, 255, 255)))  #PIL solution
	imload = image_canvas.create_image(0, 0, image=im, anchor='nw')

if __name__ == '__main__':
	init() #builds all the buttons and frames
	
	root.protocol("WM_DELETE_WINDOW", on_closing) #when the "x" is hit to close the window, tkinter needs to handle it in a special way
	root.mainloop() #starts the instance of tkinter (the GUI framework)
