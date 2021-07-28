import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import cv2
from scipy.integrate import simps
import os
import tkinter
from scipy.signal import find_peaks
import math
import re
from matplotlib.ticker import (AutoMinorLocator)
from tkinter import Text, Radiobutton, Frame, Button, filedialog, messagebox, Scale, Canvas, PhotoImage, Label, Scale, Entry, StringVar
from shutil import rmtree
import xlsxwriter

#make GUI
root = tkinter.Tk()
root.title("Intensity Grapher")
smooth_val = 0
h_shift_val = 0
v_shift_val = 0
bounds = []

#ratio is 3:2
plot_disp_size = (int(430*1.5), 430)

#character limits
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

		# h_shift = StringVar()
		# horizontal_shift_box = Entry(sub_middle_frame, textvariable=h_shift, width=8)
		# horizontal_shift_box.grid(column=0, row=2, padx=(0,20), pady=(0,5))
		# h_shift.trace('w', lambda *args:character_limit(h_shift))

#creates resource folder in the current directory
if 'temp_resources' not in os.listdir('./'):
	os.mkdir('./temp_resources')

if 'cropped' not in os.listdir('./temp_resources'):
	os.mkdir('./temp_resources/cropped')

#for exiting the program
def on_closing():
	if messagebox.askokcancel("Quit", "Are you sure you want to quit (unsaved data will be discarded)?"):
		print("[Exited]")
		root.quit()
		root.destroy()
		rmtree('./temp_resources')

#widget for creating help window
class CustomText(Text):
    def __init__(self, *args, **kwargs):
        Text.__init__(self, *args, **kwargs)

    def HighlightPattern(self, pattern, tag, start="1.0", end="end", regexp=True):

        start = self.index(start)
        end = self.index(end)
        self.mark_set("matchStart",start)
        self.mark_set("matchEnd",end)
        self.mark_set("searchLimit", end)

        count = tkinter.IntVar()
        while True:
            index = self.search(pattern, "matchEnd","searchLimit",count=count, regexp=regexp)
            if index == "": break
            self.mark_set("matchStart", index)
            self.mark_set("matchEnd", "%s+%sc" % (index,count.get()))
            self.tag_add(tag, "matchStart","matchEnd")

#presents a help window with documentation on how to use our program, will make it read from the README.md file later
def help_window():
	window = tkinter.Toplevel(root)
	window.title("Help") 
	window.geometry("800x600")
	f = open("DIRECTIONS.txt", 'r')
	text = f.readlines()
	f.close()
	t = CustomText(window, wrap="word", width=100, height=10, borderwidth=2)
	t.pack(sid="top", fill="both", expand=True)
	t.insert("1.0","".join(text))
	t.config(state='disable')
	t.tag_configure("blue", foreground="blue")
	t.HighlightPattern("/\D{1,}[^:]:/g", "blue")
	Button(window, text="OK", command=window.destroy).pack()

#presents an error window in case of program error
def error_window(error_message):
	window = tkinter.Toplevel(root)
	window.title("ERROR") 
	window.geometry("300x250")
	text = error_message
	t = Label(window, text=text, width=100, height=10, borderwidth=2).pack()
	Button(window, text="OK", command=window.destroy).pack()

#opens dialog to select image
def select_file():
	root.filename = filedialog.askopenfilename(initialdir="../", title="Select image file", filetypes=(("Image files (.jpg, .jpeg, .png)", "*.jpg *.jpeg *.png"), ("all files","*.*")))

	try:
		img_path = root.filename
	except:
		error_window("Root filename not \ncompatible with image path")
		return

	global im
	imtemp = Image.open(img_path).resize(plot_disp_size)
	im = ImageTk.PhotoImage(imtemp)
	image_canvas.itemconfigure(imload, image=im)

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
		error_window("Root filename not \ncompatible with image path")
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
	cv2.imwrite('./temp_resources/cropped/' + os.path.split(img_path)[1], img_crop)

	global im
	imtemp = Image.open('./temp_resources/cropped/' + os.path.split(img_path)[1]).resize(plot_disp_size)
	im = ImageTk.PhotoImage(imtemp)
	image_canvas.itemconfigure(imload, image=im)

#finding regions of interest
def find_roi():
	global number_of_strips

	try:
		global img_path
		img_path = './temp_resources/cropped/' + os.path.split(root.filename)[1]
	except:
		error_window("Image path not defined")
		return
	
	try:
		img_raw = cv2.imread(img_path)
		img_raw = cv2.resize(img_raw, (1032, 688))
	except:
		error_window("Must threshold image first")
		return

	try:
		number_of_strips = int(strip_number.get())

		roi_list = []

		for i in range(number_of_strips):
			roi = cv2.selectROI(img_raw) #select roi
			roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] #crop with selection
			roi_list.append(roi_cropped) #append to global list

		try:
			for i in range(number_of_strips):
				cv2.imwrite('./temp_resources/strip_{}.jpeg'.format(i+1), roi_list[i])
		except:
			print("No ROI selected")
			cv2.destroyAllWindows()
			return

		cv2.destroyAllWindows()
	except:
		error_window("Strip number must be \nin integer format! \n(1, 2, 3, etc.)")
		return
	
	preview_button['state'] = 'normal'
	bounds_button['state'] = 'normal'

#smoothing filter slider
def update_smooth(val):
	global smooth_val
	smooth_val = val
	make_graph()
	os.remove('./temp_resources/temp.png')

#curve smoothing
def smooth(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	return np.convolve(interval, window, mode='valid')

# #updates after baseline selection
# def update_baseline():
# 	#preview_button['state'] = 'normal'

# 	global baseline_grabbed
# 	baseline_grabbed = baseline_choice.get()

# #updates after selecting number of peak bounds
# def update_peaks():
# 	#bounds_button['state'] = 'normal'

# 	global peaks_num_grabbed
# 	peaks_num_grabbed = peak_num_choice.get()

#choosing peak bounds for integration step 
def choose_peak_bounds():
	global bounds
	make_graph(bounds = True)
	return bounds

#horizontal shift slider
def update_h_shift(val):
	global h_shift_val
	h_shift_val = val
	make_graph()
	os.remove('./temp_resources/temp.png')

#vertical shift slider
def update_v_shift(val):
	global v_shift_val
	v_shift_val = val
	make_graph()
	os.remove('./temp_resources/temp.png')

#preview button
def preview_graph():
	make_graph()
	try:
		os.remove('./temp_resources/temp.png')
	except:
		return

	curve_smoothing_slider['state'] = 'normal'
	horizontal_shift_slider['state'] = 'normal'
	vertical_shift_slider['state'] = 'normal'

#displays graph
def make_graph(bounds = False):
	global vals
	vals = []

	#in case matplotlib crashes
	plt.clf()

	strips = []

	try:	
		for i in range(number_of_strips):
			strips.append(Image.open('./temp_resources/strip_{}.jpeg'.format(i+1)).convert("L"))
	except:
		error_window("No ROI selected")
		return
	
	#special treatment for this disaster
	export_button['state'] = 'normal'

	#convert to numpy array
	X = [] # capital to denote 2d array
	 
	for i in range(number_of_strips):
		np_strips = np.array(strips[i])
		temp_a = []

		for strip in np_strips:
			if strip.sum() != 0:
				temp_a.append(strip)

		X.append([float(sum(l))/len(l) for l in zip(*temp_a)])
	
	#initial values
	if len(init_vals) == 0:
		for i in range(number_of_strips):
			init_vals.extend([np.arange(len(X[i])), X[i]])

	S = [] # smoothed x's

	#smoothing	
	if int(smooth_val) > 0:
		for i in range(number_of_strips):
			temp_b = smooth(X[i], int(smooth_val))
			temp_b = temp_b[1:(len(temp_b) - 1)]
			S.append(temp_b)
	else:
		S = X

	#baseline adjustment
	X_mids = []

	if baseline_choice.get() == 101:
		for i in range(number_of_strips):
			X_mids.append(S[i][int(len(S[i])/2)])
			S[i] = [x - X_mids[i] for x in S[i]]

	# flattened_array = S[0] 
	# for i in range(len(S)):
	# 	if i != 0:
	# 		flattened_array.extend(S[i])

	flattened_S = [j for sub in S for j in sub] # turns 2d array into single 1d array
	
	low_val = min(flattened_S)

	for i in range(number_of_strips):
		S[i] = [j-low_val for j in S[i]]

	#convert to numpy array
	# np_top = np.array(top_line)
	# top_line_array = []
	# for elem in np_top:
	# 	if elem.sum() != 0:
	# 		top_line_array.append(elem)
			
	# np_bottom = np.array(bottom_line)
	# bottom_line_array = []
	# for elem in np_bottom:
	# 	if elem.sum() != 0:
	# 		bottom_line_array.append(elem)

	# x1 = [float(sum(l))/len(l) for l in zip(*top_line_array)]
	# x2 = [float(sum(l))/len(l) for l in zip(*bottom_line_array)]

	#initial vals
	# if len(init_vals) == 0:
	# 	t1 = np.arange(len(x1))
	# 	t2 = np.arange(len(x2))
	# 	init_vals.extend([t1, x1, t2, x2])

	# #smoothing
	# if int(smooth_val) > 0:
	# 	x1 = smooth(x1, int(smooth_val))
	# 	x2 = smooth(x2, int(smooth_val))
		
	# 	x1 = x1[1:(len(x1) - 1)]
	# 	x2 = x2[1:(len(x2) - 1)]

	# #baseline adjustment
	# if baseline_grabbed == 101: #midpoint
	# 	x1_mid = x1[int(len(x1)/2)]
	# 	x2_mid = x2[int(len(x2)/2)]

	# 	x1 = [i - x1_mid for i in x1]
	# 	x2 = [i - x2_mid for i in x2]

	#######   END FOR LOOP   ###########

	# #low val (shifts all to y=0 for standard axis)
	# low_val = min(list(np.append(x1, x2)))

	# x1 = [i-low_val for i in x1]
	# x2 = [i-low_val for i in x2]

	
	#converts values to percentages of max intensity to nearest hundredth (to make uniform across pictures)
	high_val = max(flattened_S)
	
	control_peak_indices = []
	
	for i in range(number_of_strips):
		for j in range(len(S[i])):
			S[i][j] = round((float(S[i][j]) / float(high_val)) * 100.00000, 2)

	#peak detection and adjustment
	for i in range(number_of_strips):	
		peak_indices, _ = find_peaks(np.array(S[i]), prominence=5, distance=10, width=5)

		index = 0

		print("STRIP {} PEAKS".format(i+1), peak_indices, [S[i][j] for j in peak_indices])

		for j in peak_indices:
			if S[i][j] > S[i][index]:
				index = j

		control_peak_indices.append(index)

	max_index = np.amax(control_peak_indices)

	T = []

	for i in range(number_of_strips):
		t = np.arange(len(S[i]))
		t = [j+max_index-control_peak_indices[i] for j in t]
		T.append(t)


	# t = [j+int(h_shift_val) for j in t]
	# S[i] = [j+int(v_shift_val) for j in S[i]]

####################################################################################

	# for i in range(len(x1)):
	# 	x1[i] = round((float(x1[i]) / float(high_val)) * 100.00000, 2)
	# for i in range(len(x2)):
	# 	x2[i] = round((float(x2[i]) / float(high_val)) * 100.00000, 2)

	#new auto peak detector for initial horizontal adjustment
	
	# x1_peaks, _ = find_peaks(np.array(x1), height=15, distance=10, width=10)
	# x2_peaks, _ = find_peaks(np.array(x2), height=15, distance=10, width=10)	

	# x1_peak = 0
	# x2_peak = 0

	# for i in x1_peaks:
	# 	if x1[i] > x1[x1_peak]:	
	# 		x1_peak = i

	# for i in x2_peaks:
	# 	if x2[i] > x2[x2_peak]:
	# 		x2_peak = i

	# t1 = np.arange(len(x1))
	# t2 = np.arange(len(x2))

	# if x1_peak < x2_peak:
	# 	t1 = [i+x2_peak-x1_peak for i in t1]
	
	# if x2_peak < x1_peak:
	# 	t2 = [i+x1_peak-x2_peak for i in t2]

	# #manual h and v shift 
	# t1 = [i+int(h_shift_val) for i in t1]
	# x1 = [i+int(v_shift_val) for i in x1]
	
	# plotting_points = [sub[item] for item in range(len(T)) for sub in [S, T]]
	# plotting_points = []
	# print(len(S), len(T))
	# for i in range(len(T)):
	# 	plotting_points.append([S[i], T[i]])

	#bounds selection
	if bounds == True:
		plt.clf()
		plt.figure(figsize=(9.5,5))
		plt.title("Select left and right bounds of Control Peak (right)")
		for i in range(number_of_strips):
			plt.plot(T[i], S[i], linewidth=2)
		clicked = plt.ginput(2)
		plt.close()
		control_peak = [math.floor(float(str(clicked).split(', ')[0][2:])), math.ceil(float(str(clicked).split(', ')[2][1:]))]
		left_point = min(range(len(T[0])), key=lambda i: abs(T[0][i]-control_peak[0]))
		right_point = min(range(len(T[0])), key=lambda i: abs(T[0][i]-control_peak[1]))
		points_right_peak = [left_point + T[0][0], right_point + T[0][0]]
		plt.clf()

		if peak_num_choice.get() == 102:
			plt.clf()
			plt.figure(figsize=(9.5,5))
			plt.title("Select left and right bounds of Test Peak (left)")
			for i in range(number_of_strips):
				plt.plot(T[i], S[i], linewidth=2)
			clicked = plt.ginput(2)
			plt.close()
			test_peak = [math.floor(float(str(clicked).split(', ')[0][2:])), math.ceil(float(str(clicked).split(', ')[2][1:]))]
			left_point = min(range(len(T[0])), key=lambda i: abs(T[0][i]-test_peak[0]))
			right_point = min(range(len(T[0])), key=lambda i: abs(T[0][i]-test_peak[1]))
			points_left_peak = [left_point + T[0][0], right_point + T[0][0]]
			plt.clf()
	
	#matplot plotting
	hfont = {'fontname': 'Arial', 'weight': 'bold', 'size': 45}
	ax = plt.subplot(111)

	# plt.plot(plotting_points, linewidth=2)
	for i in range(len(S)):
		plt.plot(T[i], S[i], linewidth=2)
		#print([T[i], S[i]])
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

	plt.title(str(img_path).split('cropped/')[1], loc = 'right')
	plt.ylabel('Rel. Int. (% max)', **hfont)
	plt.xlabel('Pixel distance', **hfont)

	plt.setp(ax.get_yticklabels(), fontweight="bold", fontname="Arial")
	plt.setp(ax.get_xticklabels(), fontweight="bold", fontname="Arial")
	
	print(len(S), len(T), "\n")
	for i in range(number_of_strips):
		vals.extend([S[i], T[i]])

	strip_legend = []
	for i in range(number_of_strips):
		strip_legend.extend(["Strip {}".format(i+1)])
	plt.legend(strip_legend, frameon=False, prop={'family': 'Arial', 'weight': 'bold', 'size': 32})

	#resizing
	figure = plt.gcf()
	figure.set_size_inches(15, 10)

	#shading of area under curve
	if bounds == True:
		for i in range(number_of_strips):
			try:
				T[i] = T[i].tolist()
			except:
				pass
		
		print("Shading...")
		
		try:
			for i in range(number_of_strips):
				plt.fill_between(T[i], S[i], 0, where = (T[i] > points_right_peak[0]) & (T[i] <= points_right_peak[1]), color = (0, 0, 1, 0.15))
				vals.extend([simps(S[i][T[i].index(points_right_peak[0]):T[i].index(points_right_peak[1])], np.linspace(points_right_peak[0], points_right_peak[1], num=len(S[i][T[i].index(points_right_peak[0]):T[i].index(points_right_peak[1])])), dx=0.01)])
			for i in range(number_of_strips):
				vals.extend([max(S[i][T[i].index(points_right_peak[0]):T[i].index(points_right_peak[1])])])
			print('Worked Control Peak')
		except:
			error_window("Invalid bounds on control peak")
		
		if peak_num_choice.get() == 102:
			try:
				for i in range(number_of_strips):
					plt.fill_between(T[i], S[i], 0, where = (T[i] > points_left_peak[0]) & (T[i] <= points_left_peak[1]), color = (1, 0, 0, 0.15))
					vals.extend([simps(S[i][T[i].index(points_left_peak[0]):T[i].index(points_left_peak[1])], np.linspace(points_left_peak[0], points_left_peak[1], num=len(S[i][T[i].index(points_left_peak[0]):T[i].index(points_left_peak[1])])), dx=0.01)])
				for i in range(number_of_strips):
					vals.extend([max(S[i][T[i].index(points_left_peak[0]):T[i].index(points_left_peak[1])])])
				print('Worked Test Peak')
			except:
				error_window("Invalid bounds on test peak")
		
	global im
	plt.savefig('./temp_resources/temp.png', bbox_inches='tight')
	im = ImageTk.PhotoImage(Image.open('./temp_resources/temp.png').resize(plot_disp_size))
	image_canvas.itemconfigure(imload, image=im)

#saves graph
def save_graph():
	f = filedialog.askdirectory(initialdir='../', title='Choose Location to Save Data')
	if f:
		plt.savefig(f+'/'+re.sub(r'\W','',os.path.split(root.filename)[1].split('.jpg')[0]) + '.png', bbox_inches='tight')
		workbook = xlsxwriter.Workbook(f+'/'+re.sub(r'\W','',os.path.split(root.filename)[1].split('.jpg')[0]) + '_DATA.xlsx')
		worksheet = workbook.add_worksheet()
		#adds a bold format to use to highlight cells
		bold = workbook.add_format({'bold': True})

		#initialize the top row labels, all with bold text
		worksheet.write('A1', 'Top Strip X-values (initial)', bold)
		worksheet.write('B1', 'Top Strip Y-values (initial)', bold)
		worksheet.write('C1', 'Bottom Strip X-values (initial)', bold)
		worksheet.write('D1', 'Bottom Strip Y-Values (initial)', bold)
		worksheet.write('E1', 'Top Strip X-values (adjusted)', bold)
		worksheet.write('F1', 'Top Strip Y-values (adjusted)', bold)
		worksheet.write('G1', 'Bottom Strip X-values (adjusted)', bold)
		worksheet.write('H1', 'Bottom Strip Y-Values (adjusted)', bold)
		worksheet.write('I1', 'Area of control (right) peak - Top Strip', bold)
		worksheet.write('J1', 'Area of control (right) peak - Bottom Strip', bold)
		worksheet.write('K1', 'Area of test (left) peak - Top Strip', bold)
		worksheet.write('L1', 'Area of test (left) peak - Bottom Strip', bold)
		worksheet.write('I3', 'Max of control (right) peak - Top Strip', bold)
		worksheet.write('J3', 'Max of control (right) peak - Bottom Strip', bold)
		worksheet.write('K3', 'Max of test (left) peak - Top Strip', bold)
		worksheet.write('L3', 'Max of test (left) peak - Bottom Strip', bold)
		worksheet.write('I5', 'Left bound of control (right) peak', bold)
		worksheet.write('J5', 'Right bound of control (right) peak', bold)
		worksheet.write('K5', 'Left bound of test (left) peak', bold)
		worksheet.write('L5', 'Right bound of test (left) peak', bold)

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
			worksheet.write('I6', vals[8])
			worksheet.write('J6', vals[9])

		if len(vals) >= 12:
			worksheet.write('K2', vals[10])
			worksheet.write('L2', vals[11])
			worksheet.write('K4', vals[12])
			worksheet.write('L4', vals[13])
			worksheet.write('K6', vals[14])
			worksheet.write('L6', vals[15])

		#inserts cropped ROI image
		worksheet.insert_image('J8', f+'/'+re.sub(r'\W','',os.path.split(root.filename)[1].split('.jpg')[0]) + '.png', {'x_scale': 0.40, 'y_scale': 0.40})	
		worksheet.insert_image('J27', img_path, {'x_scale': 0.40, 'y_scale': 0.40})	

		workbook.close()
		
		print("Data for " + os.path.split(root.filename)[1].split('.jpg')[0] + " successfully exported")

	elif f is None:
		return

#initializes tkinter GUI
def init():
	#setting variables to global scope that need to be accessed outside of init()
	global curve_smoothing_slider, horizontal_shift_slider, vertical_shift_slider, image_canvas, bounds_button, preview_button, export_button, baseline_choice, im, imload, peak_num_choice, strip_number

	left_frame = Frame(root)
	left_frame.pack(side="left")

	middle_frame = Frame(root)
	middle_frame.pack(side="right")

	right_frame = Frame(root)
	right_frame.pack(side="right")

	sub_middle_frame = Frame(middle_frame)
	sub_middle_frame.pack(side="bottom", pady=(0,10))

	#LEFT SIDE
	#help button
	Button(left_frame, text="Help", command=help_window).pack(anchor='nw', padx=(10,0),pady=(10,10))

	#button for selecting image file to analyze
	Button(left_frame, text="Select a file", command=select_file).pack(anchor= 'n',pady=(0,15))

	#Number of strips to be analyzed
	Label(left_frame, text="Enter number of strips to be analyzed: ").pack(anchor='n', pady=(0,0))
	strip_number = Entry(left_frame)
	strip_number.pack(anchor= 'n',pady=(0,15))

	#slider for scaling the cropped image
	Label(left_frame, text="Threshold and Crop", justify="center").pack()
	threshold_slider = Scale(left_frame, orient="horizontal", length=200, from_=1.0, to=30.0, command=update_thresh)
	threshold_slider.pack(padx=20, pady=(0,10))

	#button for selecting the region of interest (ROI), this ROI is then analyzed for the graph
	Button(left_frame, text="Select a ROI", command=find_roi).pack(pady=(0,15))

	#slider for determining how much the curve is smoothed out (typically has very many oscillations and spikes)
	Label(left_frame, text="Curve Smoothing", justify="center", padx=20).pack()
	curve_smoothing_slider = Scale(left_frame, orient="horizontal", length=200, from_=0.0, to=30.0, command=update_smooth)
	curve_smoothing_slider.pack(padx=20, pady=(0,20))
	curve_smoothing_slider['state'] = 'disable'

	#determines whether the baselining will happen from the lowest value (from both curves lowest val is zeroed) or midpoint (average value of both is zeroed and then lowest value brought to zero)
	baseline_choice = tkinter.IntVar()
	baseline_choice.set(101)
	modes = [("Midpoint", 101), ("Lowest Value", 102)]
	Label(left_frame, text="Baseline from:", justify="left", padx=20).pack()
	for mode, val in modes:
		Radiobutton(left_frame, text=mode, indicatoron=1, justify="left", padx=20,  variable=baseline_choice, value=val).pack(anchor='w')

	#a multiple choice field for how many peaks you want analyzed at the current moment
	peak_num_choice = tkinter.IntVar()
	peak_num_choice.set(101)
	modes = [("One Peak", 101), ("Two Peaks", 102)]
	Label(left_frame, text="How many peaks to compare:", justify="left", padx=20).pack(pady=(20,0))
	for mode, val in modes:
		Radiobutton(left_frame, text=mode, indicatoron=1, justify="left", padx=20,  variable=peak_num_choice, value=val).pack(anchor='w')

	#building the bounds button, for selecting left and right bounds of target peaks
	bounds_button = Button(left_frame, text="Choose Bounds", command=choose_peak_bounds)
	bounds_button.pack(side="left", padx=(15,10), pady=(30,10))
	bounds_button["state"] = "disable"

	#building the preview button, used to look at the current strip being analyzed
	preview_button = Button(left_frame, text="Preview", command=preview_graph)
	preview_button.pack(side="left", padx=(10,10), pady=(30,10))
	preview_button["state"] = "disable"

	#building the export button, disabled at first until you have data to export
	export_button = Button(left_frame, text="Export", command=save_graph)
	export_button.pack(side="left", padx=(10,0), pady=(30,10))
	export_button["state"] = "disable"

	#RIGHT SIDE
	#building the horizontal shift slider (used to shift one line left and right)
	Label(sub_middle_frame, text="Horizontal Shift").grid(column=0, row=1, padx=(0,20))
	horizontal_shift_slider = Scale(sub_middle_frame, orient="horizontal", length=300, from_=-10.0, to=10.0, command=update_h_shift)
	horizontal_shift_slider.grid(column=0, row=0, padx=(0,20))
	horizontal_shift_slider['state'] = 'disable'

	#building the vertical shift slider (shifts one line up and down)
	Label(sub_middle_frame, text="Vertical Shift").grid(column=1, row=1)
	vertical_shift_slider = Scale(sub_middle_frame, orient="horizontal", length=300, from_=-10.0, to=10.0, command=update_v_shift)
	vertical_shift_slider.grid(column=1, row=0)
	vertical_shift_slider['state'] = 'disable'

	#right side graph
	width, height = plot_disp_size
	image_canvas = Canvas(middle_frame, width=width, height=height)
	image_canvas.pack(padx=(20,0), pady=(0,0))

	#blanks canvas with a white frame, image_canvas is modified to add the graph onto screen each time
	im = ImageTk.PhotoImage(Image.new("RGB", plot_disp_size, (255, 255, 255)))  #PIL solution
	imload = image_canvas.create_image(0, 0, image=im, anchor='nw')

if __name__ == '__main__':
	init() #builds all the buttons and frames
	
	root.protocol("WM_DELETE_WINDOW", on_closing) #when the "x" is hit to close the window, tkinter needs to handle it in a special way
	root.mainloop() #starts the instance of tkinter (the GUI framework)