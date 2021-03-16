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

#make GUI
root = tkinter.Tk()

root.title("Intensity Grapher")

smooth_val = 0


text_entry_arr = []
text_box_arr = []


def update_thresh(val):
	global thresh_val
	thresh_val = val
	thresh_and_crop()
	print(thresh_val)

def update_thresh2(val):
	global smooth_val
	smooth_val = val
	print(smooth_val)

def thresh_and_crop():
	try:
		img_path = root.filename
		leave=False
	except:
		leave =True

	if leave:
		return

	if 'cropped' not in os.listdir('./'):
		os.mkdir('cropped')

	img = cv2.imread(img_path)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	#have them ROI color picker a color for low thresh

	_, test = cv2.threshold(gray, 255*(float(thresh_val)/100), 255, cv2.THRESH_TOZERO)
	#print(255*(float(thresh_val)/100))

	cnts, hierarchy = cv2.findContours(test, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) 

	cnt = sorted(cnts, key=cv2.contourArea)

	#keep this line pls might use pointpolytest later
	#print(cnt, cv2.pointPolygonTest(cnt[-1], (0, 0), False))

	cv2.drawContours(test, cnt[:-2], -1, 0, -1)
	cnt = cnt[-2:]

	x_min_val = cnt[-1][0][0][0]
	x_max_val = 0
	y_max_val = 0
	y_min_val = cnt[-1][0][0][1]

	#finding lowest x val and highest x val
	for i in range(len(cnt)):
		for j in range(len(cnt[i])):
			for z in range(len(cnt[i][j])):
				f = cnt[i][j]

				if f[z][0] < x_min_val:
					x_min_val = f[z][0]
				if f[z][0] > x_max_val:
					x_max_val = f[z][0]

				if f[z][1] < y_min_val:
					y_min_val = f[z][1]
				if f[z][1] > y_max_val:
					y_max_val = f[z][1]

	#print((y_max_val, y_min_val, x_min_val, x_max_val))
	
	#cropping
	crop = test[y_min_val:y_max_val, x_min_val:x_max_val]

	cv2.imwrite('./cropped/' + os.path.split(img_path)[1], crop)

	global im1
	imtemp = Image.open('./cropped/' + os.path.split(img_path)[1]).resize((450, 300))
	
	im1 = ImageTk.PhotoImage(imtemp)
	c.itemconfigure(theimg, image = im1)

def find_roi():

	try:
		img_path = './cropped/' + os.path.split(root.filename)[1]
	except:
		return
	
	img_raw = cv2.imread(img_path)
	
	#select ROI function
	roi = cv2.selectROI(img_raw)

	roi_cropped = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]

	img_raw = cv2.imread(img_path)
	
	#select ROI function
	roi = cv2.selectROI(img_raw)

	roi_cropped2 = img_raw[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])] 
	
	cv2.imwrite("topline.jpeg",roi_cropped)
	cv2.imwrite('bottomline.jpeg', roi_cropped2)

	cv2.destroyAllWindows()

def is_number(s):
	try:
		float(s)
		return True
	except:
		if s == "-":
			return True
		return False

def make_num(num):
	if num > 3:
		return str(num)+"th"
	elif num == 3:
		return "3rd"
	elif num == 2:
		return "2nd"
	else:
		return "1st"

def button_press():
	root.filename = filedialog.askopenfilename(initialdir = "./",title = "Select image file",filetypes = (("Image files (.jpg, .jpeg, .png)", "*.jpg *.jpeg *.png"), ("all files","*.*")))

def update_choice():
	b["state"] = "normal"
	e["state"] = "normal"
	global grabbed
	grabbed = choice.get()

	try:
		_ = root.filename
	except:
		print("No file chosen")

#curve smoothing
def smooth(interval, window_size):
	window = np.ones(int(window_size))/float(window_size)
	print("window {}".format(window))
	return np.convolve(interval, window, 'valid')


def make_graph():
#UNCOMMENT LATER

	#folder_selected = filedialog.askdirectory(title='Choose Location to Save Data')

	try:
		plt.clf()
	except:
		print('You should be impressed you managed to get this error')
		
	control_line = Image.open('topline.jpeg').convert("L")
	test_line = Image.open('bottomline.jpeg').convert("L")



	# convert to numpy array
	np_control = np.array(control_line)
	control_line_array = []

	for elem in np_control:
		if elem.sum() != 0:
			control_line_array.append(elem)
			
	np_test = np.array(test_line)
	test_line_array = []
	for elem in np_test:
		if elem.sum() != 0:
			test_line_array.append(elem)

	x = [float(sum(l))/len(l) for l in zip(*control_line_array)]
	x2 = [float(sum(l))/len(l) for l in zip(*test_line_array)]

	if int(smooth_val) > 0:
		print(smooth_val)
		print("forst x {}".format(x))
		print(float(len(x)/3 * (float(float(smooth_val)/100.000))))
		x = smooth(x, int(len(x)/3 * (float(float(smooth_val))/100.000)))
		x2 = smooth(x2, int(len(x2)/3 * (float(float(smooth_val))/100.000)))
		print("second x {}".format(x))
		x = x[1:(len(x) - 1)]
		x2 = x2[1:(len(x2) - 1)]

	
	y = np.arange(len(x))
	y2 = np.arange(len(x2))

	print(len(x), len(y))


	xmin_val = xmax_val = ymin_val = ymax_val = 0

	if not is_number(xmin_entry.get()) or xmin_entry.get=="":
		xmin_val = -100000
	else:
		xmin_val = float(xmin_entry.get())
	if not is_number(xmax_entry.get()) or xmax_entry.get=="":
		xmax_val = 100000
	else:
		xmax_val = float(xmax_entry.get())

	if not is_number(ymin_entry.get()) or ymin_entry.get=="":
		ymin_val = -100000
	else:
		ymin_val = float(ymin_entry.get())
	if not is_number(ymax_entry.get()) or ymax_entry.get=="":
		ymax_val = 100000
	else:
		ymax_val = float(ymax_entry.get())

	hfont = {'fontname': 'Arial', 'weight': 'bold', 'size': 45}

	ax = plt.subplot(111)

		#peak detection
	if grabbed == 101:
		x_avg = x[int(len(x)/2)]
		x2_avg = x2[int(len(x2)/2)]

		x = [i - x_avg for i in x]
		x2 = [i - x2_avg for i in x2]

		x_min = x[0]
		x2_min = x2[0]

		minimum = min(x[np.argmin(np.array(x))], x2[np.argmin(np.array(x2))])

		print(minimum, np.argmin(np.array(x)))

		x = [i - minimum for i in x]
		x2 = [i - minimum for i in x2]
	else:
		x_min = x[np.argmin(np.array(x))]
		x2_min = x2[np.argmin(np.array(x2))]
		
		x = [i - x_min for i in x]
		x2 = [i - x2_min for i in x2]

	highest_intensity = max(x[np.argmax(np.array(x))], x2[np.argmax(np.array(x2))])

	#normalization ???? i already forgot check back later but looks like it
	for i in range(len(x)):
		x[i] = round((float(x[i]) / float(highest_intensity)) * 100.00000, 2)
	for i in range(len(x2)):
		x2[i] = round((float(x2[i]) / float(highest_intensity)) * 100.00000, 2)








	'''
	HERE LIES THE DEAD PEAK DETECTION CODE R.I.P (2020-2021)
	FROM NOW ON WE'RE DOING MANUAL PEAK DETECTION USING MATPLOTLIB.GINPUT()



	# make peak detection iterative, if you only found one peak in the control keep increasing and increasing until you only get two
	# in the other one if you get three increase until two or something

	# height, distance, width for the next two lines (height moves at 1.5 the rate of the others)
	h = d = w = 0
	#maybe change increase rates later


	deriv1 = []
	deriv2 = []

	for i in range(len(x) - 1):
		deriv1.append((x[i+1] - x[i]))
	for i in range(len(x2) - 1):
		deriv1.append((x2[i+1] - x2[i]))

	x_temp = np.linspace(0, len(deriv1), len(deriv1))
	x2_temp = np.linspace(0, len(deriv2), len(deriv2))


	plt.clf()
	plt.plot(x_temp, deriv1)
	plt.plot(x2_temp, deriv2)
	plt.show()

	# # me trying to make a do-while loop in python
	while True:
		peaks_x, _ = find_peaks(np.array(x), height=15, distance=10, width=10)
		peaks_x2, _ = find_peaks(np.array(x2), height=15, distance=10, width=10)

		if len(peaks_x2) <= 2 and len(peaks_x) == 0:
		
		else:
			if len(peaks_x) > 2 or len(peaks_x2) > 2:
				#decrease h/d/w (decreasing by half to find local minima)
				h -= 0.75
				d -= 0.5
				w -= 0.5
			elif len(peaks_x) < 2:
				#increase h/d/w
				h += 1.5
				d += 1
				w += 1
			elif h > 100 or d > 100 or w > 100:


	if len(peaks_x2) > 2:
		peaks_x2 = peaks_x2[:2]
	if len(peaks_x) > 2:
		peaks_x = peaks_x[:2]

	if len(peaks_x2) == 2:
		if x2[peaks_x2[1]] > x2[peaks_x2[0]]:
			if peaks_x2[1] > peaks_x[1]:
				y = [i + (peaks_x2[1] - peaks_x[1]) for i in y]
				peaks_x[1] += (peaks_x2[1] - peaks_x[1])
				peaks_x[0] += (peaks_x2[1] - peaks_x[1])
				print('A')
			else:
				y2 = [i + (peaks_x[1] - peaks_x2[1]) for i in y2]
				peaks_x2[1] += (peaks_x[1] - peaks_x2[1])
				peaks_x2[0] += (peaks_x[1] - peaks_x2[1])
				print('D')
		else:
			if peaks_x2[0] > peaks_x[0]:
				y = [i + (peaks_x2[0] - peaks_x[0]) for i in y]
				peaks_x[1] += (peaks_x2[0] - peaks_x[0])
				peaks_x[0] += (peaks_x2[0] - peaks_x[0])
				print('E')
			else:
				y2 = [i + (peaks_x[0] - peaks_x2[0]) for i in y2]
				peaks_x2[1] += (peaks_x[0] - peaks_x2[0])
				peaks_x2[0] += (peaks_x[0] - peaks_x2[0])
				print('F')
	elif len(peaks_x2) == 1:
		if peaks_x2[0] > x2[int(len(x2)/2)]:
			if peaks_x2[0] > peaks_x[1]:
				y = [i + (peaks_x2[0] - peaks_x[1]) for i in y]
				peaks_x[1] += (peaks_x2[0] - peaks_x[1])
				peaks_x[0] += (peaks_x2[0] - peaks_x[1])
				print('Q')
			else:
				y2 = [i + (peaks_x[1] - peaks_x2[0]) for i in y2]
				peaks_x2[0] += (peaks_x[1] - peaks_x2[0])
				print('W')
		else:
			if peaks_x2[0] > peaks_x[0]:
				y = [i + (peaks_x2[0] - peaks_x[0]) for i in y]
				peaks_x[1] += (peaks_x2[0] - peaks_x[0]) 
				peaks_x[0] += (peaks_x2[0] - peaks_x[0]) 
				print('AA')
			else:
				y2 = [i + (peaks_x[0] - peaks_x2[0]) for i in y2]
				peaks_x[0] += (peaks_x[0] - peaks_x2[0])
				print('A')
	
	if y[0] > y2[0]:
		y = y[y2[0]:]
		x = x[y2[0]:]
	else:
		y2 = y2[y[0]:]
		x2 = x2[y[0]:]


	'''



	#remove later just for quick testing
	peaks_x = []
	peaks_x2 = []


	

	print(len(x), len(y))
	plt.plot(y, x, linewidth=3)
	plt.plot(y2, x2, linewidth=3)

	print(len(x), len(y))
	ax.tick_params(width=1)
	ax.spines['right'].set_visible(False)
	ax.spines['top'].set_visible(False)
	ax.yaxis.set_ticks_position('left')
	ax.xaxis.set_ticks_position('bottom')
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax.yaxis.set_minor_locator(AutoMinorLocator(2))
	ax.legend()
	plt.setp(ax.spines.values(), linewidth=1.5)
	ax.tick_params(which="minor", width=1, length=5, labelsize=14)
	ax.tick_params(which="major", width=1.5, length=15, labelsize=32)

	plt.ylabel('Intensity (a.u.)', **hfont)
	plt.xlabel("Pixel distance", **hfont)

	plt.setp(ax.get_yticklabels(), fontweight="bold", fontname="Arial")
	plt.setp(ax.get_xticklabels(), fontweight="bold", fontname="Arial")

	plt.legend(["Top Line", "Bottom Line"], frameon=False, prop={'family': "Arial", "weight": 'bold', "size": 32})

	for i in peaks_x:
		plt.annotate('Peak: {}'.format(x[i]), xy = (i, x[i]))
	for i in peaks_x2:
		plt.annotate('Peak: {}'.format(x2[i]), xy = (i, x2[i]))

	print(peaks_x, peaks_x2)


	# plt.show()
	figure = plt.gcf()

	figure.set_size_inches(15, 10)

	plt.savefig("temp.png",bbox_inches='tight')

	global im1
	im1 = ImageTk.PhotoImage(Image.open('temp.png').resize((450, 300)))
	c.itemconfigure(theimg, image = im1)

	os.remove('./temp.png')

def save_graph():
	#plt.savefig(bbox_inches='tight')
	f = filedialog.asksaveasfilename(defaultextension=".png")
	if f:
		plt.savefig(f, bbox_inches='tight')
	elif f is None:
		return

left_frame = Frame(root)
left_frame.pack(side="left")

middle_frame = Frame(root)
middle_frame.pack(side="right")

right_frame = Frame(root)
right_frame.pack(side="right")

Button(left_frame, text="Select a file", command=button_press).pack(pady=(20, 10))
#Button(left_frame, text="test", command=thresh_and_crop).pack()

Label(left_frame, text="Threshold Slider", justify = "center").pack(pady=(0,5))
s = Scale(left_frame, orient="horizontal", length=200, from_=1.0, to=50.0, command=update_thresh)
s.pack(padx=20, pady=(0, 10))

Button(left_frame, text="Select a ROI", command=find_roi).pack(pady=(0, 15))

Label(left_frame, text="Curve Smoothing", justify = "center", padx = 20).pack()
s2 = Scale(left_frame, orient="horizontal", length=200, from_=0.0, to=100.0, command=update_thresh2)
s2.pack(padx=20, pady=(0, 20))

choice = tkinter.IntVar()
choice.set(1)

modes = [("Midpoint", 101), ("Lowest Value", 102)]

Label(left_frame, text="Baseline from:", justify = "left", padx = 20).pack()
i=0

for mode, val in modes:
	Radiobutton(left_frame, text=mode, indicatoron = 1, command=update_choice, justify ="left", padx = 20,  variable=choice, value=val).pack(anchor ='w')
	i+=1


c = Canvas(middle_frame, width=450, height = 300) #height = width too
c.pack(padx=(20, 0), pady=(0,5))


sub_middle_frame = Frame(middle_frame)
sub_middle_frame.pack(side="bottom", pady=(0, 10))

Label(sub_middle_frame, text="X-Minimum Value: ").grid(column=0,row=0)
xmin_entry = StringVar()
xmin_box = Entry(sub_middle_frame, textvariable=xmin_entry, width=8)
xmin_box.grid(column=1,row=0)
xmin_entry.trace("w", lambda *args: character_limit(xmin_entry))

Label(sub_middle_frame, text="X-Maximum Value: ").grid(column=0,row=1)
xmax_entry = StringVar()
xmax_box = Entry(sub_middle_frame, textvariable=xmax_entry, width=8)
xmax_box.grid(column=1,row=1, padx=10)
xmax_entry.trace("w", lambda *args: character_limit(xmax_entry))

Label(sub_middle_frame, text="Y-Minimum Value: ").grid(column=2,row=0)
ymin_entry = StringVar()
ymin_box = Entry(sub_middle_frame, textvariable=ymin_entry, width=8)
ymin_box.grid(column=3,row=0)
ymin_entry.trace("w", lambda *args: character_limit(ymin_entry))

Label(sub_middle_frame, text="Y-Maximum Value: ").grid(column=2,row=1)
ymax_entry = StringVar()
ymax_box = Entry(sub_middle_frame, textvariable=ymax_entry, width=8)
ymax_box.grid(column=3,row=1)
ymax_entry.trace("w", lambda *args: character_limit(ymax_entry))

e = Button(left_frame, text="Preview", command=make_graph)
e.pack(side="left",padx = (20, 5), pady=(40,10))
e["state"] = "disable"

b = Button(left_frame, text="Save to .png", command=save_graph)
b.pack(side = "right",pady=(40,10))
b["state"] = "disable"


def character_limit(e):
	if len(e.get()) > 8 or is_number(e.get()) == False:
		e.set(e.get()[:-1])

im1 = ImageTk.PhotoImage(Image.new("RGB", (450, 300), (255, 255, 255)))  # PIL solution
theimg = c.create_image(0, 0, image=im1, anchor = 'nw')


root.mainloop()
