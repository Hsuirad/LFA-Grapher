# LFA Grapher

A simple utility tool to help scan Lateral Flow Assay (LFA) strips and quantitatively return intensity. Allows the comparison between two or more consecutive strips to see which band is more intense, which is more of a sharp spike, etc.

# Directions:

## Overall Flow:
	1. Import image file
	2. Select regions of interest (ROI)
	3. Choose baseline mode
	4. Preview image and adjust smoothing/shift if necessary
	5. Choose how many peaks will be analyzed based on number of peaks in graph
	6. Select bounds for each peak
	7. Export graph and data to your desired destination

## Importing/Image Manipulation:
	1. Press "Select a file" and choose the image file to be imported
	2. Use the "Threshold and Crop" slider to update the window on the right and adjust the image so that both strips are in view
	3. Press "Select a ROI" to select the Regions of Interest (ROI)
	4. Drag the rectangle over the region of the top strip you want analyzed and nothing more, then press enter/space (press c to cancel the selection)
	5. Repeat for the bottom strip

## Graphing/Bounds Selection:
	1. Under "Baseline from:", choose where to align baselines
		Midpoint (RECOMMENDED): aligns the midpoint value of both curves
		Lowest Value: aligns the lowest value of both curves
	2. Press "Preview" to see the current graph
	3. Change the curve smoothing if desired to get a less bumpy curve using the "Curve Smoothing" slider
	4. Use "Horizontal Shift" and "Vertical Shift" sliders to fix the peak alignments
	5. Under "How many peaks to compare:", choose the number of peaks you would like to analyze
	6. Press "Choose Bounds"
	7. Click on the left, then the right bounds of the right peak
	8. If you selected two peaks, repeat for the left peak

## Exporting/Saving:
	1. Once you are happy with your graph, press "Export" and choose the folder you would like to save your graph and data to
		The data will be saved in an Excel with the table of values for each line and various calculations based on the selected peaks
	2. Close out of the program by clicking the "X" in the top right, or continue for more samples


# Examples:

## Test Image

![test image](test-data/example.jpg)

## Graph + Software

![example image](test-data/screenshot.png)