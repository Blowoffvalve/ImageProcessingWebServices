import cv2
import time
import numpy as np
import psutil
import os
image = cv2.imread("Image.jpg")

def saveImage(imageFormats):
	"""
	Print the performance characteristics for outputting an image using various output formats
	"""
	pid = os.getpid()
	py = psutil.Process(pid)
	formatTimers = {}
	bytesToGB = 1024**3
	for imageFormat in imageFormats:
		formatTimer, cpu, memory = [], [], []
		for i in range(30):
			starttime = time.time()
			outputFile = "newImage."+ imageFormat
			cv2.imwrite(outputFile, image)
			duration = time.time()-starttime
			formatTimer.append(duration)
			memory.append(py.memory_info()[0]/bytesToGB)
			cpu.append(psutil.cpu_percent())
		formatTimers[imageFormat] = {"AverageRunTime":np.mean(formatTimer), "AverageCPU": np.mean(cpu), "AverageMemory":np.mean(memory)}
		print(imageFormat, ": AverageRunTime ", str(np.mean(formatTimer)), " AverageCPU ", str(np.mean(cpu)), " AverageMemory ", str(np.mean(memory)), sep = "")
		
saveImage(["bmp", "dib", "jpeg", "jpg", "jpe", "jp2", "png", "webp", "pbm", "pgm", "ppm", "pnm", "sr", "tiff", "tif", "hdr", "pic"])