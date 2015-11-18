all: clean v1 v2

clean:
	rm -rf convolution

v1: clean
	nvcc -arch=sm_21 -w -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -Iutil convolution_v1.cu -o convolution
	
v3: clean
	nvcc -arch=sm_21 -w -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -DEBUG -g -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -Iutil convolution_v3.cu -o convolution

v2: clean
	nvcc -arch=sm_21 -w -I/usr/local/include/opencv -I/usr/local/include -L/usr/local/lib -DEBUG -g -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lopencv_video -lopencv_features2d -lopencv_calib3d -lopencv_objdetect -lopencv_contrib -lopencv_legacy -Iutil convolution_v2.cu -o convolution
