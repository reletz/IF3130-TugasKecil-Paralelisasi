CXX = g++
CXXFLAGS = -fopenmp

# Try to detect OpenCV paths automatically
OPENCV_INCLUDE = $(shell pkg-config --cflags opencv4 2>/dev/null || echo "-I/usr/include/opencv4")
OPENCV_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || echo "-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs")

TARGET = open_mp
SOURCE = open_mp.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET) $(OPENCV_INCLUDE) $(OPENCV_LIBS)

clean:
	rm -f $(TARGET)