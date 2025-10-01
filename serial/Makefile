# Serial

CXX = g++
TARGET = serial
SRCS = serial.cpp
OBJS = $(SRCS:.cpp=.o)
INCLUDES = -I$(OPENCV_INCLUDE)/opencv4
LDFLAGS = -L$(OPENCV_LIB)

LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	@$(CXX) $(OBJS) $(LDFLAGS) $(LIBS) -o $(TARGET)
	@rm serial.o

%.o: %.cpp
	@$(CXX) $(INCLUDES) -c $< -o $@

clean:
	@rm -f $(OBJS) $(TARGET)

.PHONY: all clean