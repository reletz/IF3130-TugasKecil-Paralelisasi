CXX := g++
TARGET := avx2

CXXFLAGS := -mavx2 -mfma -march=native
LDFLAGS := -I$$OPENCV_INCLUDE/opencv4 -L$$OPENCV_LIB -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

SRC_CPP := $(wildcard src/*.cpp)
SRCS := $(SRC_CPP)

OBJS := $(patsubst src/%.cpp,%.o,$(SRC_CPP))

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $^
	rm *.o

%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

clean:
	@rm -f $(OBJS) $(TARGET)