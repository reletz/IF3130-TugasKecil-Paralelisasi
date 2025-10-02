CXX := g++
BUILD_DIR := build
TARGET := avx2

CXXFLAGS := -std=gnu++17  -mavx2 -mfma -march=native
LDFLAGS := -I$$OPENCV_INCLUDE -L$$OPENCV_LIB -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

SRC_CPP := $(wildcard src/*.cpp)
SRCS := $(SRC_CPP)

OBJS := $(patsubst src/%.cpp,$(BUILD_DIR)/%.o,$(SRC_CPP))

.PHONY: all clean distclean

all: $(BUILD_DIR)/$(TARGET)

$(BUILD_DIR)/$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(LDFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: src/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -c $< -o $@

clean:
	@rm -rf $(BUILD_DIR)

distclean: clean