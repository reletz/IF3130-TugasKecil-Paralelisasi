NVCC = nvcc
TARGET = cuda
SRCS = cuda.cu
OBJS = $(SRCS:.cu=.o)

NVCC_FLAGS = -diag-suppress 611
INCLUDES = -I$(OPENCV_INCLUDE)/opencv4
LDFLAGS = -L$(OPENCV_LIB)
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

# Default target
all: $(TARGET)

$(TARGET): $(OBJS)
	@echo "Linking $(TARGET)..."
	@$(NVCC) $(NVCC_FLAGS) $(OBJS) $(LDFLAGS) $(LIBS) -o $(TARGET)
	@rm -f $(TARGET).o

%.o: %.cu
	@echo "Compiling $<..."
	@$(NVCC) $(NVCC_FLAGS) $(INCLUDES) -c $< -o $@

clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(OBJS) $(TARGET)

help:
	@echo "Available targets:"
	@echo "  all     - Build the CUDA program (default)"
	@echo "  clean   - Remove build artifacts"
	@echo "  help    - Show this help message"

.PHONY: all clean help