CXX ?= g++

# path #
SRC_PATH = src
BUILD_PATH = build
BIN_PATH = $(BUILD_PATH)/bin

# executable # 
BIN_NAMES = benchmark

# extensions #
SRC_EXT = cpp

# ===== IMPORTANTE: só benchmark.cpp + utils.cpp =====
SOURCES = $(SRC_PATH)/benchmark.cpp $(SRC_PATH)/utils.cpp

OBJECTS = $(SOURCES:$(SRC_PATH)/%.$(SRC_EXT)=$(BUILD_PATH)/%.o)
DEPS = $(OBJECTS:.o=.d)

# flags #
COMPILE_FLAGS = -std=c++11 -Wall -Wextra -g -fopenmp
INCLUDES = -I include/ -I /usr/local/include \
           -I /home/intel/oneapi/vtune/2021.1.1/include

LIBS =

.PHONY: default_target
default_target: release

.PHONY: release
release: export CXXFLAGS := $(CXXFLAGS) $(COMPILE_FLAGS)
release: dirs
	@$(MAKE) all

.PHONY: dirs
dirs:
	@echo "Creating directories"
	@mkdir -p $(dir $(OBJECTS))
	@mkdir -p $(BIN_PATH)

.PHONY: clean
clean:
	@echo "Cleaning binaries"
	@$(RM) -r $(BUILD_PATH)
	@$(RM) -r $(BIN_PATH)

-include $(DEPS)

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.$(SRC_EXT)
	@echo "Compiling: $< -> $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MP -MMD -c $< -o $@

# ===== SÓ BENCHMARK =====

$(BIN_PATH)/benchmark: $(BUILD_PATH)/benchmark.o $(BUILD_PATH)/utils.o
	@echo "Linking benchmark"
	$(CXX) $^ -o $@ $(LIBS) -fopenmp \
	    -L /home/intel/oneapi/vtune/2021.1.1/lib64 \
	    -littnotify -ldl

all: $(BIN_PATH)/benchmark