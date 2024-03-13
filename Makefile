CMAKE = cmake
NINJA = ninja
BUILD = build
TARGET = shoz

.PHONY: all clean

all: $(BUILD) $(TARGET)

$(BUILD):
	mkdir -p $(BUILD)

$(TARGET):
	$(CMAKE) -B $(BUILD) -S . -G Ninja
	$(NINJA) -C $(BUILD) -j`nproc`
	ln -sf ./build/src/shoz .

clean:
	$(NINJA) -C $(BUILD) clean
