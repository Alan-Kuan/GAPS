CMAKE = cmake
NINJA = ninja
BUILD = build
LINK = shoz

.PHONY: all compile clean

all: $(BUILD) compile $(LINK)

$(BUILD):
	mkdir -p $(BUILD)
	$(CMAKE) -B $(BUILD) -S . -G Ninja

compile:
	$(NINJA) -C $(BUILD) -j`nproc`

$(LINK):
	ln -sf $(BUILD)/src/shoz .

clean:
	$(NINJA) -C $(BUILD) clean
