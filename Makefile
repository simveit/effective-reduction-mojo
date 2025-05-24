OUT_DIR = out
KERNEL_DIR = kernel
MOJO_OUTPUT_FILE = -o $(OUT_DIR)/$@

MOJO_FLAGS = --optimization-level 3
MOJO_BASE = mojo build $(MOJO_FLAGS)

naive_1: kernel/naive_1.mojo
	$(MOJO_BASE) $^ $(MOJO_OUTPUT_FILE)

naive_2: kernel/naive_2.mojo
	$(MOJO_BASE) $^ $(MOJO_OUTPUT_FILE)
