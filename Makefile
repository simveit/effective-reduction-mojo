OUT_DIR = out
KERNEL_DIR = kernel
MOJO_OUTPUT_FILE = -o $(OUT_DIR)/$@

MOJO_FLAGS = --optimization-level 3
MOJO_BASE = mojo build $(MOJO_FLAGS)

two_pass_1: kernel/two_pass_1.mojo
	$(MOJO_BASE) $^ $(MOJO_OUTPUT_FILE)

two_pass_2: kernel/two_pass_2.mojo
	$(MOJO_BASE) $^ $(MOJO_OUTPUT_FILE)

two_pass_3: kernel/two_pass_3.mojo
	$(MOJO_BASE) $^ $(MOJO_OUTPUT_FILE)

two_pass_4: kernel/two_pass_4.mojo
	$(MOJO_BASE) $^ $(MOJO_OUTPUT_FILE)

one_pass_1: kernel/one_pass_1.mojo
	$(MOJO_BASE) $^ $(MOJO_OUTPUT_FILE)

one_pass_2: kernel/one_pass_2.mojo
	$(MOJO_BASE) $^ $(MOJO_OUTPUT_FILE)

compile_all: two_pass_1 two_pass_2 two_pass_3 two_pass_4 one_pass_1 one_pass_2

run_all: compile_all
	@echo "Running two_pass_1..."; ./$(OUT_DIR)/two_pass_1
	@echo "Running two_pass_2..."; ./$(OUT_DIR)/two_pass_2
	@echo "Running two_pass_3..."; ./$(OUT_DIR)/two_pass_3
	@echo "Running two_pass_4..."; ./$(OUT_DIR)/two_pass_4
	@echo "Running one_pass_1..."; ./$(OUT_DIR)/one_pass_1
	@echo "Running one_pass_2..."; ./$(OUT_DIR)/one_pass_2

clean:
	rm -f $(OUT_DIR)/*
