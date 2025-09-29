.PHONY: all build run clean_build_artifacts

# Default path SSH Public Key. Ubah kalau beda :)
SSH_KEY_PATH = ${HOME}/.ssh/id_ed25519.pub

CLUSTER_PACKAGE = cluster-vms
RUNNER = ./result/bin/run-master-vm
TARGET_EXEC = mpi

# 1. BUILD VM (HOST)
# Target buat build semua VM
build:
	@echo "--- 1. EXPORTING SSH KEY FOR NIX BUILD ---"
	export SSH_PUBKEY_PATH=$(SSH_KEY_PATH) && \
	echo "Using key: $${SSH_PUBKEY_PATH}"
	
	@echo "--- 2. BUILDING MPI CLUSTER VM IMAGES ---"
	export SSH_PUBKEY_PATH=$(SSH_KEY_PATH) && \
	nix build .#$(CLUSTER_PACKAGE) --impure

# 2. RUN VM (HOST)
# Ngejalanin master-vm
run:
	@echo "--- RUNNING MPI CLUSTER (QEMU WINDOWS WILL APPEAR) ---"
	@echo "Setelah boot, SSH ke: ssh mpiuser@localhost -p 10022 (Pass: mpipass)"
	$(RUNNER)
	