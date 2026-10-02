# MPI Cluster Manager Makefile
# Jalanin di Host

# === COMMAND ===
# make up         		- Build (kalau perlu) dan jalanin semua VM
# make down       		- Matiin semua VM yang lagi jalan
# make clean      		- Matiin semua VM dan bersihin isinya
# make ssh-<nama mesin> - Melakukan SSH ke node yang dipilih
# make status     		- Nampilin status tiap VM (Jalan + PID atau ga jalan samsek)

# Variabel
VM_NAMES := master worker1 worker2 worker3
HOST_SSH_PUB_KEY_PATH := ${HOME}/.ssh/id_ed25519.pub
WORKER_PORTS := 10024 10025 10026
DIST_DIR := ./dist

SSH_MASTER_CMD := ssh -o StrictHostKeyChecking=no -p 10023 mpiuser@localhost
PROJECT_DIR_REMOTE := ~/src
SERIAL_DIR_REMOTE := ~/serial
TC_DIR_REMOTE := ~/test_cases
RSYNC_CMD := rsync -avz --delete -e "ssh -o StrictHostKeyChecking=no -p 10023"
RDIST_CMD := rsync -avz -e "ssh -o StrictHostKeyChecking=no -p $$port"

.PHONY: all up down clean build check-ssh-key distribute pull
.PHONY: run-master run-worker1 run-worker2 run-worker3
.PHONY: stop-master stop-worker1 stop-worker2 stop-worker3
.PHONY: ssh-master ssh-worker1 ssh-worker2 ssh-worker3
.PHONY: status

# Target default
all: up remote-build distribute

up: run-master run-worker1 run-worker2 run-worker3
	@echo "--- [SUCCESS] Semua VM telah dinyalakan. ---"
	@echo "Gunakan 'make ssh-master' untuk masuk ke node master."

down: stop-master stop-worker1 stop-worker2 stop-worker3
	@echo "--- [SUCCESS] Semua VM telah dihentikan. ---"

clean: down
	@echo "--- Membersihkan artifak build dan file state... ---"
	@rm -rf result .result*
	@rm -f .pid-*
	@rm -f *.qcow2 vm-*.log

# -----------------------------------------------------------------
# BUILD & STATUS
# -----------------------------------------------------------------
build: check-ssh-key
	@echo "--- 1. BUILDING MPI CLUSTER VM IMAGES ---"
	export SSH_PUBKEY_PATH=$(HOST_SSH_PUB_KEY_PATH) && \
	nix build .#cluster-vms --impure --extra-experimental-features nix-command --extra-experimental-features flakes

status:
	@echo "--- Status VM (berdasarkan file .pid) ---"
	@for vm in $(VM_NAMES); do \
		if [ -f .pid-$$vm ]; then \
			if ps -p $$(cat .pid-$$vm) > /dev/null; then \
				echo "VM '$$vm' BERJALAN dengan PID $$(cat .pid-$$vm)."; \
			else \
				echo "VM '$$vm' TIDAK BERJALAN (proses tidak ditemukan, hapus file .pid-$$vm)."; \
			fi; \
		else \
			echo "VM '$$vm' TIDAK BERJALAN (file .pid tidak ditemukan)."; \
		fi; \
	done

# -----------------------------------------------------------------
# RUN
# -----------------------------------------------------------------
run-master: build
	@if [ -f .pid-master ] && ps -p $$(cat .pid-master) > /dev/null; then \
		echo "--- [INFO] VM 'master' sudah berjalan. ---"; \
	else \
		echo "--- Menjalankan VM 'master' di background... ---"; \
		nohup ./result/bin/run-master-vm > vm-master.log 2>&1 & \
		echo $$! > .pid-master; sleep 1; \
		echo "--- [SUCCESS] VM 'master' berjalan dengan PID $$(cat .pid-master). ---"; \
	fi

run-worker1: build
	@if [ -f .pid-worker1 ] && ps -p $$(cat .pid-worker1) > /dev/null; then \
		echo "--- [INFO] VM 'worker1' sudah berjalan. ---"; \
	else \
		echo "--- Menjalankan VM 'worker1' di background... ---"; \
		nohup ./result/bin/run-worker1-vm > vm-worker1.log 2>&1 & \
		echo $$! > .pid-worker1; sleep 1; \
		echo "--- [SUCCESS] VM 'worker1' berjalan dengan PID $$(cat .pid-worker1). ---"; \
	fi

run-worker2: build
	@if [ -f .pid-worker2 ] && ps -p $$(cat .pid-worker2) > /dev/null; then \
		echo "--- [INFO] VM 'worker2' sudah berjalan. ---"; \
	else \
		echo "--- Menjalankan VM 'worker2' di background... ---"; \
		nohup ./result/bin/run-worker2-vm > vm-worker2.log 2>&1 & \
		echo $$! > .pid-worker2; sleep 1; \
		echo "--- [SUCCESS] VM 'worker2' berjalan dengan PID $$(cat .pid-worker2). ---"; \
	fi

run-worker3: build
	@if [ -f .pid-worker3 ] && ps -p $$(cat .pid-worker3) > /dev/null; then \
		echo "--- [INFO] VM 'worker3' sudah berjalan. ---"; \
	else \
		echo "--- Menjalankan VM 'worker3' di background... ---"; \
		nohup ./result/bin/run-worker3-vm > vm-worker3.log 2>&1 & \
		echo $$! > .pid-worker3; sleep 1; \
		echo "--- [SUCCESS] VM 'worker3' berjalan dengan PID $$(cat .pid-worker3). ---"; \
	fi

# -----------------------------------------------------------------
# ATURAN STOP
# -----------------------------------------------------------------
stop-master:
	@if [ -f .pid-master ]; then kill $$(cat .pid-master); rm -f .pid-master; echo "--- VM 'master' dihentikan. ---"; else echo "--- VM 'master' tidak berjalan. ---"; fi

stop-worker1:
	@if [ -f .pid-worker1 ]; then kill $$(cat .pid-worker1); rm -f .pid-worker1; echo "--- VM 'worker1' dihentikan. ---"; else echo "--- VM 'worker1' tidak berjalan. ---"; fi

stop-worker2:
	@if [ -f .pid-worker2 ]; then kill $$(cat .pid-worker2); rm -f .pid-worker2; echo "--- VM 'worker2' dihentikan. ---"; else echo "--- VM 'worker2' tidak berjalan. ---"; fi

stop-worker3:
	@if [ -f .pid-worker3 ]; then kill $$(cat .pid-worker3); rm -f .pid-worker3; echo "--- VM 'worker3' dihentikan. ---"; else echo "--- VM 'worker3' tidak berjalan. ---"; fi

# -----------------------------------------------------------------
# SSH
# -----------------------------------------------------------------
ssh-master:
	@echo "--- Menghubungkan SSH ke master (mpiuser@localhost:10023) ---"
	@${SSH_MASTER_CMD}

ssh-worker1:
	@echo "--- Menghubungkan SSH ke worker1 (mpiuser@localhost:10024) ---"
	@ssh -o StrictHostKeyChecking=no -p 10024 mpiuser@localhost

ssh-worker2:
	@echo "--- Menghubungkan SSH ke worker2 (mpiuser@localhost:10025) ---"
	@ssh -o StrictHostKeyChecking=no -p 10025 mpiuser@localhost

ssh-worker3:
	@echo "--- Menghubungkan SSH ke worker3 (mpiuser@localhost:10026) ---"
	@ssh -o StrictHostKeyChecking=no -p 10026 mpiuser@localhost

sync:
	@echo "--- Menyinkronkan direktori 'src' dan 'test_cases' ke VM master... ---"
	@${RSYNC_CMD} ./src/ mpiuser@localhost:${PROJECT_DIR_REMOTE}/
	@${RSYNC_CMD} ../test_cases/ mpiuser@localhost:${TC_DIR_REMOTE}/
	@${RSYNC_CMD} ../serial/ mpiuser@localhost:${SERIAL_DIR_REMOTE}/
	
distribute:
	@echo "--- Menyinkronkan direktori 'src' ke VM worker... ---"
	${RSYNC_CMD} mpiuser@localhost:${PROJECT_DIR_REMOTE}/mpi ${DIST_DIR}/
	@for port in $(WORKER_PORTS); do \
		${RDIST_CMD} ${DIST_DIR}/ mpiuser@localhost:${PROJECT_DIR_REMOTE}/; \
		${RDIST_CMD} ../test_cases/ mpiuser@localhost:${TC_DIR_REMOTE}/; \
	done
	@rm -rf ${DIST_DIR}/
	@echo "--- [SUCCESS] Sinkronisasi selesai. ---"

pull:
	@echo "--- Menarik hasil dari VM master ke host... ---"
	@mkdir -p ./results/
	@${RSYNC_CMD} mpiuser@localhost:~/tc_results/ ./results/
	@echo "--- [SUCCESS] Hasil telah ditarik ke direktori './results/'. ---"

remote-build: sync
	@echo "--- Menjalankan kompilasi di dalam VM master... ---"
	@${SSH_MASTER_CMD} "cd ${PROJECT_DIR_REMOTE} && mpic++ open_mpi.cpp -o mpi -I\$$OPENCV_INCLUDE/opencv4 -L\$$OPENCV_LIB -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs"
	@${SSH_MASTER_CMD} "cd ${PROJECT_DIR_REMOTE} && g++ -o serial ~/serial/serial.cpp -I\$$OPENCV_INCLUDE/opencv4 -L\$$OPENCV_LIB -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs"
	@echo "--- [SUCCESS] Kompilasi selesai. ---"

test-opencv:
	@echo "--- Testing OpenCV detection di VM master... ---"
	@${SSH_MASTER_CMD} "echo 'PKG_CONFIG_PATH:' \$$PKG_CONFIG_PATH && echo 'OPENCV_INCLUDE:' \$$OPENCV_INCLUDE && echo 'OPENCV_LIB:' \$$OPENCV_LIB && pkg-config --modversion opencv4 || echo 'pkg-config failed, but we have manual paths'"

# -----------------------------------------------------------------
# UTIL
# -----------------------------------------------------------------
check-ssh-key:
	@if [ ! -f "$(HOST_SSH_PUB_KEY_PATH)" ]; then \
		echo "Error: Public key SSH tidak ditemukan di $(HOST_SSH_PUB_KEY_PATH)"; \
		exit 1; \
	fi