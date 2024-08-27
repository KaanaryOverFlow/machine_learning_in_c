android=$(shell locate toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android33-clang | head -1)
default=clang

CC=$(android)

all:
	$(CC) *.c -o exploit
#	adb push exploit /data/local/tmp
	scp ./exploit root@localhost:/tmp



