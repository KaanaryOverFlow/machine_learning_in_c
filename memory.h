void kmalloc(size_t size);
void kfree();

void setup_memory_management(void);

void do_shape(size_t size, unsigned int count);

void release_kernel_heap();
