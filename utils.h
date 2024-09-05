#include <stdarg.h>


void note(const char *fmt, ...);
void info(const char *fmt, ...);
void warn(const char *fmt, ...);
void write_line(const char *fmt, ...);
void die(const char *fmt);
void hexdump(const void* data, size_t size);
void pin_cpu(int num);
void print_progress_bar(size_t progress, size_t total);

#define pd(val) note("int(%s) : %d", #val, val)
#define pu(val) note("unsigned int(%s) : %u", #val, val)
#define px(val) note("int hex(%s) : %x", #val, val)
#define pf(val) note("float(%s) : %f", #val, val)
#define plu(val) note("unsigned long(%s) : %lu", #val, val)
#define pld(val) note("long(%s) : %ld", #val, val)
#define plx(val) note("long hex(%s) : %lx", #val, val)
#define plf(val) note("double(%s) : %lf", #val, val)


#define in() {note("Calling => %s",__func__);}
#define out() {note("Returing <= %s",__func__);}
#define FOR(iterator, number) for (size_t iterator = 0; iterator < number; iterator++)
