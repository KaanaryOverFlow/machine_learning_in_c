#include <stdarg.h>


void note(const char *fmt, ...);
void info(const char *fmt, ...);
void warn(const char *fmt, ...);
void write_line(const char *fmt, ...);
void die(const char *fmt);
void hexdump(const void* data, size_t size);
void pin_cpu(int num);


#define in() {note("Calling => %s",__func__);}
#define out() {note("Returing <= %s",__func__);}
#define FOR(number) for (unsigned long i = 0; i < number; i++)
#define FORR(number) for (unsigned long ii = 0; ii < number; ii++)
