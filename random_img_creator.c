#include<stdio.h>
#include<stdlib.h>
#include "PGM.h"
int main(){
    int w = 4096;
    int h = 4096;
    const char path[] = "4096.pgm";
    write_rand_pgm(w, h, path);
    printf("%d X %d image is saved as %s", w, h, path);
    return 0;
}
