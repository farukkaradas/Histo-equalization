#include<stdio.h>
#include<stdlib.h>
#include<time.h>

typedef struct{
    int w;  // image weight
    int h;  // image height
    unsigned int * img; // 1D image pointer arrayx
} PGM_IMG;    

/*
    This function reads PGM B/W images and returns a struct
    path is path of image
    returns a struct that keeps important informations of image
*/
PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    while(getc(in_file) != '\n');             /* skip to end of line */
    while (getc(in_file) == '#'){             /* skip comment lines */
        while (getc(in_file) != '\n');}       /* skip to end of comment line */
    fseek(in_file, -1, SEEK_CUR);             /* backup one character */
    fscanf(in_file, "%d", &result.w);
    fscanf(in_file, "%d", &result.h);
    fscanf(in_file, "%d\n", &v_max);
    printf("Input Image size: %d x %d\n", result.w, result.h);
    result.img = (unsigned int *)malloc(result.w * result.h * sizeof(unsigned int));     
    for(int i=0; i < result.w * result.h; i++)
        fscanf(in_file, "%d ",&result.img[i]);   
    fclose(in_file);
    return result;
}


/*
    This function write PGM (B/W) images and 
    path is output path of image
    img is a struct that keeps important informations of image
*/
void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P2\n%d %d\n255\n",img.w, img.h);
    for (int i = 0; i < img.h*img.w; ++i)
    {
        fprintf(out_file, " %d ", (img.img[i]));
        if (i%17 == 0) fprintf(out_file, "\n\r");
    }
    fclose(out_file);}

void write_rand_pgm(int w,int h, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P2\n %d %d\n255\n",w,h);
    srand(time(NULL));
    
	int i;
	for (i = 0; i< h*w; ++i)
    {
        fprintf(out_file, " %d ", rand() % 155 + 50);
        if(i % 17 == 0) 
			fprintf(out_file, "\n\r"); // newline
    }
    fclose(out_file);
}
void free_pgm(PGM_IMG img){
    free(img.img);} 
