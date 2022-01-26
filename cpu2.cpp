#include "image.hpp"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <cstring>

#define PRECISION_HIST 100

using namespace std;

//g++ -o cpu2 cpu2.cpp image.cpp

struct HSVcolor{
  float h;
  float s;
  float v;
};
typedef struct HSVcolor HSVcolor;

struct RGBcolor{
  int r;
  int g;
  int b;
};
typedef struct RGBcolor RGBcolor;

float fmin2(float a, float b){ return a < b ? a : b; }
float fmax2(float a, float b){ return a > b ? a : b; }
int round2(float f){ return (int)(f+0.5); }

/*source: https://gist.github.com/mjackson/5311256*/
void HSVtoRGB(const HSVcolor* in, RGBcolor* out) {
  float fr, fg, fb;

  int i = (int)(in->h * 6);
  float f = in->h * 6 - i;
  float p = in->v * (1 - in->s);
  float q = in->v * (1 - f * in->s);
  float t = in->v * (1 - (1 - f) * in->s);

  switch (i % 6) {
    case 0: fr = in->v, fg = t, fb = p; break;
    case 1: fr = q, fg = in->v, fb = p; break;
    case 2: fr = p, fg = in->v, fb = t; break;
    case 3: fr = p, fg = q, fb = in->v; break;
    case 4: fr = t, fg = p, fb = in->v; break;
    case 5: fr = in->v, fg = p, fb = q; break;
  }
  out->r = round2(fr * 255);
  out->g = round2(fg * 255);
  out->b = round2(fb * 255);

}

void HSVtoRGBv2(const HSVcolor* in, RGBcolor* out) {
  float fr, fg, fb;

  float C = in->v * in->s;
  float X = C * (1 - abs(fmod((in->h/60), 2) - 1));
  float m = in->v - C;


  if (in->h < 60) {
      fr = C; fg = X; fb = 0;
  } else if (in->h < 120) {
      fr = X; fg = C; fb = 0;
  } else if (in->h < 180) {
      fr = 0; fg = C; fb = X;
  } else if (in->h < 240) {
      fr = 0; fg = X; fb = C;
  } else if (in->h < 300) {
      fr = X; fg = 0; fb = C;
  } else if (in->h < 360) {
      fr = C; fg = 0; fb = X;
  }

  out->r = round2((fr + m) * 255);
  out->g = round2((fg + m) * 255);
  out->b = round2((fb + m) * 255);

}


/*source: https://gist.github.com/mjackson/5311256*/
void RGBtoHSV(const RGBcolor* in, HSVcolor* out) {
  float fr = in->r/255.f;
  float fg = in->g/255.f;
  float fb = in->b/255.f;

  float max = fmax2(fmax2(fr, fg), fb);
  float min = fmin2(fmin2(fr, fg), fb);
  float h, s, v = max;

  float d = max - min;
  s = max == 0 ? 0 : d / max;

  if (max == min) {
    h = 0; // achromatic
  } else {
    if(max == fr)
    {h = (fg - fb) / d + (fg < fb ? 6 : 0);}
    else if(max == fg)
    {h = (fb - fr) / d + 2;}
    else
    {h = (fr - fg) / d + 4;}

    h /= 6;
  }
  out->h = h;
  out->s = s;
  out->v = v;
}

void RGBtoHSVv2(const RGBcolor* in, HSVcolor* out)
{
    float h, s, v,  r, g, b;
    float cmax ,cmin, diff;

    r =in->r/255.0;
    g =in->g/255.0;
    b =in->b/255.0;
    cmax = fmax2(fmax2(r, g), b); // maximum of r, g, b
    cmin = fmin2(fmin2(r, g), b); // minimum of r, g, b
    diff = cmax-cmin; // diff of cmax and cmin.
    if (cmax == cmin)
    h = 0;
    else if (cmax == r)
    h = 60*fmod((g - b)/diff, 6);
    else if (cmax == g)
    h = 60*( ((b - r)/diff) + 2 );
    else if (cmax == b)
    h = 60*( ((r - g)/diff) + 4 );
    // if cmax equal zero
    if (cmax == 0)
       s = 0;
    else
       s = (diff / cmax) * 100;
    // compute v
    v = cmax;

    out->h = h;
    out->s = s;
    out->v = v;
}


/* fonction de répartition */
int rep(int l, unsigned int* hist){
  int somme = 0;
  for (size_t i=0; i<=l; i++) {
    somme += hist[i];
  }
  return somme;
}


void print_bits_char(char c) {
    for(int bit=0;bit<(sizeof(char) * 8); bit++) {
        printf("%i", c & 0x01);
        c = c >> 1;
    }
}



int main()
{
    //_____________________________________________________________________________
    int width,height,channels;
    Image orig;
    orig.load("img/Chateau.png") ;
    width=orig._width;
    height=orig._height;
    channels=orig._nbChannels;

    //______________________________________________________________________________
    Image img_out(width,height,channels) ;
    //______________________________________________________________________________

    unsigned int n = width * height * channels;

    std::cout << "============================================"	<<std:: endl;
    std::cout << "         Information about The Image Loaded "	<<std:: endl;
    std::cout << "============================================"	<<std:: endl;
    std::cout<<"The width of the image "<<std::endl;
    std::cout<<"The height of the image :"<<orig._height<<std::endl;
    std::cout<<"The number of channels  : "<<orig._nbChannels<<std::endl;
    std::cout << "============================================"	<<std:: endl;

    RGBcolor rgbtest = {217, 204, 211};
    HSVcolor hsvtest;
    RGBtoHSVv2(&rgbtest, &hsvtest);
    printf("rgb[%d,%d,%d] to hsv: [%f,%f,%f]\n", rgbtest.r, rgbtest.g, rgbtest.b, hsvtest.h, hsvtest.s, hsvtest.v);
    HSVtoRGBv2(&hsvtest, &rgbtest);
    printf("hsv[%f,%f,%f] to rgb: [%d,%d,%d]\n", hsvtest.h, hsvtest.s, hsvtest.v, rgbtest.r, rgbtest.g, rgbtest.b);


    unsigned int histv[PRECISION_HIST+1]; /*de 0 à 100*/
    RGBcolor pixelIn, pixelOut;
    /*for (size_t i=0; i<PRECISION_HIST+1; i++) {
        histv[i] = 0;
    }

    for ( int i = 0; i < width * height; ++i ) {
        pixelIn.r = orig._pixels[i*3];
        pixelIn.g = orig._pixels[i*3+1];
        pixelIn.b = orig._pixels[i*3+2];
        if (i<10) {
            printf("rgb: [%d,%d,%d]\n", pixelIn.r, pixelIn.g, pixelIn.b);
        }
        HSVcolor hsvc;
        RGBtoHSV(&pixelIn, &hsvc);
        histv[ (int)(hsvc.v*PRECISION_HIST+0.5) ]++;
  	}
    for (size_t i = 0; i < PRECISION_HIST+1; i++) {
        printf("%d ", histv[i]);
        if ((i+1)%12 == 0) {
            printf("\n");
        }
    }printf("\n");*/

    HSVcolor hsvc, newhsvc;
    for ( int i = 0; i < width*height; ++i ) {
        pixelIn.r = orig._pixels[i*3];
        pixelIn.g = orig._pixels[i*3+1];
        pixelIn.b = orig._pixels[i*3+2];
        RGBtoHSV(&pixelIn, &hsvc);

        newhsvc.h = hsvc.h;
        newhsvc.s = hsvc.s;
        //newhsvc.v = ( (PRECISION_HIST-1) / (float)(PRECISION_HIST*width*height)) * (rep( (int)(hsvc.v*PRECISION_HIST+0.5), histv) );
        newhsvc.v = hsvc.v;
        //if (i<10) { printf("hsv[%f,%f,%f]\n", newhsvc.h, newhsvc.s, newhsvc.v); }

        HSVtoRGB(&newhsvc, &pixelOut);
  		img_out._pixels[i*3]    = pixelOut.r;
        img_out._pixels[i*3+1]  = pixelOut.g;
        img_out._pixels[i*3+2]  = pixelOut.b;
    }




    img_out.save("Result.png");





    return 0;
}
