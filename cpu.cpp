#include "image.hpp"
#include <math.h>
#include <iostream>
#include <stdio.h>
#include <cstring>
using namespace std;
float kmax(float a, float b, float c) {
    return ((a > b)? (a > c ? a : c) : (b > c ? b : c));
}

float kmin(float a, float b, float c) {
    return ((a < b)? (a < c ? a : c) : (b < c ? b : c));
}

void rgb2hsv(Image orig,Image gray ,int n)
{
    float h, s, v,  r, g, b;
    float cmax ,cmin,diff;
    int channels=orig._nbChannels;

    for(int i=0;i<n;i+=channels) {

        r =orig._pixels[i]/255.0;
        g =orig._pixels[i+1]/255.0;
        b =orig._pixels[i+2]/255.0;
        cmax = kmax(r, g, b); // maximum of r, g, b
        cmin = kmin(r, g, b); // minimum of r, g, b
        diff = cmax-cmin; // diff of cmax and cmin.
        if (cmax == cmin)
        h = 0;
        else if (cmax == r)
        h = fmod((60 * ((g - b) / diff) + 360), 360.0);
        else if (cmax == g)
        h = fmod((60 * ((b - r) / diff) + 120), 360.0);
        else if (cmax == b)
        h = fmod((60 * ((r - g) / diff) + 240), 360.0);
        // if cmax equal zero
        if (cmax == 0)
           s = 0;
        else
           s = (diff / cmax) * 100;
        // compute v
        v = cmax * 100;

        /*if (i == 0) {
            std::cout<<"rgb pixel 0,0 : ["<< (unsigned short)orig._pixels[0]
                                    <<", "<< (unsigned short)orig._pixels[1]
                                    <<", "<< (unsigned short)orig._pixels[2] << "]" << std::endl;
            std::cout<<"hsv pixel 0,0 : ["<< h <<", "<< s <<", "<< v << "]" << std::endl;
        }*/

        gray._pixels[i]=h;
        gray._pixels[i+1]=s;
        gray._pixels[i+2]=v;
    }
}


void rgb2hsv_v2(Image orig,Image gray ,int n){
    float h, s, v,  r, g, b;
    float cmax ,cmin,diff;
    int channels=orig._nbChannels;

    for(int i=0;i<n;i+=channels) {

        r =orig._pixels[i]/255.0;
        g =orig._pixels[i+1]/ 255.0;
        b =orig._pixels[i+2]/255.0;
        cmax = kmax(r, g, b); // maximum of r, g, b
        cmin = kmin(r, g, b); // minimum of r, g, b
        diff = cmax-cmin; // diff of cmax and cmin.
        if (cmax == cmin)
        h = 0;
        else if (cmax == r)
        h = fmod((60 * ((g - b) / diff) + 360), 360.0);
        else if (cmax == g)
        h = fmod((60 * ((b - r) / diff) + 120), 360.0);
        else if (cmax == b)
        h = fmod((60 * ((r - g) / diff) + 240), 360.0);
        // if cmax equal zero
        if (cmax == 0)
           s = 0;
        else
           s = (diff / cmax) * 100;
        // compute v
        v = cmax * 100;

        /*if (i == 0) {
            std::cout<<"rgb pixel 0,0 : ["<< (unsigned short)orig._pixels[0]
                                    <<", "<< (unsigned short)orig._pixels[1]
                                    <<", "<< (unsigned short)orig._pixels[2] << "]" << std::endl;
            std::cout<<"hsv pixel 0,0 : ["<< h <<", "<< s <<", "<< v << "]" << std::endl;
        }*/

        gray._pixels[i]=h;
        gray._pixels[i+1]=s;
        gray._pixels[i+2]=v;
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
    Image gray(width,height,channels) ;
    Image h_img(width,height,channels) ;
    Image s_img(width,height,channels) ;
    Image v_img(width,height,channels) ;
    //______________________________________________________________________________

    unsigned int n = width * height * channels;
    rgb2hsv(orig,gray,n);

    std::cout << "============================================"	<<std:: endl;
    std::cout << "         Information about The Image Loaded "	<<std:: endl;
    std::cout << "============================================"	<<std:: endl;
    std::cout<<"The width of the image "<<std::endl;
    std::cout<<"The height of the image :"<<orig._height<<std::endl;
    std::cout<<"The number of channels  : "<<orig._nbChannels<<std::endl;
    std::cout << "============================================"	<<std:: endl;

    std::cout << "Information about The Image Result "	<<std:: endl;
    std::cout << "============================================"	<<std:: endl;
    std::cout<<"The width  of the image :"<<gray._width<<std::endl ;
    std::cout<<"The height of the image :"<<gray._height<<std::endl;
    std::cout<<"The number of channels  : "<<gray._nbChannels<<std::endl;

    //std::cout<<"hsv pixel 0,0 : "<<gray._pixels[0]<<std::endl;

    gray.save("Result.png");





    return 0;
}
