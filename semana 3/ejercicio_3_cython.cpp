#include <iostream>
#include <fstream>
#include <math.h>
float funtion(float x){
    float potencia=pow(x,2);
    float y=pow(M_E,-potencia);
    return y;
}

float derivada_central(float x,float h){
    float a = funtion(x+h);
    float b = funtion(x-h);
    float d= (a-b)/(2*h);
    return d;
}
int main(){
    std::ofstream *File;
    File = new std::ofstream[1];
    File[0].open("Mis datos.txt",std::ofstream::trunc);
    float h=0.01;
    float a=-20.0;
    float b= 20.0;
    float i=a;
    while (i<b){
        float f_prima=derivada_central(i,h);
        File[0]<<i<<" "<<f_prima<<std::endl;
        i+=h;
    }
        

    File[0].close();
    return 0;
}
