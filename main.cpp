#include <iostream>
#include <fstream>
#include <OpenCL/opencl.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

//Szybkość rozchodzenia się fali w ośrodku
#define c 1
#define x 20
#define y 20
#define z 20
#define t 50
#define dt 0.25
#define dx 1
#define dy 1
#define dz 1
#define MAX_SOURCE_SIZE (0x100000)

int probki = t/dt-1;


/******************************** FUNKCJE ***********************************************/

void clearTableInteger(int table[]);
void clearTableDouble(float table[probki][x][y][z]);
void writeTableInteger(int table[]);
void writeTableDouble(float table[probki][x][y][z], int sample);
void zeroingEdges(float table[probki][x][y][z], int sample);
void randomizeArray(cl_int* data, size_t vectorSize);
void zeroingEdges1(float table[x][y][z]);


int main(int argc, const char * argv[]) {
    //TABLICA WARTOŚCI
    float u[probki][x][y][z];
    float Cx = c * dt/dx;
    //float Cy = c * dt/dy;
    //float Cz = c * dt/dz;
    
    //ZMIENNA DO PLIKU
    fstream kopia, kopiaGPU, daneX, daneXGPU;
    
    //CZYSZCZENIE TABLIC
    //clearTableInteger(t);
    clearTableDouble(u);
    
    //IMPULS WEJŚCIOWY
    u[0][10][10][10] = 15;
    
    for(int i=1; i<x-1; i++){
        
        for(int j=1; j<y-1; j++){
            
            for(int k=1; k<z-1; k++){
                u[1][i][j][k] = (2*u[0][i][j][k] + (Cx*Cx)*(u[0][i+1][j][k] + u[0][i-1][j][k] + u[0][i][j+1][k] + u[0][i][j-1][k] + u[0][i][j][k+1] + u[0][i][j][k-1] - 6*u[0][i][j][k]));
            }
        }
    }
    
    
    
    // Create the two input vectors
    const int LIST_SIZE = 20;
    float A[LIST_SIZE][LIST_SIZE][LIST_SIZE];
    float B[LIST_SIZE][LIST_SIZE][LIST_SIZE];
    float C[LIST_SIZE][LIST_SIZE][LIST_SIZE];
    for(int i=0; i<LIST_SIZE; i++) {
        for(int j=0; j<LIST_SIZE; j++){
            for(int k=0; k<LIST_SIZE; k++){
                A[i][j][k] = u[0][i][j][k];
                B[i][j][k] = u[1][i][j][k];
                C[i][j][k] = 0;
            }
        }
    }
    //A[LIST_SIZE/2][LIST_SIZE/2][LIST_SIZE/2] = 15;
    zeroingEdges1(A);
    zeroingEdges1(B);
    zeroingEdges1(C);
    
    
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
    
    fp = fopen("/Users/mateuszstolowski/Downloads/SESJA/ProjektCUDA/Projekt/Projekt/kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );
    
    // Get platform and device information
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1,
                         &device_id, &ret_num_devices);
    
  
    
    //Zerowanie brzegów
    zeroingEdges(u,1);
    
    // Create an OpenCL context
    cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);
    
    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);
    
    // Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      LIST_SIZE * LIST_SIZE * LIST_SIZE * sizeof(float), NULL, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      LIST_SIZE * LIST_SIZE * LIST_SIZE * sizeof(float), NULL, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                      LIST_SIZE * LIST_SIZE * LIST_SIZE * sizeof(float), NULL, &ret);
    
    
    
    
    for(int sample = 2; sample < probki-1; sample++){
        // Copy the lists A to its respective memory buffer
        ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                                          LIST_SIZE * LIST_SIZE * LIST_SIZE * sizeof(float), A, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
                                          LIST_SIZE * LIST_SIZE * LIST_SIZE * sizeof(float), B, 0, NULL, NULL);
        ret = clEnqueueWriteBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                                          LIST_SIZE * LIST_SIZE * LIST_SIZE * sizeof(float), C, 0, NULL, NULL);
        
        // Create a program from the kernel source
        cl_program program = clCreateProgramWithSource(context, 1,
                                                       (const char **)&source_str, (const size_t *)&source_size, &ret);
        
        // Build the program
        ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);
        
        // Create the OpenCL kernel
        cl_kernel kernel = clCreateKernel(program, "ProjectCUDA", &ret);
        
        // Set the arguments of the kernel
        ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&a_mem_obj);
        ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&b_mem_obj);
        ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&c_mem_obj);
        
        // Execute the OpenCL kernel on the list
        //size_t global_item_size = LIST_SIZE; // Process the entire lists
        //size_t local_item_size = LIST_SIZE*LIST_SIZE; // Divide work items into groups of 64
        
        size_t global_item_size[3] = { LIST_SIZE, LIST_SIZE, LIST_SIZE };
        size_t local_item_size[3] = {1,1,1};
        
        
        
        // Read the memory buffer C on the device to the local variable C
        
        ret = clEnqueueNDRangeKernel(command_queue, kernel, 3, NULL,
                                     global_item_size, local_item_size, 0, NULL, NULL);
        
        ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                                  LIST_SIZE * LIST_SIZE * LIST_SIZE * sizeof(float), C, 0, NULL, NULL);
        /*ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                                  LIST_SIZE * LIST_SIZE * LIST_SIZE * sizeof(float), A, 0, NULL, NULL);
        ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                                  LIST_SIZE * LIST_SIZE * LIST_SIZE * sizeof(float), B, 0, NULL, NULL);*/
        zeroingEdges1(A);
        zeroingEdges1(B);
        zeroingEdges1(C);
        for(int i=0; i<LIST_SIZE; i++) {
            for(int j=0; j<LIST_SIZE; j++){
                for(int k=0; k<LIST_SIZE; k++){
                    u[sample][i][j][k] = C[i][j][k];
                    A[i][j][k] = B[i][j][k];
                    B[i][j][k] = C[i][j][k];
                    C[i][j][k] = 0;
                }
            }
        }
        ret = clReleaseKernel(kernel);
        ret = clReleaseProgram(program);
    }
    
    
    
    
    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    
    
    
    
    
    /*///ALGORYTM DLA T = 1,2, ... ,99
    for(int s = 1; s<probki-1; s++){
        
        for(int i=1; i<x-1; i++){
            
            for(int j=1; j<y-1; j++){
                
                for(int k=1; k<z-1; k++){
                    u[s+1][i][j][k] = (2*u[s][i][j][k] - u[s-1][i][j][k] + (Cx*Cx)*(u[s][i+1][j][k] + u[s][i-1][j][k] + u[s][i][j+1][k] + u[s][i][j-1][k] + u[s][i][j][k+1] + u[s][i][j][k-1] - 6*u[s][i][j][k]));
     
                }
            }
        }
        //Zerowanie brzegów
        zeroingEdges(u, s);
    }*/
    
    //Zapisanie danych do pliku
    kopiaGPU.open("/Users/mateuszstolowski/Downloads/SESJA/ProjektCUDA/Projekt/KOPIAGPU.txt", std::ios::out);
    daneXGPU.open("/Users/mateuszstolowski/Downloads/SESJA/ProjektCUDA/Projekt/daneXGPU.txt", std::ios::out);
    //Sprawdzenie poprawnego otwarcia pliku
    if(kopiaGPU.good() && daneXGPU.good()){
        cout << "Plik został otwarty pomyślnie\n";
        kopiaGPU << "  t    x    y    z    U" << endl;
        
        //OPERACJE NA PLIKU
        for(int s=0; s<probki-1; s++){
            for(int i=0; i<x; i++){
                
                for(int j=0; j<y; j++){
                    
                    for(int k=0; k<z; k++){
                        daneXGPU << s << " " << i << " " << j << " " << k << " " << u[s][i][j][k] << endl;
                    }
                }
            }
        }
        kopiaGPU.close();
        daneXGPU.close();
        cout << "Pliki zostały zamknięte" << endl;
    }
    else cout << "Nie udało się otworzyć pliku\n";
    

    
   
   
    
    return 0;
}



void clearTableInteger(int table[]){
    for(int i=0; i<t; i++){
        table[i]=i;
    }
}


/**/
void clearTableDouble(float table[probki][x][y][z]){
    for(int s=0; s<probki; s++){
        
        for(int i=0; i<x; i++){
            
            for(int j=0; j<y; j++){
                
                for(int k=0; k<z; k++){
                    
                    table[s][i][j][k] = 0;
                }
            }
        }
    }
}

/**/
void writeTableInteger(int table[]){
    for(int i=0; i<t; i++){
        cout << table[i];
    }
}

/**/
void writeTableDouble(float table[probki][x][y][z], int sample){
    int numOfElem = 0;
    for(int s=0; s<sample; s++){
        cout << "############################ NEXT TIME SAMPLE  S: " << s << " ###############################" << endl;
        for(int i=0; i<x; i++){
            cout << "X: " << i << endl;
            
            for(int j=0; j<y; j++){
                
                for(int k=0; k<z; k++){
                    //cout << "X: " << i << " Y: " << j << " Z: " << k << endl;
                    cout << " " << table[s][i][j][k] << " ";
                    numOfElem++;
                }
                cout << endl;
            }
            cout << "\n\n!\n\n";
        }
    }
    cout << "\nLiczba wypisanych elementów: " << numOfElem << endl;
}

void zeroingEdges(float table[probki][x][y][z], int sample){
    for(int i=0; i<x; i++){
        
        for(int j=0; j<y; j++){
            
            for(int k=0; k<z; k++){
                if((i == 0) || (i == x-1) || (j == 0) || (j == y-1) || (k == 0) || (k == z-1)){
                    table[sample][i][j][k] = 0;
                }
            }
        }
    }
}

void zeroingEdges1(float table[x][y][z]){
    for(int i=0; i<x; i++){
        
        for(int j=0; j<y; j++){
            
            for(int k=0; k<z; k++){
                if((i == 0) || (i == x-1) || (j == 0) || (j == y-1) || (k == 0) || (k == z-1)){
                    table[i][j][k] = 0;
                }
            }
        }
    }
}
