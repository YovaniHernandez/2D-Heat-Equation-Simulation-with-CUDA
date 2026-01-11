#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <sys/time.h>

#define L 100000.0               // Tamaño de la placa (LxL)
#define steps 5000         // Número de pasos de tiempo
#define N 700             // Número de puntos en cada eje
#define tile_w 16         // Tamaño del tile para memoria compartida
#define tile_h 16        // Tamaño del tile para memoria compartida
#define cudaCheckError() {\
 cudaError_t e=cudaGetLastError();\
 if(e!=cudaSuccess) {\
 printf("Cuda Error %s: %d: '%s'\n",\
  __FILE__,__LINE__,cudaGetErrorString(e)); \
  exit(0); } }


/*  Simulación de la difusión de calor de una placa de metal LxL usando CUDA,
    utilizando la ecuacion de calor de ecuaciones diferenciales parciales, a su vez usamos el 
    método explícito de diferencias finitas para resolver la ecuación FTCS.
    dT/dt = k(d²T/dx² + d²T/dy²)

    Condiciones de frontera:
    - T(0,y) = 10
    - T(L,y) = 10
    - T(x,0) = 10
    - T(x,L) = 10
 
    Condiciones iniciales:
    - T(x,y) = 15  si dentro de la placa
    - T(x,y) = 100 si (x - L/2)^2 + (y - L/2)^2 < r^2  con r = L/4

    para que la solucion no diverja, se debe cumplir:
    que el limite de estabilidad sea <= 0.25

*/
__global__ void heat_kernel(float *d_, float *d_new, float r)
{   
    __shared__ float tile[tile_h+2][tile_w+2];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //Calculo de los índices globales de cada hilo
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    //Cargar datos en el centro del tile
    if(i<N && j<N){
        tile[ty+1][tx+1] = d_[i*N + j];
    }
    //Cargar bordes del tile
    //Borde superior
    if(ty==0 && i>0 && j<N){
        tile[0][tx+1] = d_[(i-1)*N + j];
    }
    //Borde inferior
    if(ty==blockDim.y-1 && i<N-1 && j<N){
        tile[tile_h+1][tx+1] = d_[(i+1)*N + j];
    }
    //Borde izquierdo
    if(tx==0 && j>0 && i<N){
        tile[ty+1][0] = d_[i*N + (j-1)];
    }
    //Borde derecho
    if(tx==blockDim.x-1 && j<N-1 && i<N){
        tile[ty+1][tile_w+1] = d_[i*N + (j+1)];
    }   
    __syncthreads();
    //Calculo del nuevo valor de temperatura
    if(i>0 && i<N-1 && j>0 && j<N-1){
        d_new[i*N + j] = tile[ty+1][tx+1] + r*(tile[ty+2][tx+1] + tile[ty][tx+1] +
                        tile[ty+1][tx+2] + tile[ty+1][tx] - 4* tile[ty+1][tx+1]);
    }   

}

void save_csv(const char* filename, float* data){
    FILE* file = fopen(filename, "w");
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            fprintf(file, "%f", data[i*N + j]);
            if(j<N-1) fprintf(file, ",");
        }
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(){

    struct timeval start, end;

    float *h_T, *h_T_new, *h_T_cpu;
    int size = N * N * sizeof(float);
    h_T = (float*)malloc(N*N*sizeof(float));
    h_T_new = (float*)malloc(N*N*sizeof(float));
    h_T_cpu = (float*)malloc(N*N*sizeof(float));
    float *d_T, *d_T_new;
    cudaMalloc((void**)&d_T, size);
    cudaMalloc((void**)&d_T_new, size);
    float h = L / (N - 1); // Tamaño del paso espacial
    float alpha = 0.0001; // Difusividad térmica del material
    float dt = 50.0; // paso de tiempo (debe cumplir el limite de estabilidad <= 0.25)
    float r = alpha * dt / (h * h);
    cudaEvent_t ini, fin;
    float tiempo_gpu;
    
    
    // Inicialización de la placa con condiciones iniciales y de frontera
    for (int i=0;i<N;i++){
        for (int j=0;j<N;j++){
            double x = i * h;
            double y = j * h;

            double dx = x - L / 2.0;
            double dy = y - L / 2.0;
            
            double r2= dx * dx + dy * dy;
            double R2= (L / 4.0) * (L / 4.0);
            if (r2 < R2){
                h_T[i*N + j] = 100.0; // Zona caliente en el centro
            } else {
                h_T[i*N + j] = 15.0;  // Resto de la placa
            }
        }
    }
    gettimeofday(&start, NULL);
    //Bucle de tiempo
    for(int s=0;s<steps;s++){
        //aplicar condiciones de frontera
        for(int k=0;k<N;k++){
            h_T[0*N+k] =10;      // T(0,y) = 10
            h_T[(N-1)*N+k] =10;    // T(L,y) = 10
            h_T[k*N+0] =10;      // T(x,0) = 10
            h_T[k*N+N-1] =10;    // T(x,L) = 10
        }
        //calculo del paso siguiente 
        for(int i=1;i<N-1;i++){
            for(int j=1;j<N-1;j++){
                h_T_new[i*N+j]=h_T[i*N+j] + r*(h_T[(i+1)*N+j] + h_T[(i-1)*N+j] +
                h_T[i*N+j+1] + h_T[i*N+j-1] -4* h_T[i*N+j]);
            }
        }

        //actulaizar la matriz T con los nuevos valores
        for(int i=1;i<N-1;i++){
            for(int j=1;j<N-1;j++){
                h_T[i*N+j]=h_T_new[i*N+j];    
                h_T_cpu[i*N+j]=h_T_new[i*N+j];
            }
        }
    }
    gettimeofday(&end, NULL);
    
    //ejecucion en GPU
    cudaMemcpy(d_T, h_T, size, cudaMemcpyHostToDevice);
    cudaEventCreate(&ini);
    cudaEventCreate(&fin);
    cudaEventRecord(ini, 0);
    dim3 blockDim(16,16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y); 

    //ejecucion del kernel
    for (int s = 0; s < steps; s++){

        heat_kernel<<<gridDim, blockDim>>>(d_T, d_T_new, r);
        cudaCheckError();
        // Intercambiar punteros
        float *temp = d_T;
        d_T = d_T_new;
        d_T_new = temp;
    }
    cudaEventRecord(fin, 0);
    cudaEventSynchronize(fin);
    cudaEventElapsedTime(&tiempo_gpu, ini, fin);
    cudaEventDestroy(ini);
    cudaEventDestroy(fin);

    cudaMemcpy(h_T, d_T, size, cudaMemcpyDeviceToHost);
    save_csv("temp.csv", h_T);
    
    cudaFree(d_T);
    cudaFree(d_T_new);
    
    // Resultados
    float max_error = 0.0f;
    float sum_error = 0.0f;

    for(int i=0;i<N*N;i++){
        float diff = fabs(h_T[i] - h_T_cpu[i]);
        sum_error += diff;
        if(diff > max_error){
            max_error = diff;
        }
    }

    float avg_error = sum_error / (N*N);
    

    printf("Error maximo CPU vs GPU: %e\n", max_error);
    printf("Error medio CPU vs GPU: %e\n", avg_error);

    printf("Temperatura en el centro de la placa despues de %d pasos: %f \n", steps, h_T[(N/2)*N+N/2]);
    printf("Tiempo de ejecucion secuencial: %f segundos\n", 
        ((end.tv_sec - start.tv_sec) + 
        (end.tv_usec - start.tv_usec)/1e6));
    
    //Resultados GPU
    printf("Temperatura en el centro de la placa despues de %d pasos (GPU): %f \n", steps, h_T[(N/2)*N+N/2]);
    printf("Tiempo de ejecucion en GPU: %f segundos\n", tiempo_gpu/1000.0);
    free(h_T);
    free(h_T_new);

    return 0;
}
