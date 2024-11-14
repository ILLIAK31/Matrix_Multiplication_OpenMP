#include <iostream>
#include <omp.h>
#include <vector>
#include <thread>
#include <chrono>

using namespace std;

int** a;
int** b;
int** c;
int*** C; 
int** bt;

int** allocateMatrix(int N) 
{
    int** matrix = new int*[N];
    for (int i = 0; i < N; ++i) 
    {
        matrix[i] = new int[N];
    }
    return matrix;
}

void deallocateMatrix(int** matrix, int N) 
{
    for (int i = 0; i < N; ++i) 
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}

int*** allocate3DMatrix(int num_threads, int tileSize) 
{
    int*** matrix = new int**[num_threads];
    for (int t = 0; t < num_threads; ++t) 
    {
        matrix[t] = new int*[tileSize];
        for (int i = 0; i < tileSize; ++i) 
        {
            matrix[t][i] = new int[tileSize];
            for (int j = 0; j < tileSize; ++j) 
            {
                matrix[t][i][j] = 0;
            }
        }
    }
    return matrix;
}

void deallocate3DMatrix(int*** matrix, int num_threads, int tileSize) 
{
    for (int t = 0; t < num_threads; ++t) 
    {
        for (int i = 0; i < tileSize; ++i) 
        {
            delete[] matrix[t][i];
        }
        delete[] matrix[t];
    }
    delete[] matrix;
}

void multiply(int N) 
{
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; ++i) 
        {      
            for (int j = 0; j < N; ++j) 
            {
                for (int k = 0; k < N; ++k) 
                {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
}

void multiply_P(int tid, int N, int num_threads)
{
    int lb = (N / num_threads) * tid;
    int ub = lb + (N / num_threads) - 1;
    if (tid == num_threads - 1)
        ub = N - 1;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = lb; i <= ub; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                for (int k = 0; k < N; ++k)
                {
                    c[i][j] += a[i][k] * b[k][j];
                }
            }
        }
    }
}

void multiply_P_Transposed(int tid, int N, int num_threads)
{
    int lb = (N / num_threads) * tid;
    int ub = lb + (N / num_threads) - 1;
    if (tid == num_threads - 1)
        ub = N - 1;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = lb; i <= ub; ++i)
        {
            for (int j = 0; j < N; ++j)
            {
                for (int k = 0; k < N; ++k)
                {
                    c[i][j] += a[i][k] * bt[j][k];
                }
            }
        }
    }
}

void multiplyWithTiling(int N, int tileSize) 
{
    int num_threads = omp_get_max_threads();

    int*** C = allocate3DMatrix(num_threads, tileSize);    
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        #pragma omp for
        for (int jj = 0; jj < N / tileSize; jj++) 
        {
            for (int ii = 0; ii < N / tileSize; ii++) 
            {

                for (int j = jj * tileSize; j < (jj + 1) * tileSize; j++) 
                {
                    for (int i = ii * tileSize; i < (ii + 1) * tileSize; i++) 
                    {
                        int i_tab = i - ii * tileSize;
                        int j_tab = j - jj * tileSize;

                        for (int k = 0; k < N; k++) 
                        {
                            C[tid][i_tab][j_tab] += a[i][k] * b[k][j];
                        }
                    }
                }

                for (int i = 0; i < tileSize && (ii * tileSize + i) < N; i++) 
                {
                    for (int j = 0; j < tileSize && (jj * tileSize + j) < N; j++) 
                    {
                        #pragma omp atomic
                        c[ii * tileSize + i][jj * tileSize + j] += C[tid][i][j];
                        C[tid][i][j] = 0;
                    }
                }
            }
        }
    }

    deallocate3DMatrix(C, num_threads, tileSize);
}

int main() 
{
    int num_threads;
    cout << "Enter number of threads (1,2,4...): ";
    cin >> num_threads;
    int N;
    cout << "Enter matrix size N: ";
    cin >> N;

    omp_set_num_threads(num_threads);

    a = allocateMatrix(N);
    b = allocateMatrix(N);
    c = allocateMatrix(N);
    bt = allocateMatrix(N); 
   
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; ++i) 
        {
            for (int j = 0; j < N; ++j) 
            {
                a[i][j] = rand() % 100;
                b[i][j] = rand() % 100;
                c[i][j] = 0;
            }
        }

        #pragma omp for
        for (int i = 0; i < N; ++i) 
        {
            for (int j = 0; j < N; ++j) 
            {
                bt[i][j] = b[j][i];  
            }
        }
    }

    double start = omp_get_wtime();
    multiply(N); 
    double end = omp_get_wtime();
    cout << "Multiplication time: " << (end - start) * 1000 << " ms" << endl;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; ++i) 
        {
            for (int j = 0; j < N; ++j) 
            {
                c[i][j] = 0;
            }
        }
    }

    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < num_threads; ++i)
        {
           multiply_P(i, N, num_threads); 
        }
    }
    end = omp_get_wtime();
    cout << "Parallel multiplication time: " << (end - start) * 1000 << " ms" << endl;

    // Clear result matrix c
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; ++i) 
        {
            for (int j = 0; j < N; ++j) 
            {
                c[i][j] = 0;
            }
        }
    }

    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < num_threads; ++i)
        {
           multiply_P_Transposed(i, N, num_threads); 
        }
    }
    end = omp_get_wtime();
    cout << "Parallel multiplication with transposed time: " << (end - start) * 1000 << " ms" << endl;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < N; ++i) 
        {
            for (int j = 0; j < N; ++j) 
            {
                c[i][j] = 0;
            }
        }
    }
    
    for (int i = 0; i < N; ++i) 
    {
        for (int j = 0; j < N; ++j) 
        {
            c[i][j] = 0;
        }
    }

    int tileSize = 32; 
    start = omp_get_wtime();
    multiplyWithTiling(N, tileSize);
    end = omp_get_wtime();
    cout << "Parallel multiplication with tiling: " << (end - start) * 1000 << " ms" << endl;

    deallocateMatrix(a, N);
    deallocateMatrix(b, N);
    deallocateMatrix(c, N);
    deallocateMatrix(bt, N);

    return 0;
}


