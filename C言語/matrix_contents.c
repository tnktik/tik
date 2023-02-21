#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<windows.h>

#define M 2/*行数*/
#define N 2 /*列数*/
//各セルにランダムで値を代入する。(行列のアドレスが引数)
int random_cell(int* matrix_address, int m, int n, int min, int max) {
	int i, j;
	Sleep(1 * 1000);
	/*if (min == NULL) {
		min = 1;
	}
	if (max == NULL) {
		max = 10;
	}*/
	//乱数を時間に基づいて初期化
	srand((unsigned int)time(NULL));
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			//各セルにランダムな値を代入
			*matrix_address = rand() % (max-min)+min;
			//アドレスを一つずらして、次のセルへと変える
			matrix_address = matrix_address + 1;
		}
	}
}
//行列の成分を表示する。(行列のアドレスが引数)
int show_the_matrix(int* matrix_address, int m, int n) {
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%d ", *matrix_address);
			matrix_address = matrix_address + 1;
		}
		printf("\n");
	}
}
//行列同士の足し算
int plus_matrix(int* matrix_address_A, int* matrix_address_B, int* matrix_address_C, int m, int n) {
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			*matrix_address_C = *matrix_address_A + *matrix_address_B;
			matrix_address_A = matrix_address_A + 1;
			matrix_address_B = matrix_address_B + 1;
			matrix_address_C = matrix_address_C + 1;
		}
	}
}
//行列の引き算
int minus_matrix(int* matrix_address_A, int* matrix_address_B, int* matrix_address_C, int m, int n) {
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			*matrix_address_C = *matrix_address_A - *matrix_address_B;
			matrix_address_A = matrix_address_A + 1;
			matrix_address_B = matrix_address_B + 1;
			matrix_address_C = matrix_address_C + 1;
		}
	}
}
//行列の積
int multi_matrix(int* matrix_address_A,int m, int n, int* matrix_address_B, int o, int p, int* matrix_address_C) {
	int *ap, *bp, i, j, k, l;
	//アドレスの初期値を保存
	ap = matrix_address_A;
	bp = matrix_address_B;
	//m x n x o x p = m x pにするため。エラーなら演算しない。
	if (n != o) {
		printf("型が正しくない");
		return 0;
	}
	//得られる行列の成分をiとjでずらしていく。
	for (i = 0; i < m; i++) {
		for (j = 0; j < p; j++) {
			//得られる行列の成分を0とし初期化
			*matrix_address_C = 0;
			//行列AとBのアドレスを上手いこと変えてその中身を計算する。
			//Aは行で見ている
			matrix_address_A = ap + (i * n);
			//Bは列で見ている。
			matrix_address_B = bp + j;
			//得られる行列の成分の中身を求める
			for (k = 0; k < n; k++) {
				*matrix_address_C = *matrix_address_A * *matrix_address_B + *matrix_address_C;
				matrix_address_A = matrix_address_A + 1;
				matrix_address_B = matrix_address_B + p;
			}
			//得られる行列のセルを隣に変える
			matrix_address_C = matrix_address_C + 1;
		}
	}
}



int main(void) {
	int matrixA[M][N];
	int matrixB[M][N];
	int matrixC[M][N];
	random_cell(&matrixA, M, N,0,10);
	show_the_matrix(&matrixA, M, N);
	printf("\n");
	random_cell(&matrixB, M, N,0,10);
	show_the_matrix(&matrixB, M, N);
	printf("\n");
	plus_matrix(&matrixA, &matrixB, &matrixC, M, N);
	show_the_matrix(&matrixC, M, N);
	printf("\n");
	minus_matrix(&matrixA, &matrixB, &matrixC, M, N);
	show_the_matrix(&matrixC, M, N);
	printf("\n");
	multi_matrix(&matrixA, M, N, &matrixB, M, N, &matrixC);
	show_the_matrix(&matrixC, M, N);

	return 0;
}