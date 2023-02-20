#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define M 3 /*行数*/
#define N 3 /*列数*/
//各セルにランダムで値を代入する。(行列のアドレスが引数)
int random_cell(int *matrix_address){
	int i, j;
	//乱数を時間に基づいて初期化
	srand((unsigned int)time(NULL));
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			//各セルにランダムな値を代入
			*matrix_address = rand() % 10;
			//アドレスを一つずらして、次のセルへと変える
			matrix_address = matrix_address + 1;
		}
	}
}
//行列の成分を表示する。(行列のアドレスが引数)
int show_the_matrix(int *matrix_address) {
	int i, j;
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			printf("%d ", *matrix_address);
			matrix_address = matrix_address + 1;
		}
		printf("\n");
	}

}

int main(void) {
	int matrix[M][N];
	random_cell(&matrix);
	show_the_matrix(&matrix);
	return 0;
}