#include<stdio.h>
#include<stdlib.h>
#include<time.h>

#define M 3 /*�s��*/
#define N 3 /*��*/
//�e�Z���Ƀ����_���Œl��������B(�s��̃A�h���X������)
int random_cell(int *matrix_address){
	int i, j;
	//���������ԂɊ�Â��ď�����
	srand((unsigned int)time(NULL));
	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			//�e�Z���Ƀ����_���Ȓl����
			*matrix_address = rand() % 10;
			//�A�h���X������炵�āA���̃Z���ւƕς���
			matrix_address = matrix_address + 1;
		}
	}
}
//�s��̐�����\������B(�s��̃A�h���X������)
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