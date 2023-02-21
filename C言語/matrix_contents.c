#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<windows.h>

#define M 2/*�s��*/
#define N 2 /*��*/
//�e�Z���Ƀ����_���Œl��������B(�s��̃A�h���X������)
int random_cell(int* matrix_address, int m, int n, int min, int max) {
	int i, j;
	Sleep(1 * 1000);
	/*if (min == NULL) {
		min = 1;
	}
	if (max == NULL) {
		max = 10;
	}*/
	//���������ԂɊ�Â��ď�����
	srand((unsigned int)time(NULL));
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			//�e�Z���Ƀ����_���Ȓl����
			*matrix_address = rand() % (max-min)+min;
			//�A�h���X������炵�āA���̃Z���ւƕς���
			matrix_address = matrix_address + 1;
		}
	}
}
//�s��̐�����\������B(�s��̃A�h���X������)
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
//�s�񓯎m�̑����Z
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
//�s��̈����Z
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
//�s��̐�
int multi_matrix(int* matrix_address_A,int m, int n, int* matrix_address_B, int o, int p, int* matrix_address_C) {
	int *ap, *bp, i, j, k, l;
	//�A�h���X�̏����l��ۑ�
	ap = matrix_address_A;
	bp = matrix_address_B;
	//m x n x o x p = m x p�ɂ��邽�߁B�G���[�Ȃ牉�Z���Ȃ��B
	if (n != o) {
		printf("�^���������Ȃ�");
		return 0;
	}
	//������s��̐�����i��j�ł��炵�Ă����B
	for (i = 0; i < m; i++) {
		for (j = 0; j < p; j++) {
			//������s��̐�����0�Ƃ�������
			*matrix_address_C = 0;
			//�s��A��B�̃A�h���X����肢���ƕς��Ă��̒��g���v�Z����B
			//A�͍s�Ō��Ă���
			matrix_address_A = ap + (i * n);
			//B�͗�Ō��Ă���B
			matrix_address_B = bp + j;
			//������s��̐����̒��g�����߂�
			for (k = 0; k < n; k++) {
				*matrix_address_C = *matrix_address_A * *matrix_address_B + *matrix_address_C;
				matrix_address_A = matrix_address_A + 1;
				matrix_address_B = matrix_address_B + p;
			}
			//������s��̃Z����ׂɕς���
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