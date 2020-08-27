#include<stdio.h>
#include<math.h>  
#include<time.h>
const double pi = acos(-1.0);
struct Sample{
	char c;
	int a[16];
}data[20000];
double norm_max(Sample&x, Sample&y){
	double s = 0;
	for (int i = 0; i < 16; i++){
		double now = abs(x.a[i] - y.a[i]);
		if (now>s)s = now;
	}
	return s;
}
double norm2(Sample&x, Sample&y){
	double s = 0;
	for (int i = 0; i < 16; i++){
		double t = x.a[i] - y.a[i];
		s += t*t;
	}
	return s;
}
double kernel_rectangle(double x, double h, int dim){
	return x / h <= 0.5 ? 1 / pow(h, dim) : 0;
}
double kernel_circle(double x, double h){
    //球形空间
	return x <= h ? 1 : 0;
}
double kernel_gauss(double x, double h){
	return exp(-x*x / (2 * h*h)) / sqrt(2 * pi) / h;
}
double kernel_zhishu(double x, double h){
	return exp(-x / h);
}
int main(){
	FILE *infile = fopen("letter-recognition.data", "r");
	if (infile == NULL){
		puts("缺少数据文件");
		return -1;
	}
	for (int i = 0; i < 20000; i++){
		fscanf(infile, "\n%c", &data[i].c);
		for (int j = 0; j < 16; j++){
			fscanf(infile, ",%d", &data[i].a[j]);
		}
	}
	int beg_time = time(0);
	int test_size = 4000, train_size = 2e4 - test_size;
	Sample*train = data, *test = &data[train_size];
	int right_cnt = 0;
	for (int i = 0; i < test_size; i++){
		double p[26] = { 0 };
		for (int j = 0; j < train_size; j++){
			//p[train[j].c - 'A'] += kernel_rectangle(norm2(test[i], train[j]),h,16);
			//p[train[j].c - 'A'] += kernel_circle(norm2(test[i], train[j]), 15);
			p[train[j].c - 'A'] += kernel_gauss(norm2(test[i], train[j]), 3.7);
			//高斯核，h=3.7，正确率96.2%
			//p[train[j].c - 'A'] += kernel_zhishu(norm2(test[i], train[j]), 1.5);
		}
		char ans = 'A';
		double ansP = p[0];
		for (int j = 1; j < 26; j++){
			if (p[j]>ansP){
				ansP = p[j];
				ans = j + 'A';
			}
		}
		if (ans == test[i].c){
			right_cnt++;
		}
	}
	int end_time = time(0);
	printf("耗时%d秒，正确率为%lf\n", end_time - beg_time, right_cnt*1.0 / test_size);
	return 0;
}