#define _CRT_SECURE_NO_WARNINGS 1

#include <iostream>
#include <vector>
#include <list>
#include <opencv2/opencv.hpp>
#include <tchar.h>
#include <fstream>


using namespace std;
using namespace cv;
#include "variance.h"
#if _DEBUG
#pragma comment(lib, "opencv_calib3d248d.lib")
#pragma comment(lib, "opencv_contrib248d.lib")
#pragma comment(lib, "opencv_core248d.lib")
#pragma comment(lib, "opencv_features2d248d.lib")
#pragma comment(lib, "opencv_flann248d.lib")
#pragma comment(lib, "opencv_gpu248d.lib")
#pragma comment(lib, "opencv_highgui248d.lib")
#pragma comment(lib, "opencv_imgproc248d.lib")
#pragma comment(lib, "opencv_legacy248d.lib")
#pragma comment(lib, "opencv_ml248d.lib")
#pragma comment(lib, "opencv_nonfree248d.lib")
#pragma comment(lib, "opencv_objdetect248d.lib")
#pragma comment(lib, "opencv_ocl248d.lib")
#pragma comment(lib, "opencv_photo248d.lib")
#pragma comment(lib, "opencv_stitching248d.lib")
#pragma comment(lib, "opencv_superres248d.lib")
#pragma comment(lib, "opencv_ts248d.lib")
#pragma comment(lib, "opencv_video248d.lib")
#pragma comment(lib, "opencv_videostab248d.lib")
#else
#pragma comment(lib, "opencv_calib3d248.lib")
#pragma comment(lib, "opencv_contrib248.lib")
#pragma comment(lib, "opencv_core248.lib")
#pragma comment(lib, "opencv_features2d248.lib")
#pragma comment(lib, "opencv_flann248.lib")
#pragma comment(lib, "opencv_gpu248.lib")
#pragma comment(lib, "opencv_highgui248.lib")
#pragma comment(lib, "opencv_imgproc248.lib")
#pragma comment(lib, "opencv_legacy248.lib")
#pragma comment(lib, "opencv_ml248.lib")
#pragma comment(lib, "opencv_nonfree248.lib")
#pragma comment(lib, "opencv_objdetect248.lib")
#pragma comment(lib, "opencv_ocl248.lib")
#pragma comment(lib, "opencv_photo248.lib")
#pragma comment(lib, "opencv_stitching248.lib")
#pragma comment(lib, "opencv_superres248.lib")
#pragma comment(lib, "opencv_ts248.lib")
#pragma comment(lib, "opencv_video248.lib")
#pragma comment(lib, "opencv_videostab248.lib")
#endif


variance::variance(int w, int s) :
m_wsz(w), m_th(s),
m_roi_rc(0, 0, -1, -1), m_cmp(true)
{}


inline void variance::calc_vrnc_map(const cv::Mat &src, const cv::Mat &sum,
	const cv::Mat &sqsum, const cv::Mat  &mask, cv::Mat &dst)
{
	cout << "Entering calc_vrnc_map" << endl;
	dst = cv::Mat::zeros(src.rows, src.cols, CV_64F);
	const double  isqwsz = 1 / (double)(m_wsz*m_wsz);
	const int xmax = src.cols - m_wsz;
	// const int ymax = src.rows - m_wsz + 1 - src.rows/2;
	const int ymax = src.rows - m_wsz;
	for (int y = 0; y < ymax; ++y){
		double *pdst = dst.ptr<double>(y);
		const int ywsz = y + m_wsz;
		const int *psum0 = sum.ptr<int>(y);
		const int *psum1 = sum.ptr<int>(ywsz);
		const double *psqsum0 = sqsum.ptr<double>(y);
		const double *psqsum1 = sqsum.ptr<double>(ywsz);
		const uchar *pmask = mask.ptr<uchar>(ywsz);
		for (int x = 0; x < xmax; ++x){
			const int xwsz = x + m_wsz;
			const double ave_sq = (psum1[xwsz] - psum1[x] - psum0[xwsz] + psum0[x])*isqwsz;
			pdst[x] = (psqsum1[xwsz] - psqsum1[x] - psqsum0[xwsz] + psqsum0[x])*isqwsz - ave_sq*ave_sq;
			pdst[x] *= pmask[x];
		}
	}
	cout << "Exiting vrnc_map " << endl;
}

inline void variance::addRect(cv::Rect rc)
{
	if (m_cmp_rcs.size() == 0){
		m_cmp_rcs.push_back(rc);
		return;
	}

	list<cv::Rect>::iterator it = m_cmp_rcs.begin();
	for (; !(it == m_cmp_rcs.end());){
		cv::Rect new_rc;
		new_rc.x = rc.x < it->x ? rc.x : it->x;

		const int rc0_right = rc.x + rc.width;
		const int rc1_right = it->x + it->width;
		new_rc.width = rc0_right > rc1_right ? rc0_right - new_rc.x : rc1_right - new_rc.x;

		if (new_rc.width > rc.width + it->width){
			++it;
			continue;
		}
		new_rc.y = rc.y < it->y ? rc.y : it->y;

		const int rc0_upper = rc.y + rc.height;
		const int rc1_upper = it->y + it->height;
		new_rc.height = rc0_upper > rc1_upper ? rc0_upper - new_rc.y : rc1_upper - new_rc.y;

		if (new_rc.height > rc.height + it->height){
			++it;
			continue;
		}

		m_cmp_rcs.erase(it);
		rc = new_rc;
		it = m_cmp_rcs.begin();
	}
	m_cmp_rcs.push_back(rc);
}

inline void variance::detectRect(const cv::Mat &dst, const double th,
	const double wsz)
{
	const double *pdst = dst.ptr<double>(0);
	for (int y = 0; y < dst.rows; ++y){
		for (int x = 0; x < dst.cols; ++x){
			if (pdst[x] > th){
				m_cmp_rcs.push_back(cv::Rect(x, y, (int)wsz, (int)wsz));
			}
		}
	}
}

void variance::detect(const Mat &src_img, list<Rect> &rects){
	Mat hsv_img;
	cvtColor(src_img, hsv_img, CV_BGR2HSV);
	vector<Mat> planes;
	split(hsv_img, planes);
	Mat img;
	img = planes[0];
	Mat mask = Mat(img.size(), CV_8U, Scalar(1));
	cv::Mat sum, sqsum, var;
	cv::integral(img, sum, sqsum, CV_32S);
	calc_vrnc_map(img, sum, sqsum, mask, var);
	if (m_cmp){
		for (int y = 0; y < var.rows / 3; ++y){
			const double *pvar = var.ptr<double>(y);
			for (int x = 0; x < var.cols; ++x){
				if (pvar[x] > m_th)
					addRect(cv::Rect(x, y, m_wsz, m_wsz));
			}
		}
	}
	else{
		for (int y = 0; y < var.rows / 2; ++y){
			const double *pvar = var.ptr<double>(y);
			for (int x = 0; x < var.cols; ++x){
				if (pvar[x] > m_th)
					m_cmp_rcs.push_back(cv::Rect(x, y, m_wsz, m_wsz));
			}
		}
	}

	rects = m_cmp_rcs;
}

void variance::draw(Mat &dst){
	cout << "m_cmp_rcs: "<< m_cmp_rcs.size() << endl;
	for (list<cv::Rect>::iterator it = m_cmp_rcs.begin();
		!(it == m_cmp_rcs.end()); ++it) {
		rectangle(dst, *it, cv::Scalar(0, 255, 0));
	}
}
void variance::kenshou(int dsc, int s, int win, Mat &dst){//検証
	FILE *kekka;
	kekka = fopen("Result.csv", "a");
	
	Rect awaseta;
	Rect zissai[500];
	Rect zissai2[500];
	char str[500];
	//string dsc;
	sprintf(str, "shashin/DSC00%03d.jpg", dsc);

	Mat img = imread(str);

	char fname[500];
	
	
		sprintf(fname, "rtat/DSC00%03d.txt", dsc);
		ifstream ifs(fname);
		string strr;

		
		int a = 0, b = 0, rtat = 0;

		if (ifs.fail()){ 
			cerr << "File does not exist." << endl;
			exit(0);
		}
		int p = 0;
		while (getline(ifs, strr)){
			
			//テキストデータから船舶の座標を読み取り、船舶の数をカウント。pは行数
			sscanf(strr.data(), "%d %d", &a, &b);
			switch ((p + 1) % 2){
			case 1://1行目 左上の(x,y)座標
				atai[rtat].x = a;
				atai[rtat].y = b;
				p++;
				
				continue;

			default://2行目 右下の(x,y)座標
				atai[rtat].width = a - atai[rtat].x;
				atai[rtat].height = b - atai[rtat].y;
				rtat++, p++;
				continue;
			}
			
		}

		ifs.close();
		double rectSum = 0;
		for (int n = 0; n < rtat;n++){
			rectSum += atai[n].width * atai[n].height;//正しい船舶の面積
			
			
		}
		//printf("rectSum: %d\n", rectSum);
		int rectArea = 0;
		Rect rc;
		for (list<cv::Rect>::iterator it = m_cmp_rcs.begin();
			!(it == m_cmp_rcs.end()); ++it) {
			rc = *it;
			cout << rc.area() << " " << rc.size() << endl;////rc.area()に面積
			rectArea += rc.area();
			
		}
		//cout << rectSum << endl;
		double areaPercentage = rectSum / rectArea;
		//printf(" areaPercentage: %f\n", areaPercentage);

		//cout << m_cmp_rcs.size() << endl;
		
			
		int shogo = 0;

		for (int m = 0; m < rtat; m++){

			rectangle(dst, atai[m], cv::Scalar(0, 0, 255));

			for (list<cv::Rect>::iterator it = m_cmp_rcs.begin();
				!(it == m_cmp_rcs.end()); ++it) {
				
				rc = *it;
				cout << "実際の値" << atai[m].x << "," << atai[m].y << "," <<atai[m].width << "," <<atai[m].height << endl;
				cout << "計算" << rc.x << "," << rc.y << "," << rc.width << "," << rc.height << endl;
				awaseta = atai[m] & rc;
				cout << awaseta.x << "," << awaseta.y << "," << awaseta.width << "," << awaseta.height << endl;
				if (awaseta.area() > atai[m].area() / 2 && (rc.area() - awaseta.area()) < awaseta.area()*0.5){
					shogo++;
					break;
				}
			}
		}
		cout << shogo << endl;
		double kenshutsu = (double)shogo / (double)rtat;
		double gokenshutsu = ((double)m_cmp_rcs.size() - (double)shogo) / (double)m_cmp_rcs.size();
		fprintf(kekka, "%d,%d,%d,%f,%d,%d,%d,%d,%f,%f", dsc, s, win, rectSum, m_cmp_rcs.size(), rtat, rectArea, areaPercentage, kenshutsu, gokenshutsu);
		fprintf(kekka, "\n");

		fclose(kekka);
}

int main(int argc, char ** argv){
	FILE *TIME,*kekka;
	kekka = fopen("Result.csv", "w");
	TIME = fopen("time.csv", "w");
	double areaPercentage[500];	
	TickMeter time;
	fprintf(kekka, "画像番号,閾値,ウィンドウサイズ,正しい船舶の面積,プログラムが検出した船舶数,正しい船舶数,窓の総面積,抽出面積の割合,検出率,誤検出率\n");
	fclose(kekka);
	fprintf(TIME, "画像番号,検測時間\n");
	for (int dsc = 2; dsc <= 968; dsc++)//968
	{
		
		for (int win = 5; win < 50; win += 2){//ウィンドウサイズ
			for (int s = 10; s < 50; s += 1){//閾値
				
				char result[500];
				sprintf_s(result, "sikiiti/s%d_w%d_DSC00%03d.JPG",s,win, dsc);
				//imwrite(result, img);
				char str[500];
				sprintf_s(str, "shashin/DSC00%03d.jpg", dsc);
				Mat img = imread(str);
				if (img.empty()) continue;
				//resize(img, img, Size(640, 480));
				variance var(win, s);

				list<Rect> rects;
				
				areaPercentage[dsc];// = kenshou(dsc);

				time.start();//計測開始
				var.detect(img, rects);
				var.draw(img);
				time.stop();//計測終了
				var.kenshou(dsc,s,win, img);
				double jikan = time.getTimeMilli()/1000;//sec
				
				fprintf(TIME,"%d,%f\n",dsc,jikan);
				time.reset();
				imwrite(result, img);
				
			}
		}
	}
	fclose(TIME);
	return 0;
}
