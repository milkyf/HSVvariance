class variance
{
private:
	//list<cv::Rect> m_cmp_rcs;
	char *m_mask_str;
	cv::Rect m_roi_rc;
	bool m_cmp;
	inline void calc_vrnc_map(const cv::Mat &src, const cv::Mat &sum, const cv::Mat &sqsum,
		const cv::Mat &mask, cv::Mat &dst);
	inline void addRect(cv::Rect rc);
	inline void detectRect(const cv::Mat &dst, const double threshold, const double wsize);
	int m_wsz, m_th;

public:
	list<cv::Rect> m_cmp_rcs;
	int s, win;
	Rect awaseta;
	Rect zissai[500];
	Rect zissai2[500];
	TickMeter time;
	int64 timeCalc();
	int rectArea;
	Rect atai[500];
	
	int rtat;
	double areaPercentage[500];


	variance(int w, int s);
	void kenshou(int dsc, int s, int win, Mat &dst);
	void detect(const Mat &img, list<Rect> &rects);
	void draw(Mat &dst);


};
