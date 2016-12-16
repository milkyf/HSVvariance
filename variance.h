class variance
{
private:
	list<cv::Rect> m_cmp_rcs;
	char *m_mask_str;
	cv::Rect m_roi_rc;
	bool m_cmp;
	inline void calc_vrnc_map(const cv::Mat &src, const cv::Mat &sum, const cv::Mat &sqsum,
		const cv::Mat &mask, cv::Mat &dst);
	inline void addRect(cv::Rect rc);
	inline void detectRect(const cv::Mat &dst, const double threshold, const double wsize);
	int m_wsz, m_th;

public:
	variance(int w, int s);
	void kenshou(int64 time);
	void detect(const Mat &img, list<Rect> &rects);
	void draw(Mat &dst);
};
