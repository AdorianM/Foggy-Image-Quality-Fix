// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include "guidedfilter.h"


void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}




//Functia asta este foarte buna. Nu trebuie nimic modificat la ea.
void getDarkChannel(Mat src, Mat* darkChannel)
{
	int startRow, endRow;
	int startCol, endCol;

	//Patratele generate pt dark channel au o raza anume. Raza calculata mai jos.
	double radius = 0;
	//Raza este numarul de linii/coloane * 0.02 rotunjit. Se alege minimul din cele doua.
	radius = darkChannel->rows < darkChannel->cols ? round(darkChannel->rows * 0.02) : round(darkChannel->cols * 0.02);

	printf("%lf\n", radius);

	for (int i = 0; i < darkChannel->rows; i++)
		for (int j = 0; j < darkChannel->cols; j++)
		{
			//Construiesc patrate de raza 'radius' pentru darkChannel (Ne asiguram si sa nu depaseasca limitele)
			startRow = i - radius;
			startRow = startRow > 0 ? startRow : 0;

			endRow = i + radius;
			endRow = endRow < darkChannel->rows ? endRow : (darkChannel->rows - 1);

			startCol = j - radius;
			startCol = startCol > 0 ? startCol : 0;

			endCol = j + radius;
			endCol = endCol < darkChannel->cols ? endCol : (darkChannel->cols - 1);

			int min = 256;
			int current = 0;

			//Parcurg patratele de raza 'radius' si le schimb valoarea in cea minima dintre cele 3 canale de culoare

			for (int iS = startRow; iS <= endRow; iS++)
				for (int jS = startCol; jS <= endCol; jS++)
				{
					for (int canal = 0; canal < 3; canal++)
					{
						current = src.at<Vec3b>(iS, jS)[canal];
						if (current < min)
							min = current;
					}

				}
			darkChannel->at<uchar>(i, j) = min;
		}
}

//Sunt 90% sigur ca functia asta e buna si nu mai trebuie schimbat nimic la ea
//(cu exceptia lucrului scris in comentariul de mai jos)
Vec3b getAtmosphericLightEstimation(Mat src, Mat darkChannel)
{
	int m = 0;
	//Ar trebui sa luam maximul dintre 0.1% din cei mai luminati pixeli.
	//M-am gandit ca n-are sens sa ii aflu pe cei 0.1%, ca maximu' ii cel global oricum.


	int brightestPixels[1000] = { 0 };
	int sizeBPx = darkChannel.rows * darkChannel.cols * 0.1 / 100;

	Vec3b aLight = Vec3b(0, 0, 0);
	for (int i = 0; i < darkChannel.rows; i++)
		for (int j = 0; j < darkChannel.cols; j++)
		{
			if (darkChannel.at<uchar>(i, j) > brightestPixels[sizeBPx - 1]) {
				for (int k = 0; k < sizeBPx - 1; k++)
					brightestPixels[k] = brightestPixels[k + 1];
				brightestPixels[sizeBPx - 1] = darkChannel.at<uchar>(i, j);
			}
				
		}

	for (int i = 0; i < darkChannel.rows; i++)
		for (int j = 0; j < darkChannel.cols; j++)
		{
			for(int k = 0; k < sizeBPx ;k++)
				if (darkChannel.at<uchar>(i, j) == brightestPixels[k] && m < darkChannel.at<uchar>(i, j))
				{
					m = darkChannel.at<uchar>(i, j);
					aLight = src.at<Vec3b>(i, j);
				}
		}
	return aLight;
}


uchar trPixel(uchar iPixel, uchar aLight, float w)
{
	float rez = 255 * (1 - w * abs((float)iPixel / aLight));
	return rez;
}

void TransmissionMapEstimation(Mat src, Vec3f aLight, Mat* tm)
{
	double const w = 0.85; 

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			tm->at<uchar>(i, j) = max(trPixel(src.at<Vec3b>(i, j)[0], aLight[0], w),
									max(trPixel(src.at<Vec3b>(i, j)[1], aLight[1], w),
										trPixel(src.at<Vec3b>(i, j)[2], aLight[2], w)));
		}
}

//La asta nu sunt sigur daca am aplicat algoritmul corect.
//Ideea e ca ar trebui aplicata formula (4) de la
//https://link.springer.com/article/10.1186/s13640-020-0493-9?fbclid=IwAR1QkOEHUl5GGggiwZ-rrrwoies6llYfOpAmm2G9Uhsgj7PWF8OZadGsVeA#Equ1
//Pentru o poza mai buna, Transmission Map Estimation-ul trebuie si el rafinat (Trefined).
//Tot ce am facut e explicat la link-ul de mai sus. (in text explica numele variabilelor din pseudocod, etc.)
void RefacereImagine(Mat src, Mat tm, Vec3f aLight, Mat* dst)
{
	//Normalizare aLight
	aLight[0] /= 255.0;
	aLight[1] /= 255.0;
	aLight[2] /= 255.0;

	int t0 = 0.1; //Threshold

	for (int i = 0; i < src.rows; i++)
		for (int j = 0; j < src.cols; j++)
		{
			double temp = tm.at<uchar>(i, j);
			temp /= 255.0; //Normalizare
			for (int canal = 0; canal < 3; canal++)
			{
				double srcCurent = src.at<Vec3b>(i, j)[canal];
				srcCurent /= 255.0; //normalizare

				double temp_refacere = ((srcCurent - aLight[canal]) / max(temp, t0)) + aLight[canal];
				temp_refacere *= 255.0; //Readucere la normal
				temp_refacere = temp_refacere > 255 ? 255 : (temp_refacere < 0 ? 0 : temp_refacere);

				dst->at<Vec3b>(i, j)[canal] = round(temp_refacere);
			}
		}
}

//Aici doar am apelat functiile pe rand. Merge, dar nu e rafinat si nu e calculat fog density.
void EliminareCeata()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		//Matricile in ordinea lor de calculare.
		Mat src = imread(fname);
		Mat darkChannel = Mat(src.rows, src.cols, CV_8UC1);
		Vec3f aLight = Vec3f(0, 0, 0);
		Mat transmissionMap = Mat(src.rows, src.cols, CV_8UC1);


		getDarkChannel(src, &darkChannel);
		aLight = getAtmosphericLightEstimation(src, darkChannel);
		TransmissionMapEstimation(src, aLight, &transmissionMap);
		std::cout << aLight << "\n";


		int r = 1; // try r=2, 4, or 8
		double eps = 0.1 * 0.1; // try eps=0.1^2, 0.2^2, 0.4^2

		eps *= 255 * 255;   // Because the intensity range of our images is [0, 255]

		Mat transmissionMapReffined = guidedFilter(transmissionMap, transmissionMap, r, eps);
		Mat dst = Mat(src.rows, src.cols, CV_8UC3);

		RefacereImagine(src, transmissionMapReffined, aLight, &dst);

		for (int i = 0; i < src.rows; i++)			
		{
			for (int j = 0; j < src.cols; j++)
			{
				for (int canal = 0; canal < 3; canal++)
				{
					double tmp = dst.at<Vec3b>(i, j)[canal];
					//Normalizare + parte intreaga.
					tmp = tmp / aLight[canal] * 255.0;

					tmp = tmp <= 255 ? tmp : 255; //Astea is doar if-uri. if(tmp <= 255) then tmp ramane la fel. Else tmp = 255.
					dst.at<Vec3b>(i, j)[canal] = round(tmp);
				}
			}
		}

		imshow("Sursa", src);
		imshow("Canal Intunecat", darkChannel);
		imshow("Transmission Map", transmissionMap);
		imshow("Transmission Map Reffined", transmissionMapReffined);
		imshow("Poza fara ceata", dst);

		waitKey();
	}

}


int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Open image\n");
		printf(" 2 - Open BMP images from folder\n");
		printf(" 3 - Image negative - diblook style\n");
		printf(" 4 - BGR->HSV\n");
		printf(" 5 - Resize image\n");
		printf(" 6 - Canny edge detection\n");
		printf(" 7 - Edges in a video sequence\n");
		printf(" 8 - Snap frame from live video\n");
		printf(" 9 - Mouse callback demo\n");
		printf(" 10 - Eliminare ceata\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testOpenImage();
				break;
			case 2:
				testOpenImagesFld();
				break;
			case 3:
				testParcurgereSimplaDiblookStyle(); //diblook style
				break;
			case 4:
				//testColor2Gray();
				testBGR2HSV();
				break;
			case 5:
				testResize();
				break;
			case 6:
				testCanny();
				break;
			case 7:
				testVideoSequence();
				break;
			case 8:
				testSnap();
				break;
			case 9:
				testMouseClick();
				break;
			case 10:
				EliminareCeata();
				break;
		}
	}
	while (op!=0);
	return 0;
}