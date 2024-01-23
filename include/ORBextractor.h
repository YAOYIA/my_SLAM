
#ifndef ORBEXTRACTOR_H
#define ORBEXTRACTOR_H


#include <vector>
#include <list>
#include <opencv/cv.h>

//主要实现ORB特征点的提取以及数目的分配功能

namespace ORB_SLAM2
{

class ExtractorNode
{

public:
    ExtractorNode():bNoMore(false){}

    void DivideNode(ExtractorNode &n1,ExtractorNode &n2,ExtractorNode &n3,ExtractorNode &n4);
    
    //保存当前节点的特征点
    std::vector<cv::KeyPoint> vKeys;
    //当前节点所对应的图像坐标的边界
    cv::Point2i UL,UR,BL,BR;

    //存储提取器节点的列表（双向列表）的一个迭代器
    //这个迭代器的提供了访问总结点列表的方式
    std::list<ExtractorNode>::iterator lit;

    //如果节点中只有一个特征点的话，说明这个节点不可以再分裂，这是一个标志位
    //这个节点中如果没有了特征点，那么这个节点就会被删除
    bool bNoMore;

};




class ORBextractor
{

public:

    //定义一个枚举类型用于表示是HARRIS_SCORE响应值还是FAST响应值
    enum {HARRIS_SCORE=0,FAST_SCORE=1};


    ORBextractor(int nfeatures,float scaleFactor ,int nlevels,int iniThFAST,int minTHFAST);
    ~ORBextractor(){}


    void operator()(cv::InputArray image,cv::InputArray mask,std::vector<cv::KeyPoint>& keypoints,cv::OutputArray descriptors);


    //返回图像金字塔的层数
    int inline GetLevels(){
        return nlevels;
    }

    //获取当前提取器所在图像的缩放因子，这个不带s的因子表示是想临近层之间的
    float inline GetScaleFactor(){
        return scaleFactor;
    }

    //获取图像相较于底层图像的缩放因子
    std::vector<float> inline GetScaleFactors(){
        return mvScaleFactor;
    }

   /**
     * @brief 获取上面的那个缩放因子s的倒数
     * @return std::vector<float> 倒数
     */
    std::vector<float> inline GetInverseScaleFactors(){
        return mvInvScaleFactor;
    }
    
    /**
     * @brief 获取sigma^2，就是每层图像相对于初始图像缩放因子的平方，参考cpp文件中类构造函数的操作
     * @return std::vector<float> sigma^2
     */
    std::vector<float> inline GetScaleSigmaSquares(){
        return mvLevelSigma2;
    }

    /**
     * @brief 获取上面sigma平方的倒数
     * @return std::vector<float> 
     */
    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return mvInvLevelSigma2;
    }

    //用于存储图像金子塔的变量，一个元素存储一个图像
    std::vector<cv::Mat> mvImagePyramid;


protected:

    /**
     * @brief 针对给出的图像，计算图像的金字塔
     * @param[in] image 
    */
    void ComputerPyramid(cv::Mat image);

    /**
     * @brief 以八叉树分配特征点的方式，计算图像的金字塔中的特征点
     * @details 这里两层vector表示，第一个表示图像中的所有特征点，第二层表示存储图像金字塔中所有图像的vector of keypoints
     * @param[out] allpoints 提取到的所有特征点
    */
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint>> & allKeypoints);

    /**
     * @brief 对于某一图层，分配其特征点，通过八叉树的方式
     * @param[in] vToDistributeKeys 等待分发的特征点
    */
    std::vector<cv::KeyPoint> DistributeOctTree(
        const std::vector<cv::KeyPoint> &vToDistributeKeys,const int &minX,const int &maxX,const int &minY,const int &maxY,
        const int &nFeatures,const int &level 
    );

    /**
     * @brief  比较老的办法，提取并平均特征点的方法
    */
   void ComputeKeyPOintsold(std::vector<std::vector<cv::KeyPoint>> & allkeypoints);

   std::vector<cv::Point> pattern;//用于计算描述自的随机采样点的集合

   int nfeatures;               //整个金字塔中，要提取到的特征点的数目
   double scaleFactor;          //图像金字塔层与层之间的缩放因子
   int nlevels;                 //图像金字塔的层数
   int iniThFAST;               //初始的FAST响应值阈值
   int minThFAST;               //最小的FAST响应值阈值


   std::vector<int> mvFeaturesPerLevel;  //分配到每层图像中，要提取的特征点的数目

   std::vector<int> umax;       //计算特整点的方向时，要有一个圆形区域，这个vetcor中存储了每一行的边界

   std::vector<float> mvScaleFactor; //每层图像的缩放因子
   std::vector<float> mvInvScaleFactor;   //，每层缩放因子的倒数
   std::vector<float> mvLevelSigma2;     //存储每层的sigam^2，即上面每层图像相对于底层图像缩放的倍数的平方
   std::vector<float> mvInvLevelSigma2;   //倒数



};







}//namespace ORB_SLAM







#endif