#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iterator>

#include "include/ORBextractor.h"
#include <iostream>

using namespace cv;
using namespace std;

namespace ORB_SLAM2
{

const int PATCH_SIZE=31;
const int HALF_PATCH_SIZE=15;
const int EDGE_THRESHOLD=19;

//灰度质心法
static float IC_Angle(const Mat &image,Point2f pt,const vector<int> &u_max)
{
    int m_01=0,m_10=0;
    const uchar* center=&image.at<uchar>(cvRound(pt.x),cvRound(pt.y));

    int step = (int)image.step1();

    for(int u;u>-HALF_PATCH_SIZE && u<HALF_PATCH_SIZE;++u){
        m_10+=u*center[u];
    }

    for(int v=1;v<HALF_PATCH_SIZE;++v){
        int v_sum=0;
        int d=u_max[v];

        for(int u;u<d;++d){
            v_sum+=(center[d+v*step]-center[d-v*step]);
            m_10=u*(center[d+v*step]+center[d-v*step]);
        }

        m_01+=v*v_sum;
    }

    return fastAtan2((float)m_01,(float)m_10);

}


//乘数因子，一个弧度对应的多少弧度
const float factorPI=(float)(CV_PI/180.f);

//就散orb特征点的描述子
static void computerOrbDescriptor(const KeyPoint& kpt,const Mat &image,const Point* pattern,uchar* desc){
    float angle=(float)kpt.angle*factorPI;
    float a = (float)cos(angle);
    float b = (float)sin(angle);

    int step=(int)image.step1();
    const uchar* center=&image.at<uchar>(cvRound(kpt.pt.x),cvRound(kpt.pt.y));

    #define GET_VALUE(idx) center[cvRound(pattern[idx].x*b+pattern[idx].y*a)*step+cvRound(pattern[idx].x*a-cvRound(pattern[idx].y*b))]

    for(int i=0;i<32;++i,pattern+=16){
        int t0,t1,val;
        t0=GET_VALUE(0);t1=GET_VALUE(1);
        val=t0<t1;
        t0=GET_VALUE(2);t1=GET_VALUE(3);
        val |=(t0<t1) <<1 ;
        t0=GET_VALUE(4);t1=GET_VALUE(5);
        val |=(t0<t1) <<2 ;
        t0=GET_VALUE(6);t1=GET_VALUE(7);
        val |=(t0<t1) <<3 ;
        t0=GET_VALUE(8);t1=GET_VALUE(9);
        val |=(t0<t1) <<4 ;
        t0=GET_VALUE(10);t1=GET_VALUE(11);
        val |=(t0<t1) <<5 ;
        t0=GET_VALUE(12);t1=GET_VALUE(13);
        val |=(t0<t1) <<6 ;
        t0=GET_VALUE(14);t1=GET_VALUE(15);
        val |=(t0<t1) <<7 ;

        desc[i]=(uchar)val;

    }

    #undef GET_VALUE

}


//下面就是预先定义好的随机点集，256是指可以提取出256bit的描述子信息，
//每个bit由一对点比较得来；4=2*2，前面的2是需要两个点（一对点）进行比较，
//后面的2是一个点有两个坐标
static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,				
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};


ORBextractor::ORBextractor(
    int _nfeatures,
    float _scaleFactor,
    int _nlevels,
    int _iniThFAST,
    int _minThFAST):
    nfeatures(_nfeatures),scaleFactor(_scaleFactor),nlevels(_nlevels),
    iniThFAST(_iniThFAST),minThFAST(_minThFAST)
{
    //存储每层图像缩放系数的vector调整为符合图像数目的大小
    mvScaleFactor.resize(nlevels);
    //
    mvLevelSigma2.resize(nlevels);

    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;

    for (size_t i = 1; i < nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*_scaleFactor;
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];

    }

    for (size_t i = 0; i < nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }
    
    mvImagePyramid.resize(nlevels);
    mvFeaturesPerLevel.resize(nlevels);

    //图像降采样的缩放系数的倒数
    float factor=1.0f/scaleFactor;

    //第0层的图像对应分配特征带你的个数
    float nDesiredFeaturesPerScale = nfeatures*(1-factor)/(1-(float)((double)factor*(
    double)factor,(double)nlevels));

    int sumFeatures=0;

    for (size_t i = 0; i < nlevels-1; i++)
    {
        mvFeaturesPerLevel[i]= cvRound(nDesiredFeaturesPerScale);
        sumFeatures+=nDesiredFeaturesPerScale;
        nDesiredFeaturesPerScale=nDesiredFeaturesPerScale*factor;
    }

    mvFeaturesPerLevel[nlevels]=std::max(nfeatures-sumFeatures,0);

    //成员变量pattern的长度，也就是点的个数，这里点的个数512表示512个点
    const int npoints=512;
    //获取用于计算BRIEF描述子的随机采样点点集头指针
    //注意到pattern0数据类型为Point*,bit_pattern_31_是int型，所以这里要进行强制类型转换
    const Point* pattern0 = (const Point*)bit_pattern_31_;
    //back_inserter的作用是可以快速的覆盖掉之前这个容器的pattern之前的数据
    //这里操作，是在全局变量区域的，int格式的随机采样点以cv::point格式复制到当前类对象中的成员变量中
    std::copy(pattern0,pattern0+npoints,std::back_inserter(pattern));


    //接下来的内容与计算旋转有关
    umax.resize(HALF_PATCH_SIZE+1);

    //计算圆的最大行号，+1应该是把中间行也会考虑进去
    int v,v0,vmax=cvFloor(HALF_PATCH_SIZE*sqrt(2.f)/2+1);//cvFloor用于浮点数向下取整


    int vmin=cvCeil(HALF_PATCH_SIZE*sqrt(2.f)/2);//向上取整
    //半径的平方
    const double hp2=HALF_PATCH_SIZE*HALF_PATCH_SIZE;

    //利用圆的方程计算每行像素的u坐标边界max
    for (v = 0; v <=vmax; ++v)
    {
        umax[v]=cvRound(sqrt(hp2-v*v));//结果都是大于0的结果，表示x坐标在这一行的边界
    }
    //这里其实是使用了对称的方式计算四分之一圆上的umax，目的是为了保持严格的对称
    //因为按照常规的做法cvRound会出现不对称的情况
    //同时随机采样的特征带你也不能够满足旋转之后的采样不变性
    for(v=HALF_PATCH_SIZE,v0=0;v>=vmin;--v){
        while (umax[v0]==umax[v0+1])
        {
            ++v0;
        }
        umax[v]=v0;
        ++v0;    

    }

}

static void computeOrientation(const Mat& image,std::vector<KeyPoint> &keypoints,const std::vector<int> &umax){
    //遍历所有的特征点，为特征点添加方向信息
    for (std::vector<KeyPoint>::iterator KeyPoint=keypoints.begin();KeyPoint!=keypoints.end();++KeyPoint)
    {
        KeyPoint->angle=IC_Angle(image,KeyPoint->pt,umax);
        //KeyPoint->pt表示特征点在图像中的坐标
    }
    
}

/**
 * @brief 将提取器节点分成4个节点，同时完成图像区域的划分、特征点归属的划分，以及相关标志位的置位
*/
void ExtractorNode::DivideNode(ExtractorNode &n1,ExtractorNode &n2,ExtractorNode &n3,ExtractorNode &n4){
    //得到提取器结果的一般宽度
    const int halfX=ceil(static_cast<float>(UR.x-UL.x)/2);
    //得到提取器节点的一般高度
    const int halfY=ceil(static_cast<float>(BR.y-UR.y)/2);

    //n1的区域
    n1.UL=UL;
    n1.UR=cv::Point2i(UL.x+halfX,UL.y);
    n1.BL=cv::Point2i(UL.x,UL.y+halfY);
    n1.BR=cv::Point2i(UL.x+halfX,UL.y+halfY);

    //用来存储在该节点对应的图像网格中提取出来的特征点的vector
    n1.vKeys.reserve(vKeys.size());

    //n2
    n2.UL=n1.UR;
    n2.UR=UR;
    n2.BL=n1.BR;
    n2.BR=cv::Point2i(UL.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    //n3 存储左下区域的边界
    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    //n4 存储右下区域的边界
    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //遍历提取器节点的vkeys中存储的特征点,并将特征点放到对应的提取器节点内
    for(size_t i=0;i<vKeys.size();++i){
        const cv::KeyPoint &kp=vKeys[i];

        if(kp.pt.x<n1.UR.x){
            if(kp.pt.y<n1.UL.y){
                n1.vKeys.push_back(kp);
            }
            else{
                n3.vKeys.push_back(kp);
            }
        }else{
            if(kp.pt.y<n1.UR.y){
                n2.vKeys.push_back(kp);
            }else{
                n4.vKeys.push_back(kp);
            }
        }
    }
    //判断每个特征点提取器节点所在的图像中的特征带你数目（就是分配给子节点的特征点数目），然后做标记
    //这里判断是否数目等于1的目的是确定这个节点是否可以继续向下分裂
    if(n1.vKeys.size()==1){
        n1.bNoMore=true;
    }
    if(n2.vKeys.size()==1){
        n2.bNoMore=true;
    }
    if(n3.vKeys.size()==1){
        n3.bNoMore=true;
    }
    if(n4.vKeys.size()==1){
        n4.bNoMore=true;
    }

}
/**
 * @brief 
 * @param N  希望提出的特征点的个数
 * @param level 制定金字塔图层，但是并未使用
*/
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(const vector<cv::KeyPoint> &vToDistributeKeys,const int &minX,const int &maxX,const int &minY,const int &maxY,const int &N,const int &level)
{
    // Compute how many initial nodes
    // Step 1 根据宽高比确定初始节点数目
	//计算应该生成的初始节点个数，根节点的数量nIni是根据边界的宽高比值确定的，一般是1或者2
    // ! bug: 如果宽高比小于0.5，nIni=0, 后面hx会报错
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

	//一个初始的节点的x方向有多少个像素
    const float hX = static_cast<float>(maxX-minX)/nIni;

	//存储有提取器节点的链表
    list<ExtractorNode> lNodes;

	//存储初始提取器节点指针的vector
    vector<ExtractorNode*> vpIniNodes;

	//重新设置其大小
    vpIniNodes.resize(nIni);

	// Step 2 生成初始提取器节点
    for(int i=0; i<nIni; i++)
    {      
		//生成一个提取器节点
        ExtractorNode ni;

		//设置提取器节点的图像边界
		//注意这里和提取FAST角点区域相同，都是“半径扩充图像”，特征点坐标从0 开始 
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);    //UpLeft
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);  //UpRight
		ni.BL = cv::Point2i(ni.UL.x,maxY-minY);		        //BottomLeft
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);             //BottomRight

		//重设vkeys大小
        ni.vKeys.reserve(vToDistributeKeys.size());

		//将刚才生成的提取节点添加到链表中
		//虽然这里的ni是局部变量，但是由于这里的push_back()是拷贝参数的内容到一个新的对象中然后再添加到列表中
		//所以当本函数退出之后这里的内存不会成为“野指针”
        lNodes.push_back(ni);
		//存储这个初始的提取器节点句柄
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    // Step 3 将特征点分配到子提取器节点中
    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
		//获取这个特征点对象
        const cv::KeyPoint &kp = vToDistributeKeys[i];
		//按特征点的横轴位置，分配给属于那个图像区域的提取器节点（最初的提取器节点）
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }
    
	// Step 4 遍历此提取器节点列表，标记那些不可再分裂的节点，删除那些没有分配到特征点的节点
    // ? 这个步骤是必要的吗？感觉可以省略，通过判断nIni个数和vKeys.size() 就可以吧
    list<ExtractorNode>::iterator lit = lNodes.begin();
    while(lit!=lNodes.end())
    {
		//如果初始的提取器节点所分配到的特征点个数为1
        if(lit->vKeys.size()==1)
        {
			//那么就标志位置位，表示此节点不可再分
            lit->bNoMore=true;
			//更新迭代器
            lit++;
        }
        ///如果一个提取器节点没有被分配到特征点，那么就从列表中直接删除它
        else if(lit->vKeys.empty())
            //注意，由于是直接删除了它，所以这里的迭代器没有必要更新；否则反而会造成跳过元素的情况
            lit = lNodes.erase(lit);			
        else
			//如果上面的这些情况和当前的特征点提取器节点无关，那么就只是更新迭代器 
            lit++;
    }

    //结束标志位清空
    bool bFinish = false;

	//记录迭代次数，只是记录，并未起到作用
    int iteration = 0;

	//声明一个vector用于存储节点的vSize和句柄对
	//这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数目和其句柄
    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;

	//调整大小，这里的意思是一个初始化节点将“分裂”成为四个
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    // Step 5 利用四叉树方法对图像进行划分区域，均匀分配特征点
    while(!bFinish)
    {
		//更新迭代次数计数器，只是记录，并未起到作用
        iteration++;

		//保存当前节点个数，prev在这里理解为“保留”比较好
        int prevSize = lNodes.size();

		//重新定位迭代器指向列表头部
        lit = lNodes.begin();

		//需要展开的节点计数，这个一直保持累计，不清零
        int nToExpand = 0;

		//因为是在循环中，前面的循环体中可能污染了这个变量，所以清空
		//这个变量也只是统计了某一个循环中的点
		//这个变量记录了在一次分裂循环中，那些可以再继续进行分裂的节点中包含的特征点数目和其句柄
        vSizeAndPointerToNode.clear();

        // 将目前的子区域进行划分
		//开始遍历列表中所有的提取器节点，并进行分解或者保留
        while(lit!=lNodes.end())
        {
			//如果提取器节点只有一个特征点，
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
				//那么就没有必要再进行细分了
                lit++;
				//跳过当前节点，继续下一个
                continue;
            }
            else
            {
                // If more than one point, subdivide
				//如果当前的提取器节点具有超过一个的特征点，那么就要进行继续分裂
                ExtractorNode n1,n2,n3,n4;

				//再细分成四个子区域
                lit->DivideNode(n1,n2,n3,n4); 

                // Add childs if they contain points
				//如果这里分出来的子区域中有特征点，那么就将这个子区域的节点添加到提取器节点的列表中
				//注意这里的条件是，有特征点即可
                if(n1.vKeys.size()>0)
                {
					//注意这里也是添加到列表前面的
                    lNodes.push_front(n1);   

					//再判断其中子提取器节点中的特征点数目是否大于1
                    if(n1.vKeys.size()>1)
                    {
						//如果有超过一个的特征点，那么待展开的节点计数加1
                        nToExpand++;

						//保存这个特征点数目和节点指针的信息
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));

						//?这个访问用的句柄貌似并没有用到？
                        // lNodes.front().lit 和前面的迭代的lit 不同，只是名字相同而已
                        // lNodes.front().lit是node结构体里的一个指针用来记录节点的位置
                        // 迭代的lit 是while循环里作者命名的遍历的指针名称
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                //后面的操作都是相同的，这里不再赘述
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                //当这个母节点expand之后就从列表中删除它了，能够进行分裂操作说明至少有一个子节点的区域中特征点的数量是>1的
                // 分裂方式是后加的节点先分裂，先加的后分裂
                lit=lNodes.erase(lit);

				//继续下一次循环，其实这里加不加这句话的作用都是一样的
                continue;
            }//判断当前遍历到的节点中是否有超过一个的特征点
        }//遍历列表中的所有提取器节点

        // Finish if there are more nodes than required features or all nodes contain just one point
        //停止这个过程的条件有两个，满足其中一个即可：
        //1、当前的节点数已经超过了要求的特征点数
        //2、当前所有的节点中都只包含一个特征点
        //prevSize中保存的是分裂之前的节点个数，如果分裂之前和分裂之后的总节点个数一样，说明当前所有的节点区域中只有一个特征点，已经不能够再细分了
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)	
        {
			//停止标志置位
            bFinish = true;
        }

        // Step 6 当再划分之后所有的Node数大于要求数目时,就慢慢划分直到使其刚刚达到或者超过要求的特征点个数
        //可以展开的子节点个数nToExpand x3，是因为一分四之后，会删除原来的主节点，所以乘以3
        /**
		 * //?BUG 但是我觉得这里有BUG，虽然最终作者也给误打误撞、稀里糊涂地修复了
		 * 注意到，这里的nToExpand变量在前面的执行过程中是一直处于累计状态的，如果因为特征点个数太少，跳过了下面的else-if，又进行了一次上面的遍历
		 * list的操作之后，lNodes.size()增加了，但是nToExpand也增加了，尤其是在很多次操作之后，下面的表达式：
		 * ((int)lNodes.size()+nToExpand*3)>N
		 * 会很快就被满足，但是此时只进行一次对vSizeAndPointerToNode中点进行分裂的操作是肯定不够的；
		 * 理想中，作者下面的for理论上只要执行一次就能满足，不过作者所考虑的“不理想情况”应该是分裂后出现的节点所在区域可能没有特征点，因此将for
		 * 循环放在了一个while循环里面，通过再次进行for循环、再分裂一次解决这个问题。而我所考虑的“不理想情况”则是因为前面的一次对vSizeAndPointerToNode
		 * 中的特征点进行for循环不够，需要将其放在另外一个循环（也就是作者所写的while循环）中不断尝试直到达到退出条件。 
		 * */
        else if(((int)lNodes.size()+nToExpand*3)>N)
        {
			//如果再分裂一次那么数目就要超了，这里想办法尽可能使其刚刚达到或者超过要求的特征点个数时就退出
			//这里的nToExpand和vSizeAndPointerToNode不是一次循环对一次循环的关系，而是前者是累计计数，后者只保存某一个循环的
			//一直循环，直到结束标志位被置位
            while(!bFinish)
            {
				//获取当前的list中的节点个数
                prevSize = lNodes.size();

				//保留那些还可以分裂的节点的信息, 这里是深拷贝
                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
				//清空
                vSizeAndPointerToNode.clear();

                // 对需要划分的节点进行排序，对pair对的第一个元素进行排序，默认是从小到大排序
				// 优先分裂特征点多的节点，使得特征点密集的区域保留更少的特征点
                //! 注意这里的排序规则非常重要！会导致每次最后产生的特征点都不一样。建议使用 stable_sort
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());

				//遍历这个存储了pair对的vector，注意是从后往前遍历
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    ExtractorNode n1,n2,n3,n4;
					//对每个需要进行分裂的节点进行分裂
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
					//其实这里的节点可以说是二级子节点了，执行和前面一样的操作
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
							//因为这里还有对于vSizeAndPointerToNode的操作，所以前面才会备份vSizeAndPointerToNode中的数据
							//为可能的、后续的又一次for循环做准备
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    //删除母节点，在这里其实应该是一级子节点
                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

					//判断是是否超过了需要的特征点数？是的话就退出，不是的话就继续这个分裂过程，直到刚刚达到或者超过要求的特征点个数
					//作者的思想其实就是这样的，再分裂了一次之后判断下一次分裂是否会超过N，如果不是那么就放心大胆地全部进行分裂（因为少了一个判断因此
					//其运算速度会稍微快一些），如果会那么就引导到这里进行最后一次分裂
                    if((int)lNodes.size()>=N)
                        break;
                }//遍历vPrevSizeAndPointerToNode并对其中指定的node进行分裂，直到刚刚达到或者超过要求的特征点个数

                //这里理想中应该是一个for循环就能够达成结束条件了，但是作者想的可能是，有些子节点所在的区域会没有特征点，因此很有可能一次for循环之后
				//的数目还是不能够满足要求，所以还是需要判断结束条件并且再来一次
                //判断是否达到了停止条件
                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;				
            }//一直进行nToExpand累加的节点分裂过程，直到分裂后的nodes数目刚刚达到或者超过要求的特征点数目
        }//当本次分裂后达不到结束条件但是再进行一次完整的分裂之后就可以达到结束条件时
    }// 根据兴趣点分布,利用4叉树方法对图像进行划分区域

    // Retain the best point in each node
    // Step 7 保留每个区域响应值最大的一个兴趣点
    //使用这个vector来存储我们感兴趣的特征点的过滤结果
    vector<cv::KeyPoint> vResultKeys;

    //调整容器的大小为要提取的特征点数目
    vResultKeys.reserve(nfeatures);

    for (list<ExtractorNode>::iterator  lit = lNodes.begin(); lit !=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys=lit->vKeys;
        cv::KeyPoint* pKP=&vNodeKeys[0];
        float maxResponse=pKP->response;
        for(int i=1;i<vNodeKeys.size();i++){
            if (vNodeKeys[i].response>maxResponse)
            {
                pKP=&vNodeKeys[i];
                maxResponse=vNodeKeys[i].response;
            }
            
        }
        
        vResultKeys.push_back(*pKP);
        
    }
    return vResultKeys;
    




}//ORBextractor::DistributeOctTree

//计算四叉树的特征点，函数名字后的octtree只是说明在过滤和分配特征点的时候使用的方式
void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint>>& allkeypoints){


}//ORBextractor::ComputeKeyPointsOctTree


//这个函数已经弃用了
void ORBextractor::ComputeKeyPOintsold(vector<vector<KeyPoint>>& allkeypoints){

}//ORBextractor::ComputeKeyPOintsold

//注意这是一个不属于任何类的全局静态函数，static修饰符限定只能够被本文件中的函数调用
/**
 * @brief 计算某层金字塔上特征点的描述子
*/
static void computeDescriptors(const cv::Mat& image,vector<KeyPoint>& KeyPoints,cv::Mat& descriptors,const vector<cv::Point>& pattern)
{
    descriptors=cv::Mat::zeros((int)KeyPoints.size(),32,CV_8UC1);
    for (size_t i = 0; i < KeyPoints.size(); i++)
    {
        computerOrbDescriptor(KeyPoints[i],
                            image,
                            &pattern[0],//随机点集的首地址
                            descriptors.ptr((int) i));//取出来的描述子保存的位置
    }

}//computeDescriptors


/**
 * @brief 仿函数
*/
void ORBextractor::operator()(cv::InputArray _image,cv::InputArray _mask,std::vector<KeyPoint>& _keypoints,cv::OutputArray _descriptors){
    //step1 检查图像有效性。如果图像为空，那么就直接返回
    if(_image.empty()){
        return;
    }

    //获取图像的大小
    Mat image =_image.getMat();

    //判断图像的格式是否正确，要求是单通道的灰度值
    assert(image.type()==CV_8UC1);

    //step2 构建图像的金子塔
    ComputerPyramid(image);

    //step3 计算图像的特征点，并将特征点进行均匀化。均匀的特征点可以提高位姿计算精度
    //存储所有的节点，此处为二维的vector，第一位存储的是金字塔的层数，第二层存储的是第一层金字塔里边提取到的所有特征点
    vector<vector<cv::KeyPoint>> allkeypoins;
    //使用四叉树的方式计算每层图像的特征点进行分配
    ComputeKeyPointsOctTree(allkeypoins);

    //step4拷贝图像描述子到新的矩阵descriptors
    Mat descriptors;

    //统计整个图像金子塔的特征点
    int nkeypoints=0;

    //开始遍历每层图像金字塔，并且累加每层的特征点个数
    for (int level = 0; level < nlevels; ++level)
    {
        nkeypoints+=(int)allkeypoins[level].size();
    }

    //如果本图像金字塔中没有任何的特征点
    if(nkeypoints==0){
        _descriptors.release();
    }else{
        //如果图像金子塔中有特征点，那么就是创建这个存储描述子的矩阵，注意这个矩阵是存储整个图像金字塔中特征点的描述子
        _descriptors.create(nkeypoints,//
        32,//矩阵的列数，对应为使用32*8=256位描述子
        CV_8U);//矩阵元素的格式
        //获取这个描述子的矩阵信息
        //
        descriptors=_descriptors.getMat();
    }

    //清空用作返回特征点提取结果的vector容器
    _keypoints.clear();
    //并预分配正确大小的空间
    _keypoints.reserve(nkeypoints);

    //因为遍历是一层一层进行的，但是描述子那个矩阵存储的是整个图像金字塔中特征点的描述子，所以在这里设置offset变量来保存"寻址”时的偏移量
    //辅助进行在描述子中mat定位
    int offset=0;
    //开始遍历每一层图像
    for (int level = 0; level < nlevels; ++level)
    {
        //获取在allkeypoints中当前成本法特征点容器的句柄
        vector<KeyPoint>& keypoints=allkeypoins[level];
        //本层的特征点数
        int nkeypointsLevel=(int)keypoints.size();

        //如果特征点数目为0，跳出本次循环，继续下一层金字塔
        if(nkeypointsLevel==0){
            continue;
        }
        //step5 对图像进行高斯模糊
        //深拷贝当前金子塔所在层级的图像
        Mat workingMat = mvImagePyramid[level].clone();

        //注意：提取特征点的时候，使用的是清晰的图像；这里计算描述子的时候，为了避免图像噪声的影响，使用了搞死模糊
        GaussianBlur(workingMat,//源图像
        workingMat,//输出出图像
        Size(7,7),2,2,BORDER_REFLECT_101);

        //计算描述子
        //desc存储当前图层的描述子
        Mat desc=descriptors.rowRange(offset,offset+nkeypointsLevel);

        //step6 计算高斯模糊之后的图像的描述子
        computeDescriptors(workingMat,keypoints,desc,pattern);

        // 更新偏移量的值 
        offset += nkeypointsLevel;

        // Scale keypoint coordinates
		// Step 6 对非第0层图像中的特征点的坐标恢复到第0层图像（原图像）的坐标系下
        // ? 得到所有层特征点在第0层里的坐标放到_keypoints里面
		// 对于第0层的图像特征点，他们的坐标就不需要再进行恢复了
        if (level != 0)
        {
			// 获取当前图层上的缩放系数
            float scale = mvScaleFactor[level];
            // 遍历本层所有的特征点
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
				// 特征点本身直接乘缩放倍数就可以了
                keypoint->pt *= scale;
        }
        
        // And add the keypoints to the output
        // 将keypoints中内容插入到_keypoints 的末尾
        // keypoint其实是对allkeypoints中每层图像中特征点的引用，这样allkeypoints中的所有特征点在这里被转存到输出的_keypoints
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());




    }//开始遍历每一层图像
    
    

}

/**
 * @brief 构建图像金字塔
 * @image 输入原图像，这个输入图像所有的像素是有效的，也就是说都可以在其上边提取到fast角点
*/
void ORBextractor::ComputerPyramid(cv::Mat image){

    //遍历所有的图层
    for (int level = 0; level < nlevels; ++level)
    {
        float scale = mvInvScaleFactor[level];
        //计算本层图像的像素尺寸大小
        Size sz(cvRound((float)image.cols*scale),cvRound((float)image.rows*scale));
        //全尺寸图像。包括无效图像区域的大小。将图像进行“补边”，EDGE_THRESHOLD区域外的图像不进行FAST角点检测
        Size wholeSize(sz.width+EDGE_THRESHOLD*2,sz.height+EDGE_THRESHOLD*2);
        //定义了两个变量：temp是扩展边界的图像，maketemp并未使用
        Mat temp(wholeSize,image.type()),masktemp;
        //mvImagePyramid刚开始时是一个空的vector<Mat>
        //把图像金子塔该图层的图像指针mvImagePyramid指向temp的中间部分（这里为浅拷贝，内存相同）
        mvImagePyramid[level]=temp(Rect(EDGE_THRESHOLD,EDGE_THRESHOLD,sz.width,sz.height));
        //计算第0层以上的resize后的图像
        if(level!=0){
            //将上一层金子特图像根据设定sz缩放到当前层级
            resize(mvImagePyramid[level-1],//输入图像
                    mvImagePyramid[level],//输出图像
                    sz,//输出图像的尺寸
                    0,0,//水平方向、垂直方向的缩放系数。留0表示自动计算
                    cv::INTER_LINEAR);//图像的缩放的差值算法，这里是线性差值运算

            copyMakeBorder(mvImagePyramid[level],//源图像
            temp,  //目标图像
            EDGE_THRESHOLD,EDGE_THRESHOLD,  ////top & bottom 需要扩展的border大小
            EDGE_THRESHOLD,EDGE_THRESHOLD,  //left & right 需要扩展的border大小
            BORDER_REFLECT_101+BORDER_ISOLATED);//扩充方式
        }
        else{
            //对于第0层的为缩放的图像，直接将图像深拷贝到temp的中间，并且对其周围进行边界扩展。此时temp就是对原始图像扩展后的图像
            copyMakeBorder(image,
                            temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                            BORDER_REFLECT_101);
        }


    }
    


}



}//namespace ORB_SLAM2