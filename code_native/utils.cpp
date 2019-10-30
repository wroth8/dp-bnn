// Implementation of the functions defined in utils.h (see there for more
// details)
//
// @author Wolfgang Roth

#define _MATLAB

#include "utils.h"
#include "definitions.h"

#include <stdarg.h>
#include <string.h>
#include <math.h>

#ifndef _WIN32
#include <sys/time.h>
#endif

#ifdef _MATLAB
#include <mex.h>
#else
#include <stdio.h>
#include <stdlib.h>
#endif

void _printf_verbosity(int verbosity, int verbosity_needed, const char* format, ...)
{
  if (verbosity < verbosity_needed)
    return;

  char buffer[512];
  va_list args;
  va_start(args, format);
  vsprintf(buffer, format, args);
  va_end(args);
#ifdef _MATLAB
  mexPrintf(buffer);
  mexEvalString("drawnow;");
#else
  printf(buffer);
#endif
}

void _printf(const char* format, ...)
{
  char buffer[512];
  va_list args;
  va_start(args, format);
  vsprintf(buffer, format, args);
  va_end(args);
#ifdef _MATLAB
  mexPrintf(buffer);
  mexEvalString("drawnow;");
#else
  printf(buffer);
#endif
}

void* _realloc(void* ptr, int size)
{
#ifdef _MATLAB
  return mxRealloc(ptr, size);
#else
  return realloc(ptr, size);
#endif
}

unsigned long long get_time_in_ms()
{
#ifdef _WIN32
  SYSTEMTIME sys_time;
  FILETIME file_time;
  GetSystemTime(&sys_time);
  SystemTimeToFileTime(&sys_time, &file_time);
  ULARGE_INTEGER ularge_int;
  ularge_int.HighPart = file_time.dwHighDateTime;
  ularge_int.LowPart  = file_time.dwLowDateTime;
  return ularge_int.QuadPart / 10000;
#else
  struct timeval  tv;
  gettimeofday(&tv, NULL);

  unsigned long long time_in_mill =
           (tv.tv_sec) * 1000 + (tv.tv_usec) / 1000 ; // convert tv_sec & tv_usec to millisecond
  return time_in_mill;
#endif
}

int get_next_power2(int num)
{
  // The following implementation was taken from http://graphics.stanford.edu/~seander/bithacks.html#RoundUpPowerOf2
  num--;
  num |= num >> 1;
  num |= num >> 2;
  num |= num >> 4;
  num |= num >> 8;
  num |= num >> 16;
  num++;
  return num;
}

// Code from http://guihaire.com/code/?p=414
int log2int(unsigned int v)
{
  register unsigned int r;
  register unsigned int shift;
  r = (v > 0xFFFF) << 4; v >>= r;
  shift = (v > 0xFF ) << 3; v >>= shift; r |= shift;
  shift = (v > 0xF ) << 2; v >>= shift; r |= shift;
  shift = (v > 0x3 ) << 1; v >>= shift; r |= shift;
  r |= (v >> 1);
  return r;
}

#define TANH_DISCRETIZATION 0x10000
#define TANH_LEFT_BORDER -10.0
#define TANH_RIGHT_BORDER 10.0
#define TANH_WIDTH (TANH_RIGHT_BORDER - TANH_LEFT_BORDER)
#define TANH_INV_WIDTH (1/TANH_WIDTH)

double my_tanh_map[TANH_DISCRETIZATION + 1];
int init_tanh = 0;

void my_tanh_init()
{
  int i;
  double step = (double) TANH_WIDTH / (double) TANH_DISCRETIZATION;

  if (init_tanh)
    return;

  for (i = 0; i <= TANH_DISCRETIZATION; i++)
  {
    my_tanh_map[i] = tanh((double) TANH_LEFT_BORDER + step * (double) i);
  }
  init_tanh = 1;
}

double my_tanh(double x)
{
  if (x < TANH_LEFT_BORDER)
    return -1;
  if (x > TANH_RIGHT_BORDER)
    return 1;
  return my_tanh_map[(int) ((x - TANH_LEFT_BORDER) * TANH_INV_WIDTH * (double) TANH_DISCRETIZATION + 0.5)]; // calling round() takes more time
}

#define SIGMOID_DISCRETIZATION 0x10000
#define SIGMOID_LEFT_BORDER -10.0
#define SIGMOID_RIGHT_BORDER 10.0
#define SIGMOID_WIDTH (SIGMOID_RIGHT_BORDER - SIGMOID_LEFT_BORDER)
#define SIGMOID_INV_WIDTH (1/SIGMOID_WIDTH)

double my_sigmoid_map[SIGMOID_DISCRETIZATION + 1];
int init_sigmoid = 0;

void my_sigmoid_init()
{
  int i;
  double step = (double) SIGMOID_WIDTH / (double) SIGMOID_DISCRETIZATION;

  if (init_sigmoid)
    return;

  for (i = 0; i <= SIGMOID_DISCRETIZATION; i++)
  {
    double x = (double) SIGMOID_LEFT_BORDER + step * (double) i;
    my_sigmoid_map[i] = 1 / (1 + exp(-x));
  }
  init_sigmoid = 1;
}

double my_sigmoid(double x)
{
  if (x < SIGMOID_LEFT_BORDER)
    return 0;
  if (x > SIGMOID_RIGHT_BORDER)
    return 1;
  return my_sigmoid_map[(int) ((x - SIGMOID_LEFT_BORDER) * SIGMOID_INV_WIDTH * (double) SIGMOID_DISCRETIZATION + 0.5)]; // calling round() takes more time
}

#define LOG01_ZERO_THRESHOLD 1e-50
#define LOG01_MIN_X 0.5
#define LOG01_DISCRETIZATION 0x10000
#define LOG01_INV_WIDTH (1/(1 - LOG01_MIN_X))

double my_log01_map[LOG01_DISCRETIZATION + 1];
double my_log01_zero;
int init_log01 = 0;

void my_log01_init()
{
  int i;
  double step = (1.0 - LOG01_MIN_X) / (double) LOG01_DISCRETIZATION;

  if (init_log01)
    return;

  my_log01_zero = log(LOG01_ZERO_THRESHOLD);
  for (i = 0; i < LOG01_DISCRETIZATION; i++)
  {
    double x = LOG01_MIN_X + (double) i * step;
    my_log01_map[i] = log(x);
  }

  init_log01 = 1;
}

double my_log01(double x)
{
  if (x < LOG01_ZERO_THRESHOLD)
    return my_log01_zero;
  else if (x < LOG01_MIN_X)
    return log(x);
  return my_log01_map[(int) ((x - LOG01_MIN_X) * LOG01_INV_WIDTH * (double) LOG01_DISCRETIZATION + 0.5)]; // calling round() takes more time
}

#define SOFTPLUS_DISCRETIZATION 0x10000
#define SOFTPLUS_LEFT_BORDER -10.0
#define SOFTPLUS_RIGHT_BORDER 10.0
#define SOFTPLUS_WIDTH (SOFTPLUS_RIGHT_BORDER - SOFTPLUS_LEFT_BORDER)
#define SOFTPLUS_INV_WIDTH (1/SOFTPLUS_WIDTH)

double softplus_map[SOFTPLUS_DISCRETIZATION + 1];
int init_softplus = 0;

void softplus_approx_init()
{
  int i;
  double step = (double) SOFTPLUS_WIDTH / (double) SOFTPLUS_DISCRETIZATION;

  if (init_softplus)
    return;

  for (i = 0; i <= SOFTPLUS_DISCRETIZATION; i++)
  {
    double x = (double) SOFTPLUS_LEFT_BORDER + step * (double) i;
    softplus_map[i] = log(1 + exp(x));
  }

  init_softplus = 1;
}

double softplus_approx(double x)
{
  if (x < SOFTPLUS_LEFT_BORDER)
    return 0;
  if (x > SOFTPLUS_RIGHT_BORDER)
    return x;
  return softplus_map[(int) ((x - SOFTPLUS_LEFT_BORDER) * SOFTPLUS_INV_WIDTH * (double) SOFTPLUS_DISCRETIZATION + 0.5)]; // calling round() takes more time
}

double softplus(double x)
{
  if (x < -40)
    return 0;
  else if (x > 40)
    return x;
  return log(1 + exp(x));
}

double softplus_coef[164][4] = {
  {4.038184993712483e-10, 1.017885937220340e-09, 2.061969919854453e-09, 2.061153620314381e-09},
  {4.038184993712219e-10, 1.169317874484561e-09, 2.335370396317565e-09, 2.335593036071840e-09},
  {4.719933480023286e-10, 1.320749811748776e-09, 2.646628857096732e-09, 2.646573635406941e-09},
  {5.309781764185064e-10, 1.497747317249646e-09, 2.998940998221535e-09, 2.998960820360424e-09},
  {6.027114771853077e-10, 1.696864133406579e-09, 3.398267429553564e-09, 3.398267813720959e-09},
  {6.826844156907615e-10, 1.922880937351063e-09, 3.850735563398270e-09, 3.850741915353510e-09},
  {7.736570529575627e-10, 2.178887593235098e-09, 4.363456629721540e-09, 4.363462243423800e-09},
  {8.766483913324597e-10, 2.469008988094198e-09, 4.944443702387700e-09, 4.944450489714849e-09},
  {9.933780977354086e-10, 2.797752134843883e-09, 5.602788842754959e-09, 5.602796421841604e-09},
  {1.125643422487363e-09, 3.170268921494675e-09, 6.348791474797277e-09, 6.348800095890049e-09},
  {1.275521481134752e-09, 3.592385204927436e-09, 7.194123240600040e-09, 7.194133004447609e-09},
  {1.445355084740341e-09, 4.070705760352968e-09, 8.152009611260091e-09, 8.152020681242446e-09},
  {1.637801898337863e-09, 4.612713917130649e-09, 9.237437070945536e-09, 9.237449619305356e-09},
  {1.855872670464535e-09, 5.226889629007334e-09, 1.046738751421279e-08, 1.046740173996141e-08},
  {2.102979235145510e-09, 5.922841880431574e-09, 1.186110395289264e-08, 1.186112008100074e-08},
  {2.382987650613294e-09, 6.711459093611101e-09, 1.344039157464798e-08, 1.344040986081271e-08},
  {2.700278749577512e-09, 7.605079462591178e-09, 1.522995889417326e-08, 1.522997962873649e-08},
  {3.059816661523782e-09, 8.617683993682693e-09, 1.725780432620750e-08, 1.725782783904390e-08},
  {3.467226482829916e-09, 9.765115241754058e-09, 1.955565423063710e-08, 1.955568089663817e-08},
  {3.928882281972004e-09, 1.106532517281514e-08, 2.215945928245826e-08, 2.215948952784449e-08},
  {4.452006824276018e-09, 1.253865602855451e-08, 2.510995693262949e-08, 2.510999124218399e-08},
  {5.044784574298278e-09, 1.420815858765791e-08, 2.845330875965605e-08, 2.845334768503750e-08},
  {5.716489744380809e-09, 1.609995280301966e-08, 3.224182268349077e-08, 3.224186685279834e-08},
  {6.477631392646912e-09, 1.824363645716236e-08, 3.653477134101353e-08, 3.653482146981386e-08},
  {7.340117840374910e-09, 2.067274822940485e-08, 4.139931942683445e-08, 4.139937633089748e-08},
  {8.317442981306247e-09, 2.342529241954512e-08, 4.691157450795323e-08, 4.691163911799322e-08},
  {9.424897397570865e-09, 2.654433353753502e-08, 5.315777775258824e-08, 5.315785113136563e-08},
  {1.067980758512599e-08, 3.007867006162393e-08, 6.023565320248313e-08, 6.023573656469277e-08},
  {1.210180703167792e-08, 3.408359790604597e-08, 6.825593669844189e-08, 6.825603143390574e-08},
  {1.371314338762208e-08, 3.862177554292487e-08, 7.734410837956329e-08, 7.734421608035169e-08},
  {1.553902653633110e-08, 4.376420431328347e-08, 8.764235586158929e-08, 8.764247835383425e-08},
  {1.760802300858091e-08, 4.959133926440763e-08, 9.931179880880068e-08, 9.931193819013173e-08},
  {1.995250291256501e-08, 5.619434789262589e-08, 1.125350097034298e-07, 1.125351683871768e-07},
  {2.260914637046631e-08, 6.367653648483777e-08, 1.275188702506128e-07, 1.275190510181789e-07},
  {2.561951738633404e-08, 7.215496637376222e-08, 1.444978081079378e-07, 1.444980141710859e-07},
  {2.903071412100235e-08, 8.176228539363748e-08, 1.637374645788628e-07, 1.637376996540634e-07},
  {3.289610574988493e-08, 9.264880318901273e-08, 1.855388506516941e-07, 1.855391190492144e-07},
  {3.727616742897062e-08, 1.049848428452194e-07, 2.102430564059732e-07, 2.102433631370472e-07},
  {4.223942643346135e-08, 1.189634056310833e-07, 2.382365874655110e-07, 2.382369383717601e-07},
  {4.786353426938788e-08, 1.348031905436316e-07, 2.699574119873504e-07, 2.699578138976876e-07},
  {5.423648153096507e-08, 1.527520158946533e-07, 3.059018127921358e-07, 3.059022737137205e-07},
  {6.145797450879531e-08, 1.730906964687660e-07, 3.466321518375631e-07, 3.466326811890816e-07},
  {6.964099508413143e-08, 1.961374369095643e-07, 3.927856685098544e-07, 3.927862774075639e-07},
  {7.891356831089697e-08, 2.222528100661144e-07, 4.450844493818141e-07, 4.450851509915314e-07},
  {8.942076533734547e-08, 2.518453981827004e-07, 5.043467254129160e-07, 5.043475353846484e-07},
  {1.013269729975577e-07, 2.853781851842057e-07, 5.714996733337792e-07, 5.715006103401671e-07},
  {1.148184655768401e-07, 3.233758000582907e-07, 6.475939214890911e-07, 6.475950078945286e-07},
  {1.301063189766807e-07, 3.664327246496058e-07, 7.338199870775782e-07, 7.338212497721555e-07},
  {1.474297128666729e-07, 4.152225942658611e-07, 8.315269019420116e-07, 8.315283733837542e-07},
  {1.670596724681592e-07, 4.705087365908626e-07, 9.422433182991021e-07, 9.422450378198525e-07},
  {1.893033084989301e-07, 5.331561137664248e-07, 1.067701424593763e-06, 1.067703440039262e-06},
  {2.145086215921176e-07, 6.041448544535219e-07, 1.209864045621256e-06, 1.209866415415495e-06},
  {2.430699463244935e-07, 6.845855875505643e-07, 1.370955350871767e-06, 1.370958146620535e-06},
  {2.754341200005879e-07, 7.757368174222426e-07, 1.553495651493369e-06, 1.553498961273898e-06},
  {3.121074726502248e-07, 8.790246124224546e-07, 1.760340830223957e-06, 1.760344762748418e-06},
  {3.536637475526632e-07, 9.960649146662889e-07, 1.994727021110050e-06, 1.994731711003034e-06},
  {4.007530761189709e-07, 1.128688819998527e-06, 2.260321237943153e-06, 2.260326852440390e-06},
  {4.541121474693907e-07, 1.278971223543142e-06, 2.561278743385862e-06, 2.561285490847556e-06},
  {5.145757316671361e-07, 1.449263278844149e-06, 2.902308056184275e-06, 2.902316196926676e-06},
  {5.830897367548559e-07, 1.642229178219339e-06, 3.288744613317210e-06, 3.288754473407563e-06},
  {6.607260036632580e-07, 1.860887829502420e-06, 3.726634239282428e-06, 3.726646228123990e-06},
  {7.486990701932941e-07, 2.108660080876108e-06, 4.222827728079748e-06, 4.222842360846174e-06},
  {8.483851660433359e-07, 2.389422232198587e-06, 4.785088017214086e-06, 4.785105943491303e-06},
  {9.613437355884477e-07, 2.707566669464817e-06, 5.422211629922014e-06, 5.422233670298595e-06},
  {1.089341824646512e-06, 3.068070570310458e-06, 6.144166284893927e-06, 6.144193477732806e-06},
  {1.234381712004421e-06, 3.476573754552893e-06, 6.962246825501847e-06, 6.962280486756910e-06},
  {1.398732217134800e-06, 3.939466896554504e-06, 7.889251906890277e-06, 7.889293706640789e-06},
  {1.584964172752492e-06, 4.463991477980040e-06, 8.939684203707097e-06, 8.939736264122329e-06},
  {1.795990615635385e-06, 5.058353042762211e-06, 1.012997726879988e-05, 1.013004228957906e-05},
  {2.035112322683222e-06, 5.731849523625481e-06, 1.147875258959834e-05, 1.147883400864337e-05},
  {2.306069402019926e-06, 6.495016644631662e-06, 1.300711086063049e-05, 1.300721305990505e-05},
  {2.613099743265998e-06, 7.359792670389120e-06, 1.473896202500809e-05, 1.473909059435705e-05},
  {2.961005237249546e-06, 8.339705074113870e-06, 1.670139924307096e-05, 1.670156131839396e-05},
  {3.355226796012984e-06, 9.450082038082395e-06, 1.892512263209550e-05, 1.892532732891486e-05},
  {3.801929340039928e-06, 1.070829208658729e-05, 2.144491939767921e-05, 2.144517836710780e-05},
  {4.308098073778875e-06, 1.213401558910218e-05, 2.430020785714040e-05, 2.430053599891289e-05},
  {4.881647544511484e-06, 1.374955236676931e-05, 2.753565385162433e-05, 2.753607022868551e-05},
  {5.531545176675097e-06, 1.558017019596128e-05, 3.120186917196564e-05, 3.120239818372970e-05},
  {6.267951196041975e-06, 1.765449963721455e-05, 3.535620290111260e-05, 3.535687578871049e-05},
  {7.102377109350407e-06, 2.000498133573035e-05, 4.006363802273071e-05, 4.006449480036081e-05},
  {8.047865188729863e-06, 2.266837275173664e-05, 4.539780728366410e-05, 4.539889921686464e-05},
  {9.119191730399059e-06, 2.568632219751029e-05, 5.144214415231997e-05, 5.144353693826528e-05},
  {1.033309721765749e-05, 2.910601909640977e-05, 5.829118681406000e-05, 5.829296466298981e-05},
  {1.170854692561681e-05, 3.298093055303128e-05, 6.605205552024014e-05, 6.605432639363194e-05},
  {1.326702596292795e-05, 3.737163565013725e-05, 7.484612629563624e-05, 7.484902862926717e-05},
  {1.503287326098165e-05, 4.234677038623502e-05, 8.481092705018280e-05, 8.481463838333894e-05},
  {1.703365960173571e-05, 4.798409785910303e-05, 9.610228558085007e-05, 9.610703363248457e-05},
  {1.930061542416463e-05, 5.437172020975392e-05, 1.088967628394572e-04, 1.089028397255490e-04},
  {2.186911488127191e-05, 6.160945099381544e-05, 1.233944092399034e-04, 1.234021897232588e-04},
  {2.477922343436904e-05, 6.981036907429262e-05, 1.398218867484169e-04, 1.398318516650276e-04},
  {2.807631718676831e-05, 7.910257786218036e-05, 1.584360051154761e-04, 1.584487714461358e-04},
  {3.181178317524479e-05, 8.963119680721870e-05, 1.795277269491509e-04, 1.795440864216723e-04},
  {3.604381097409640e-05, 1.015606154979353e-04, 2.034267034872952e-04, 2.034476721294431e-04},
  {4.083828722161592e-05, 1.150770446132201e-04, 2.305064110011898e-04, 2.305332927508190e-04},
  {4.626980607278647e-05, 1.303914023213261e-04, 2.611899668680081e-04, 2.612244352277790e-04},
  {5.242281011383007e-05, 1.477425795986206e-04, 2.959567146080015e-04, 2.960009174625366e-04},
  {5.939287795634837e-05, 1.674011333913069e-04, 3.353496787317424e-04, 3.354063728957688e-04},
  {6.728817656279779e-05, 1.896734626249388e-04, 3.799840032337729e-04, 3.800567271612343e-04},
  {7.623109834357239e-05, 2.149065288359880e-04, 4.305565021663888e-04, 4.306497976388198e-04},
  {8.636010519762205e-05, 2.434931907148285e-04, 4.878564671102407e-04, 4.879761637866330e-04},
  {9.783180394438290e-05, 2.758782301639376e-04, 5.527778947200864e-04, 5.529314753607964e-04},
  {1.108232799698489e-04, 3.125651566430803e-04, 6.263333180709637e-04, 6.265303872891976e-04},
  {1.255347183741345e-04, 3.541238866317754e-04, 7.096694484803205e-04, 7.099223343393073e-04},
  {1.421923443711837e-04, 4.011994060220759e-04, 8.040848600620519e-04, 8.044093861247933e-04},
  {1.610517171108314e-04, 4.545215351612689e-04, 9.110499777099701e-04, 9.114664537742447e-04},
  {1.824014133080254e-04, 5.149159290778246e-04, 1.032229660739858e-03, 1.032764154109868e-03},
  {2.065671389573920e-04, 5.833164590683271e-04, 1.169508709258127e-03, 1.170194675854558e-03},
  {2.339163087309443e-04, 6.607791361773439e-04, 1.325020658663837e-03, 1.325901035628043e-03},
  {2.648631331323287e-04, 7.484977519514480e-04, 1.501180269679936e-03, 1.502310159754284e-03},
  {2.998742527253184e-04, 8.478214268760696e-04, 1.700720167033376e-03, 1.702170281645416e-03},
  {3.394749562475458e-04, 9.602742716480657e-04, 1.926732129348893e-03, 1.928593204219381e-03},
  {3.842560145380369e-04, 1.087577380240883e-03, 2.182713585835013e-03, 2.185102042906414e-03},
  {4.348811540685060e-04, 1.231673385692647e-03, 2.472619931576704e-03, 2.475685137730449e-03},
  {4.920951814298691e-04, 1.394753818468330e-03, 2.800923332096827e-03, 2.804856903083025e-03},
  {5.567327519814813e-04, 1.579289511504520e-03, 3.172678748343434e-03, 3.177726471409926e-03},
  {6.297277503519993e-04, 1.788064293497572e-03, 3.593597973968697e-03, 3.600075082226328e-03},
  {7.121232153516344e-04, 2.024212199879555e-03, 4.070132535640840e-03, 4.078443270570721e-03},
  {8.050816945884831e-04, 2.291258405636390e-03, 4.609566361330336e-03, 4.620229018803927e-03},
  {9.098958512658850e-04, 2.593164041107085e-03, 5.220119167173269e-03, 5.233798151743031e-03},
  {1.027999063552887e-03, 2.934374985331792e-03, 5.911061545478128e-03, 5.928608376116491e-03},
  {1.160975650677964e-03, 3.319874634164111e-03, 6.692842747915118e-03, 6.715348489118068e-03},
  {1.310570224302587e-03, 3.755240503168306e-03, 7.577232140081675e-03, 7.606094404334003e-03},
  {1.478695492376447e-03, 4.246704337281804e-03, 8.577475245137935e-03, 8.614483762175558e-03},
  {1.667437628840140e-03, 4.801215146922944e-03, 9.708465180663532e-03, 9.755911000221376e-03},
  {1.879058059024552e-03, 5.426504257737927e-03, 1.098693010624615e-02, 1.104774484859382e-02},
  {2.115990190220463e-03, 6.131151029872134e-03, 1.243163701719741e-02, 1.250957027617327e-02},
  {2.380829233730397e-03, 6.924647351204766e-03, 1.406361181483202e-02, 1.416345693150498e-02},
  {2.676312814874349e-03, 7.817458313853609e-03, 1.590637502296433e-02, 1.603425608031868e-02},
  {3.005289551581303e-03, 8.821075619431462e-03, 1.798619176462497e-02, 1.814992791780974e-02},
  {3.370672208158343e-03, 9.948059201274340e-03, 2.033233361721321e-02, 2.054190090109691e-02},
  {3.775371417357842e-03, 1.121206127933361e-02, 2.297734867728921e-02, 2.324546437242503e-02},
  {4.222205346914976e-03, 1.262782556084285e-02, 2.595733453231126e-02, 2.630019518687529e-02},
  {4.713780124445677e-03, 1.421115256593597e-02, 2.931220679815862e-02, 2.975041827262057e-02},
  {5.252335417858855e-03, 1.597882011260293e-02, 3.308595338297600e-02, 3.364569998303870e-02},
  {5.839549427917312e-03, 1.794844589429984e-02, 3.732686163383887e-02, 3.804137168778313e-02},
  {6.476297866583991e-03, 2.013827692976883e-02, 4.208770198684745e-02, 4.299907922908782e-02},
  {7.162362507886488e-03, 2.256688862973766e-02, 4.742584768178578e-02, 4.858735157374206e-02},
  {7.896086906372446e-03, 2.525277457019481e-02, 5.340330558177737e-02, 5.488217915807815e-02},
  {8.673980232000389e-03, 2.821380716008448e-02, 6.008662829806229e-02, 6.196758900319863e-02},
  {9.490275249034674e-03, 3.146654974708452e-02, 6.754667291145842e-02, 6.993619964497336e-02},
  {1.033645362565050e-02, 3.502540296547230e-02, 7.585816700052805e-02, 7.888973429254963e-02},
  {1.120076124690428e-02, 3.890157307509112e-02, 8.509903900559850e-02, 8.893946547493874e-02},
  {1.206774802697907e-02, 4.310185854268000e-02, 9.534946795781991e-02, 1.002065589167472e-01},
  {1.291788047178866e-02, 4.762726405279660e-02, 1.066906082822546e-01, 1.128222787715693e-01},
  {1.372728990393135e-02, 5.247146922971679e-02, 1.192029499425688e-01, 1.269280110429725e-01},
  {1.446773297147708e-02, 5.761920294369061e-02, 1.329642839642448e-01, 1.426750576056015e-01},
  {1.510685096767439e-02, 6.304460280799451e-02, 1.480472596832054e-01, 1.602241504380872e-01},
  {1.560881678874004e-02, 6.870967192087241e-02, 1.645165440243138e-01, 1.797446353856590e-01},
  {1.593544848504358e-02, 7.456297821664948e-02, 1.824256252915041e-01, 2.014132779827524e-01},
  {1.604784202108789e-02, 8.053877139853971e-02, 2.018133439934029e-01, 2.254126516016479e-01},
  {1.590852903697559e-02, 8.655671215644656e-02, 2.227002794377763e-01, 2.519290813453729e-01},
  {1.548410019870161e-02, 9.252241054531218e-02, 2.450851697754962e-01, 2.811501362483148e-01},
  {1.474814534696023e-02, 9.832894811982529e-02, 2.689415896086383e-01, 3.132616875182229e-01},
  {1.368429432983476e-02, 1.038595026249340e-01, 2.942151459517334e-01, 3.484445810050557e-01},
  {1.228897686040753e-02, 1.089911129986225e-01, 3.208214729046779e-01, 3.868710061148999e-01},
  {1.057378360831152e-02, 1.135994793212745e-01, 3.486452969446652e-01, 4.287006782765186e-01},
  {8.566129564428593e-03, 1.175646481743917e-01, 3.775408128816234e-01, 4.740769841801067e-01},
  {6.311380373965392e-03, 1.207769467610507e-01, 4.073335122485539e-01, 5.231232641398400e-01},
  {3.863316006228246e-03, 1.231437144012864e-01, 4.378235948938462e-01, 5.759394198788436e-01},
  {1.311249427004668e-03, 1.245924579036224e-01, 4.687906164319597e-01, 6.325990353171691e-01},
  {-1.336302195781514e-03, 1.250841764387474e-01, 5.000001957247562e-01, 6.931471805599453e-01},
  {-3.738052162312044e-03, 1.245830631153284e-01, 5.312086006690158e-01, 7.575990353171691e-01},
  {-6.787382980839141e-03, 1.231812935544614e-01, 5.621791452527395e-01, 8.259394198788436e-01},
  {-6.787382980832035e-03, 1.206360249366476e-01, 5.926563100641280e-01, 8.981232641398400e-01},
  };

double softplus_negative_approx(double x)
{
  // Spline approximation of the softplus function for non-positive arguments.
  // Note: It is not checked whether the argument is really non-negative.
  //
  // The polynomial coefficients were computed with Matlab's 'spline' function.
  // The maximum absolute error of this approximation is <8e-8 by visual
  // inspection. The x-values are computed at {-20,-19.875,...,0,0.125,0.25,
  // 0.375,0.5}. The range of these x-values extends over 0 until 0.5 because
  // splines are a global interpolation scheme where coefficients depend on all
  // x-values and this reduces the approximation error close to 0.
  int coeff_idx = 0;
  if (x < -20.0)
    return 0;
  x = (x + 20.0) * 8.0;
  coeff_idx = (int) x;
  x = (x - (double) coeff_idx) * 0.125;
  return softplus_coef[coeff_idx][3] +
      x * (softplus_coef[coeff_idx][2] +
        x * (softplus_coef[coeff_idx][1] +
          x * softplus_coef[coeff_idx][0]));
}

void free_model(int num_layers, int sharing, int** layout, int*** ZW, int*** Zb,
    double*** W, int*** num_weights, int** num_unique_weights, double** alpha,
    double** gamma)
{
  int l;
  if (*layout != NULL)
  {
    free(*layout);
    *layout = NULL;
  }
  if (*ZW != NULL)
  {
    for (l = 0; l < num_layers; l++)
    {
      if ((*ZW)[l] != NULL)
      {
        free((*ZW)[l]);
        (*ZW)[l] = NULL;
      }
    }
    free(*ZW);
    *ZW = NULL;
  }
  if (*Zb != NULL)
  {
    for (l = 0; l < num_layers; l++)
    {
      if ((*Zb)[l] != NULL)
      {
        free((*Zb)[l]);
        (*Zb)[l] = NULL;
      }
    }
    free(*Zb);
    *Zb = NULL;
  }
  if (*W != NULL)
  {
    for (l = 0; l < (sharing == SHARING_LAYERWISE ? num_layers : 1); l++)
    {
      if ((*W)[l] != NULL)
      {
        free((*W)[l]);
        (*W)[l] = NULL;
      }
    }
    free(*W);
    *W = NULL;
  }
  if (*num_weights != NULL)
  {
    for (l = 0; l < (sharing == SHARING_LAYERWISE ? num_layers : 1); l++)
    {
      if ((*num_weights)[l] != NULL)
      {
        free((*num_weights)[l]);
        (*num_weights)[l] = NULL;
      }
    }
    free(*num_weights);
    *num_weights = NULL;
  }
  if (*num_unique_weights != NULL)
  {
    free(*num_unique_weights);
    *num_unique_weights = NULL;
  }
  if (*alpha != NULL)
  {
    free(*alpha);
    *alpha = NULL;
  }
  if (*gamma)
  {
    free(*gamma);
    *gamma = NULL;
  }
}

int load_model(char* file, int** layout, int* task, int* sharing,
    int* activation, int* num_layers, int*** ZW, int*** Zb, double*** W,
    int*** num_weights, int** num_unique_weights, double** alpha, double* beta,
    double** gamma)
{
  int error = 0;
  char dummy[1024];
  int str_size = 0;
  char* str = NULL;
  size_t elements_read = 0;

  int l;

  FILE* fp = NULL;

  *num_layers = 0;
  *layout = NULL;
  *ZW = NULL;
  *Zb = NULL;
  *W = NULL;
  *num_weights = NULL;
  *num_unique_weights = NULL;
  *alpha = NULL;
  *gamma = NULL;

  fp = fopen(file, "r");
  if (fp == NULL)
    goto file_error;

  //----------------------------------------------------------------------------
  // Read 'task'
  elements_read = fread(&str_size, sizeof(int), 1, fp);
  if (elements_read != 1)
    goto file_error;

  str = (char*) malloc(sizeof(char) * (str_size + 1));
  if (str == NULL)
    goto memory_error;

  elements_read = fread(str, sizeof(char), str_size, fp);
  if (elements_read != str_size)
    goto file_error;

  str[str_size] = '\0';

  if (!strcmp(str, "regress"))
    *task = TASK_REGRESSION;
  else if (!strcmp(str, "biclass"))
    *task = TASK_BINARY_CLASSIFICATION;
  else if (!strcmp(str, "muclass"))
    *task = TASK_MULTICLASS_CLASSIFICATION;
  else
    goto file_error;

  free(str);
  str = NULL;

  //----------------------------------------------------------------------------
  // Read 'sharing'
  elements_read = fread(&str_size, sizeof(int), 1, fp);
  if (elements_read != 1)
    goto file_error;

  str = (char*) malloc(sizeof(char) * (str_size + 1));
  if (str == NULL)
    goto memory_error;

  elements_read = fread(str, sizeof(char), str_size, fp);
  if (elements_read != str_size)
    goto file_error;

  str[str_size] = '\0';

  if (!strcmp(str, "layerwise"))
    *sharing = SHARING_LAYERWISE;
  else if (!strcmp(str, "global"))
    *sharing = SHARING_GLOBAL;
  else
    goto file_error;

  free(str);
  str = NULL;

  //----------------------------------------------------------------------------
  // Read 'activation'
  elements_read = fread(&str_size, sizeof(int), 1, fp);
  if (elements_read != 1)
    goto file_error;

  str = (char*) malloc(sizeof(char) * (str_size + 1));
  if (str == NULL)
    goto memory_error;

  elements_read = fread(str, sizeof(char), str_size, fp);
  if (elements_read != str_size)
    goto file_error;

  str[str_size] = '\0';

  if (!strcmp(str, "sigmoid"))
    *activation = ACTIVATION_SIGMOID;
  else if (!strcmp(str, "tanh"))
    *activation = ACTIVATION_TANH;
  else if (!strcmp(str, "relu"))
    *activation = ACTIVATION_RELU;
  else
    goto file_error;

  free(str);
  str = NULL;

  //----------------------------------------------------------------------------
  // Read 'num_layers'
  elements_read = fread(num_layers, sizeof(int), 1, fp);
  if (elements_read != 1)
    goto file_error;

  //----------------------------------------------------------------------------
  // Read 'layout'
  *layout = (int*) malloc(sizeof(int) * (*num_layers + 1));
  if (*layout == NULL)
    goto memory_error;

  elements_read = fread(*layout, sizeof(int), *num_layers + 1, fp);
  if (elements_read != *num_layers + 1)
    goto file_error;

  *ZW = (int**) calloc(*num_layers, sizeof(int*));
  if (*ZW == NULL)
    goto memory_error;

  *Zb = (int**) calloc(*num_layers, sizeof(int*));
  if (*Zb == NULL)
    goto memory_error;

  *W = (double**) calloc(*num_layers, sizeof(double*));
  if (*W == NULL)
    goto memory_error;

  *num_weights = (int**) calloc(*num_layers, sizeof(int*));
  if (*num_weights == NULL)
    goto memory_error;

  if (*sharing == SHARING_LAYERWISE)
  {
    *num_unique_weights = (int*) malloc(sizeof(int) * (*num_layers));
    if (*num_unique_weights == NULL)
      goto memory_error;

    *alpha = (double*) malloc(sizeof(double) * (*num_layers));
    if (*alpha == NULL)
      goto memory_error;

    *gamma = (double*) malloc(sizeof(double) * (*num_layers));
    if (*gamma == NULL)
      goto memory_error;
  }
  else
  {
    *num_unique_weights = (int*) malloc(sizeof(int));
    if (*num_unique_weights == NULL)
      goto memory_error;

    *alpha = (double*) malloc(sizeof(double));
    if (*alpha == NULL)
      goto memory_error;

    *gamma = (double*) malloc(sizeof(double));
    if (*gamma == NULL)
      goto memory_error;
  }

  for (l = 0; l < *num_layers; l++)
  {
    //--------------------------------------------------------------------------
    // Read 'ZW'
    (*ZW)[l] = (int*) malloc(sizeof(int) * (*layout)[l] * (*layout)[l + 1]);
    if ((*ZW)[l] == NULL)
      goto memory_error;

    elements_read = fread((*ZW)[l], sizeof(int), (*layout)[l] * (*layout)[l + 1], fp);
    if (elements_read != (*layout)[l] * (*layout)[l + 1])
      goto file_error;

    //--------------------------------------------------------------------------
    // Read 'Zb'
    (*Zb)[l] = (int*) malloc(sizeof(int) * (*layout)[l + 1]);
    if ((*Zb)[l] == NULL)
      goto memory_error;

    elements_read = fread((*Zb)[l], sizeof(int), (*layout)[l + 1], fp);
    if (elements_read != (*layout)[l + 1])
      goto file_error;

    if (*sharing == SHARING_LAYERWISE)
    {
      //------------------------------------------------------------------------
      // Read num_unique_weights
      elements_read = fread(&(*num_unique_weights)[l], sizeof(int), 1, fp);
      if (elements_read != 1)
        goto file_error;

      //------------------------------------------------------------------------
      // Read num_weights
      (*num_weights)[l] = (int*) malloc(sizeof(int) * (*num_unique_weights)[l]);
      if ((*num_weights)[l] == NULL)
        goto memory_error;

      elements_read = fread((*num_weights)[l], sizeof(int), (*num_unique_weights)[l], fp);
      if (elements_read != (*num_unique_weights)[l])
        goto file_error;

      //------------------------------------------------------------------------
      // Read W
      (*W)[l] = (double*) malloc(sizeof(double) * (*num_unique_weights)[l]);
      if ((*W)[l] == NULL)
        goto memory_error;

      elements_read = fread((*W)[l], sizeof(double), (*num_unique_weights)[l], fp);
      if (elements_read != (*num_unique_weights)[l])
        goto file_error;

      //------------------------------------------------------------------------
      // Read alpha
      elements_read = fread(&(*alpha)[l], sizeof(double), 1, fp);
      if (elements_read != 1)
        goto file_error;

      //------------------------------------------------------------------------
      // Read gamma
      elements_read = fread(&(*gamma)[l], sizeof(double), 1, fp);
      if (elements_read != 1)
        goto file_error;
    }
  }

  if (*sharing == SHARING_GLOBAL)
  {
    //------------------------------------------------------------------------
    // Read num_unique_weights
    elements_read = fread(*num_unique_weights, sizeof(int), 1, fp);
    if (elements_read != 1)
      goto file_error;

    //------------------------------------------------------------------------
    // Read num_weights
    (*num_weights)[0] = (int*) malloc(sizeof(int) * (*num_unique_weights)[0]);
    if ((*num_weights)[0] == NULL)
      goto memory_error;

    elements_read = fread((*num_weights)[0], sizeof(int), (*num_unique_weights)[0], fp);
    if (elements_read != (*num_unique_weights)[0])
      goto file_error;

    //------------------------------------------------------------------------
    // Read W
    (*W)[0] = (double*) malloc(sizeof(double) * (*num_unique_weights)[0]);
    if ((*W)[0] == NULL)
      goto memory_error;

    elements_read = fread((*W)[0], sizeof(double), (*num_unique_weights)[0], fp);
    if (elements_read != (*num_unique_weights)[0])
      goto file_error;

    //------------------------------------------------------------------------
    // Read alpha
    elements_read = fread(*alpha, sizeof(double), 1, fp);
    if (elements_read != 1)
      goto file_error;

    //------------------------------------------------------------------------
    // Read gamma
    elements_read = fread(*gamma, sizeof(double), 1, fp);
    if (elements_read != 1)
      goto file_error;

    for (l = 1; l < *num_layers; l++)
    {
      (*W)[l] = (*W)[0];
      (*num_weights)[l] = (*num_weights)[0];
    }
  }

  elements_read = fread(beta, sizeof(double), 1, fp);
  if (elements_read != 1)
    goto file_error;


  // Check if we are at the end of the file
  elements_read = fread(dummy, sizeof(char), 1024, fp);
  if (!feof(fp) || elements_read != 0)
  {
    _printf("File is not over. At least %d more bytes in the file.\n", elements_read);
    goto file_error;
  }

  goto cleanup;

memory_error:
  error = 1;
  _printf("Error in 'load_model': Memory error\n");
file_error:
  if (!error)
  {
    error = 2;
    _printf("Error in 'load_model': File error\n");
  }

  free_model(*num_layers, *sharing, layout, ZW, Zb, W, num_weights, num_unique_weights, alpha, gamma);

cleanup:

  if (fp != NULL)
    fclose(fp);

  return error;
}

int load_data(const char* file, double** x, double** t, int* D_x, int* D_t,
    int* N)
{
  int error = 0;
  char dummy = 0;
  size_t elements_read = 0;

  FILE* fp = NULL;

  *x = NULL;
  *t = NULL;

  fp = fopen(file, "r");
  if (fp == NULL)
    goto file_error;

  elements_read = fread(D_x, sizeof(int), 1, fp);
  if (elements_read != 1)
    goto file_error;

  elements_read = fread(D_t, sizeof(int), 1, fp);
  if (elements_read != 1)
    goto file_error;

  elements_read = fread(N, sizeof(int), 1, fp);
  if (elements_read != 1)
    goto file_error;

  *x = (double*) malloc(sizeof(double) * (*N) * (*D_x));
  if (*x == NULL)
    goto memory_error;

  *t = (double*) malloc(sizeof(double) * (*N) * (*D_t));
  if (*t == NULL)
    goto memory_error;

  elements_read = fread(*x, sizeof(double), (*N) * (*D_x), fp);
  if (elements_read != (*N) * (*D_x))
    goto file_error;

  elements_read = fread(*t, sizeof(double), (*N) * (*D_t), fp);
  if (elements_read != (*N) * (*D_t))
    goto file_error;

  // Check if we are at the end of the file
  elements_read = fread(&dummy, sizeof(char), 1, fp);
  if (!feof(fp) || elements_read != 0)
    goto file_error;

  goto cleanup;

memory_error:
file_error:
  error = 1;

  if (*x != NULL)
  {
    free(*x);
    *x = NULL;
  }
  if (*t != NULL)
  {
    free(*t);
    *t = NULL;
  }

cleanup:

  if (fp != NULL)
    fclose(fp);

  return error;
}
