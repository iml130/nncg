
#if defined(__clang__)
# pragma clang diagnostic push
# pragma clang diagnostic ignored "-Wunused-result"
# pragma clang diagnostic ignored "-Wunused-variable"
# pragma clang diagnostic ignored "-Wconversion"
# pragma clang diagnostic ignored "-Wmissing-braces"
# pragma clang diagnostic ignored "-Wfloat-conversion"
#endif

#if defined(_MSC_VER)
# pragma warning(push, 0)
#endif
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <immintrin.h>
#include <math.h>
void cnn_(float *x_in, float *scores)
{
	__m128 w, x, y, y2, t, t2;
unsigned char buf alignas(16) [16];
__m128i qw, qx, qt1, qt2, lo, hi, sum1, sum2, sum3;
int res alignas(16) [4];
	float x0[36][18][1];
	for (int xi = 0; xi < 36; xi += 1)
	{
	x0[xi][0][0] = x_in[xi * 18 * 1 + 0 * 1 + 0] - 0.470134091414344f;
	x0[xi][1][0] = x_in[xi * 18 * 1 + 1 * 1 + 0] - 0.470134091414344f;
	x0[xi][2][0] = x_in[xi * 18 * 1 + 2 * 1 + 0] - 0.470134091414344f;
	x0[xi][3][0] = x_in[xi * 18 * 1 + 3 * 1 + 0] - 0.470134091414344f;
	x0[xi][4][0] = x_in[xi * 18 * 1 + 4 * 1 + 0] - 0.470134091414344f;
	x0[xi][5][0] = x_in[xi * 18 * 1 + 5 * 1 + 0] - 0.470134091414344f;
	x0[xi][6][0] = x_in[xi * 18 * 1 + 6 * 1 + 0] - 0.470134091414344f;
	x0[xi][7][0] = x_in[xi * 18 * 1 + 7 * 1 + 0] - 0.470134091414344f;
	x0[xi][8][0] = x_in[xi * 18 * 1 + 8 * 1 + 0] - 0.470134091414344f;
	x0[xi][9][0] = x_in[xi * 18 * 1 + 9 * 1 + 0] - 0.470134091414344f;
	x0[xi][10][0] = x_in[xi * 18 * 1 + 10 * 1 + 0] - 0.470134091414344f;
	x0[xi][11][0] = x_in[xi * 18 * 1 + 11 * 1 + 0] - 0.470134091414344f;
	x0[xi][12][0] = x_in[xi * 18 * 1 + 12 * 1 + 0] - 0.470134091414344f;
	x0[xi][13][0] = x_in[xi * 18 * 1 + 13 * 1 + 0] - 0.470134091414344f;
	x0[xi][14][0] = x_in[xi * 18 * 1 + 14 * 1 + 0] - 0.470134091414344f;
	x0[xi][15][0] = x_in[xi * 18 * 1 + 15 * 1 + 0] - 0.470134091414344f;
	x0[xi][16][0] = x_in[xi * 18 * 1 + 16 * 1 + 0] - 0.470134091414344f;
	x0[xi][17][0] = x_in[xi * 18 * 1 + 17 * 1 + 0] - 0.470134091414344f;
	}
	static float x1 alignas(16) [36][18][4] = {0};
	for (int i = 0; i < 36; i += 1)
	{
		for (int j = 0; j < 18; j += 1)
		{
			x1[i][j][0] = 0.22249829769134521f;
			x1[i][j][1] = 0.06990230083465576f;
			x1[i][j][2] = 0.2096979171037674f;
			x1[i][j][3] = -0.012187456712126732f;
		}
	}
	for (int ix = -1; ix < 35; ix += 1)
	{
		int x_1, x_out_1;
		x_out_1 = (ix + 1) / 1;
		for (int jx = -1; jx < 17; jx += 1)
		{
			int x_2, x_out_2;
			x_out_2 = (jx + 1) / 1;
			x_1 = ix + 0;
			if (x_1 >= 0 && x_1 < 36)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 18)
				{

					w = _mm_set_ps(-0.15198607742786407f, -0.2241060435771942f, -0.38813042640686035f, 0.1883493810892105f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 18)
				{

					w = _mm_set_ps(-0.27417755126953125f, 0.039674751460552216f, -0.2535991966724396f, -0.12116928398609161f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 18)
				{

					w = _mm_set_ps(0.30265188217163086f, 0.028574416413903236f, -0.2890501320362091f, -0.1743050217628479f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
			}
			x_1 = ix + 1;
			if (x_1 >= 0 && x_1 < 36)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 18)
				{

					w = _mm_set_ps(-0.2098965346813202f, -0.38067927956581116f, 0.10695474594831467f, 0.3916192054748535f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 18)
				{

					w = _mm_set_ps(-0.32650256156921387f, -0.148119255900383f, -0.5057913661003113f, 0.44338369369506836f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 18)
				{

					w = _mm_set_ps(0.39357879757881165f, 0.4057707190513611f, -0.46066513657569885f, -0.014672879129648209f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
			}
			x_1 = ix + 2;
			if (x_1 >= 0 && x_1 < 36)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 18)
				{

					w = _mm_set_ps(-0.23062588274478912f, -0.6404035687446594f, 0.535746157169342f, 0.0013758891727775335f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 18)
				{

					w = _mm_set_ps(0.26750001311302185f, -0.638961672782898f, 0.37385568022727966f, -0.10604061931371689f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 18)
				{

					w = _mm_set_ps(0.1663092076778412f, -0.48616454005241394f, -0.12248460948467255f, 0.5190811157226562f);
					x = _mm_load_ps1(&x0[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x1[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x1[x_out_1][x_out_2][0], x);
				}
			}
		}
	}
	for (int i = 0; i < 36; i += 1)
	{
		for (int j = 0; j < 18; j += 1)
		{

			x = _mm_load_ps((float*)&x1[i][j][0]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x1[i][j][0], x);
		}
	}
	static float x2[18][9][4] = {0};
	for (int ix = 0; ix < 35; ix += 2)
	{
		int x_1, x_out_1;
		x_out_1 = ix / 2;
	for (int jx = 0; jx < 17; jx += 2)
	{
		int x_2, x_out_2;
		x_out_2 = jx / 2;
		x = _mm_load_ps((float*)&x1[ix][jx][0]);
		y = _mm_load_ps((float*)&x1[ix + 0][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x1[ix + 0][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
		y = _mm_load_ps((float*)&x1[ix + 1][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x1[ix + 1][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x2[x_out_1][x_out_2][0], x);
		}
	}
	static float x3 alignas(16) [18][9][8] = {0};
	for (int i = 0; i < 18; i += 1)
	{
		for (int j = 0; j < 9; j += 1)
		{
			x3[i][j][0] = -0.13709181547164917f;
			x3[i][j][1] = 0.18268336355686188f;
			x3[i][j][2] = -0.11256289482116699f;
			x3[i][j][3] = -0.07205960154533386f;
			x3[i][j][4] = -0.04556238651275635f;
			x3[i][j][5] = -0.1564091295003891f;
			x3[i][j][6] = 0.11967994272708893f;
			x3[i][j][7] = 0.15809716284275055f;
		}
	}
	for (int ix = -1; ix < 17; ix += 1)
	{
		int x_1, x_out_1;
		x_out_1 = (ix + 1) / 1;
		for (int jx = -1; jx < 8; jx += 1)
		{
			int x_2, x_out_2;
			x_out_2 = (jx + 1) / 1;
			x_1 = ix + 0;
			if (x_1 >= 0 && x_1 < 18)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 9)
				{

					w = _mm_set_ps(0.22607570886611938f, -0.11077295243740082f, -0.08831577003002167f, -0.08930544555187225f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.29224705696105957f, -0.03112337365746498f, 0.4521279036998749f, 0.06321551650762558f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.01447263266891241f, 0.13241423666477203f, 0.06826398521661758f, -0.2050580382347107f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.06979723274707794f, -0.22407861053943634f, 0.02069373056292534f, -0.06716768443584442f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.15024976432323456f, -0.05192035064101219f, 0.34343811869621277f, 0.17421288788318634f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.04658420756459236f, 0.06295877695083618f, -0.17811115086078644f, 0.16878293454647064f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.05441934987902641f, 0.17193719744682312f, 0.27003103494644165f, 0.3506952226161957f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.0713929533958435f, -0.1493120640516281f, 0.2087768167257309f, 0.07805846631526947f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 9)
				{

					w = _mm_set_ps(0.08544091880321503f, -0.22442542016506195f, 0.307867169380188f, 0.21341022849082947f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.3099980354309082f, 0.28722310066223145f, -0.03730282559990883f, -0.05714372545480728f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.10060632228851318f, 0.24454161524772644f, 0.11407576501369476f, -0.045637380331754684f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.20294035971164703f, -0.011054682545363903f, 0.5465822815895081f, -0.1971655935049057f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.1755531132221222f, -0.024637991562485695f, 0.09856485575437546f, -0.04490513727068901f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1925058811903f, 0.10229521244764328f, -0.14342214167118073f, 0.09686514735221863f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.02874353528022766f, -0.25051596760749817f, -0.014097221195697784f, 0.128767728805542f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1485479474067688f, -0.413900762796402f, -0.11912482231855392f, 0.17116738855838776f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 9)
				{

					w = _mm_set_ps(-0.1779295802116394f, -0.04528253152966499f, 0.40264084935188293f, 0.1338624507188797f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.015617853961884975f, 0.2896066904067993f, -0.34721043705940247f, -0.12923525273799896f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.027982937172055244f, 0.05798114091157913f, -0.040574852377176285f, -0.06997004896402359f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.2047353833913803f, -0.31141868233680725f, -0.03203221783041954f, 0.15023928880691528f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.2604065537452698f, 0.12685778737068176f, -0.014229143038392067f, 0.11774703860282898f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.34316155314445496f, -0.37257620692253113f, 0.06613383442163467f, -0.11956408619880676f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.028933608904480934f, -0.2275354415178299f, 0.26256614923477173f, 0.01388233620673418f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.24320973455905914f, -0.310310035943985f, 0.364945650100708f, 0.04160091653466225f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);
				}
			}
			x_1 = ix + 1;
			if (x_1 >= 0 && x_1 < 18)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 9)
				{

					w = _mm_set_ps(0.1823887825012207f, 0.12937869131565094f, -0.2298927754163742f, 0.07066181302070618f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1329362392425537f, 0.24197432398796082f, 0.16940179467201233f, -0.07630990445613861f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.31878387928009033f, 0.05231841281056404f, -0.07153085619211197f, 0.26799848675727844f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.23259221017360687f, 0.21894471347332f, 0.0681588426232338f, 0.18393994867801666f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.09404727071523666f, 0.052260346710681915f, 0.07392102479934692f, -0.27591240406036377f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.23434685170650482f, -0.013602098450064659f, 0.07156242430210114f, 0.10417857021093369f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.06912987679243088f, 0.027043798938393593f, 0.2544792890548706f, 0.440534383058548f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.11856035143136978f, -0.1221865862607956f, 0.020236607640981674f, 0.24174439907073975f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 9)
				{

					w = _mm_set_ps(0.07366803288459778f, 0.23340265452861786f, 0.24363893270492554f, 0.07299742102622986f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.023196155205368996f, 0.20477089285850525f, 0.4102344810962677f, -0.0751860961318016f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.1636751890182495f, 0.18952323496341705f, -0.021070828661322594f, -0.10520024597644806f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.09376496076583862f, -0.11345937848091125f, 0.18449579179286957f, -0.16869831085205078f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.1406385451555252f, 0.1626332551240921f, 0.17004267871379852f, -0.11453918367624283f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.23383571207523346f, 0.22110585868358612f, -0.2763058841228485f, 0.12252837419509888f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.002219957299530506f, 0.053940314799547195f, -0.25163471698760986f, -0.08707649260759354f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.24269749224185944f, 0.021841762587428093f, 0.1605965942144394f, 0.3474492132663727f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 9)
				{

					w = _mm_set_ps(0.07476826757192612f, -0.18247105181217194f, 0.015387098304927349f, -0.0718783438205719f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.0630604699254036f, -0.10260435938835144f, 0.19794413447380066f, 0.18969883024692535f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.22114121913909912f, 0.2864713668823242f, 0.2527497708797455f, -0.0979309231042862f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.2253585159778595f, 0.019522806629538536f, -0.03503929451107979f, -0.2324349284172058f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.10060795396566391f, 0.1907617449760437f, 0.10146711021661758f, -0.001419443404302001f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.2974385917186737f, 0.2486482858657837f, -0.05381966754794121f, -0.10596901178359985f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.013248246163129807f, 0.18511037528514862f, 0.2572922110557556f, -0.19977065920829773f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.06655257940292358f, -0.12234117835760117f, -0.31050950288772583f, -0.2259102463722229f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);
				}
			}
			x_1 = ix + 2;
			if (x_1 >= 0 && x_1 < 18)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 9)
				{

					w = _mm_set_ps(0.026884570717811584f, 0.07154315710067749f, -0.2319313883781433f, 0.2785215973854065f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.24411413073539734f, 0.02353391982614994f, 0.009736672043800354f, 0.19012980163097382f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.5139055848121643f, 0.3483508825302124f, -0.40753012895584106f, 0.06296542286872864f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.02055070921778679f, -0.05644723027944565f, 0.39385947585105896f, 0.3216544985771179f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.24807961285114288f, 0.21482886373996735f, -0.15515367686748505f, -0.07618950307369232f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.06926098465919495f, 0.37816205620765686f, 0.36340227723121643f, 0.2789921164512634f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.13138438761234283f, 0.25156494975090027f, -0.237801656126976f, 0.013306362554430962f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.3672976791858673f, -0.08772572129964828f, 0.01675599068403244f, 0.24988953769207f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 9)
				{

					w = _mm_set_ps(0.0213750209659338f, 0.1302015334367752f, 0.0233576949685812f, -0.019912326708436012f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.10267069190740585f, -0.030284760519862175f, 0.19161392748355865f, 0.008967460133135319f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.3901565670967102f, -0.04315111041069031f, -0.30076467990875244f, 0.3335734009742737f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.09096479415893555f, -0.05789647623896599f, -0.12625160813331604f, 0.04880235716700554f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.5747390389442444f, 0.4274086654186249f, -0.337436705827713f, 0.32770901918411255f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.15938116610050201f, 0.07368305325508118f, 0.3539034128189087f, 0.20407511293888092f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.47191452980041504f, 0.11807707697153091f, -0.28272131085395813f, 0.48730871081352234f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.18956071138381958f, -0.22584407031536102f, -0.013173959217965603f, 0.30398133397102356f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 9)
				{

					w = _mm_set_ps(-0.09310212731361389f, 0.0999189019203186f, -0.2703644931316376f, 0.04490489885210991f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1662892997264862f, 0.2278016209602356f, -0.14322759211063385f, 0.32541683316230774f);
					x = _mm_load_ps1(&x2[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.16565030813217163f, 0.41950395703315735f, -0.13639633357524872f, 0.015819255262613297f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.048464104533195496f, 0.17500190436840057f, -0.13731436431407928f, -0.4909614622592926f);
					x = _mm_load_ps1(&x2[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.24968962371349335f, 0.06072467938065529f, -0.2417452335357666f, 0.08707252144813538f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.2605587840080261f, -0.1290767341852188f, -0.22482280433177948f, -0.026280641555786133f);
					x = _mm_load_ps1(&x2[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.24344339966773987f, 0.1978120505809784f, 0.07088275253772736f, 0.44296902418136597f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.18306323885917664f, -0.08479826897382736f, -0.48422500491142273f, -0.14805984497070312f);
					x = _mm_load_ps1(&x2[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x3[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x3[x_out_1][x_out_2][4], x);
				}
			}
		}
	}
	for (int i = 0; i < 18; i += 1)
	{
		for (int j = 0; j < 9; j += 1)
		{

			x = _mm_load_ps((float*)&x3[i][j][0]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x3[i][j][0], x);

			x = _mm_load_ps((float*)&x3[i][j][4]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x3[i][j][4], x);
		}
	}
	static float x4[9][4][8] = {0};
	for (int ix = 0; ix < 17; ix += 2)
	{
		int x_1, x_out_1;
		x_out_1 = ix / 2;
	for (int jx = 0; jx < 8; jx += 2)
	{
		int x_2, x_out_2;
		x_out_2 = jx / 2;
		x = _mm_load_ps((float*)&x3[ix][jx][0]);
		y = _mm_load_ps((float*)&x3[ix + 0][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x3[ix + 0][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);
		y = _mm_load_ps((float*)&x3[ix + 1][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x3[ix + 1][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x4[x_out_1][x_out_2][0], x);
		x = _mm_load_ps((float*)&x3[ix][jx][4]);
		y = _mm_load_ps((float*)&x3[ix + 0][jx + 0][4]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x3[ix + 0][jx + 1][4]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
		y = _mm_load_ps((float*)&x3[ix + 1][jx + 0][4]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x3[ix + 1][jx + 1][4]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x4[x_out_1][x_out_2][4], x);
		}
	}
	static float x5 alignas(16) [9][4][16] = {0};
	for (int i = 0; i < 9; i += 1)
	{
		for (int j = 0; j < 4; j += 1)
		{
			x5[i][j][0] = 0.1237221360206604f;
			x5[i][j][1] = 0.12151692807674408f;
			x5[i][j][2] = -0.059937287122011185f;
			x5[i][j][3] = -0.09580352157354355f;
			x5[i][j][4] = 0.08119948953390121f;
			x5[i][j][5] = 0.06620238721370697f;
			x5[i][j][6] = 0.07512454688549042f;
			x5[i][j][7] = -0.0513504296541214f;
			x5[i][j][8] = -0.01908137835562229f;
			x5[i][j][9] = 0.03639838472008705f;
			x5[i][j][10] = 0.03474555164575577f;
			x5[i][j][11] = 0.02234283648431301f;
			x5[i][j][12] = 0.08998905122280121f;
			x5[i][j][13] = 0.10293158888816833f;
			x5[i][j][14] = -0.09580739587545395f;
			x5[i][j][15] = -0.12486428767442703f;
		}
	}
	for (int ix = -1; ix < 8; ix += 1)
	{
		int x_1, x_out_1;
		x_out_1 = (ix + 1) / 1;
		for (int jx = -1; jx < 3; jx += 1)
		{
			int x_2, x_out_2;
			x_out_2 = (jx + 1) / 1;
			x_1 = ix + 0;
			if (x_1 >= 0 && x_1 < 9)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(-0.016234109178185463f, -0.12190172076225281f, 0.07132884114980698f, -0.3249861001968384f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.03394593298435211f, 0.012361474335193634f, -0.10296841710805893f, -0.0937560424208641f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.25593331456184387f, -0.031923599541187286f, -0.020904753357172012f, 0.04213723912835121f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.022856544703245163f, 0.05313870310783386f, -0.21417436003684998f, -0.2532167434692383f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.012349560856819153f, 0.07528483122587204f, -0.051161762326955795f, -0.04975141957402229f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.08212176710367203f, 0.06165098398923874f, 0.21535176038742065f, 0.0428910069167614f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.03756805881857872f, -0.020528580993413925f, 0.11067771911621094f, 0.04290521517395973f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.09960295259952545f, 0.15809635818004608f, 0.09573705494403839f, -0.30272147059440613f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.06767509877681732f, 0.03696572780609131f, 0.07305682450532913f, 0.05135713517665863f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.19327066838741302f, -0.036173440515995026f, 0.10350291430950165f, -0.026917044073343277f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.15369419753551483f, 0.12409265339374542f, -0.1102803647518158f, -0.13416650891304016f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.15964947640895844f, -0.09245456755161285f, 0.14048029482364655f, -0.09037982672452927f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.018006091937422752f, -0.007558665703982115f, 0.05383739247918129f, -0.01986766792833805f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.06775424629449844f, 0.16895413398742676f, -0.002252332167699933f, 0.056193362921476364f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.07085008919239044f, 0.02082635648548603f, 0.08277179300785065f, -0.20624111592769623f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.12180424481630325f, -0.09059931337833405f, 0.1175680011510849f, -0.15682782232761383f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.0730074942111969f, 0.12910322844982147f, 0.1259407103061676f, -0.11076691001653671f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.12546145915985107f, -0.1553126573562622f, 0.036149147897958755f, -0.10820714384317398f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.06393975019454956f, -0.2719925045967102f, 0.05035192891955376f, 0.004322339314967394f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.0407940149307251f, 0.1075783297419548f, -0.25588473677635193f, -0.20855098962783813f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.015278940089046955f, 0.0803414061665535f, -0.06852651387453079f, -0.06536860018968582f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.23325380682945251f, -0.08577457070350647f, -0.1260520964860916f, -0.0711742639541626f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.20414979755878448f, 0.2428598701953888f, -0.11341860145330429f, 0.01622624695301056f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.1678100973367691f, -0.21242043375968933f, 0.05227309837937355f, -0.22326593101024628f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.009963063523173332f, 0.2013590782880783f, -0.1715070754289627f, -0.2806216776371002f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.0328122116625309f, -0.1461302638053894f, 0.047411177307367325f, 0.030864538624882698f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.06652243435382843f, -0.05898844823241234f, 0.036739204078912735f, -0.09813136607408524f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.02821304090321064f, 0.04942840710282326f, 0.02252531610429287f, -0.21957245469093323f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.16875964403152466f, -0.0476665124297142f, 0.07909126579761505f, -0.055805426090955734f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.013242768123745918f, 0.09747844934463501f, 0.011283157393336296f, 0.21472463011741638f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.23309828341007233f, 0.12885721027851105f, 0.09248917549848557f, 0.11619171500205994f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.1776265799999237f, -0.03568883240222931f, -0.016878601163625717f, 0.0005271919071674347f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(0.09951184689998627f, -0.013631554320454597f, 0.03488534316420555f, -0.19267933070659637f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.141974538564682f, -0.01931893453001976f, -0.3086896240711212f, -0.11130780726671219f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.22237704694271088f, 0.11467544734477997f, 0.06419304013252258f, -0.024103308096528053f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.04981902614235878f, 0.18552833795547485f, -0.10004738718271255f, 0.049101799726486206f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.26686516404151917f, 0.03683452680706978f, -0.07031533122062683f, 0.19137486815452576f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1488097906112671f, -0.13855424523353577f, 0.1587439775466919f, 0.2543276846408844f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.18608300387859344f, 0.05368508771061897f, -0.13192090392112732f, -0.0866224616765976f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.2691154181957245f, 0.15762396156787872f, 0.048533156514167786f, -0.06379657983779907f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.11090318113565445f, 0.013570331037044525f, -0.14240606129169464f, 0.1393030881881714f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.05538591369986534f, 0.18478959798812866f, 0.16041387617588043f, 0.08195050060749054f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.057412415742874146f, 0.11112764477729797f, 0.04572392627596855f, 0.12080082297325134f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.18774673342704773f, 0.04552771523594856f, 0.10013395547866821f, 0.05705679953098297f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.12347860634326935f, -0.1364801675081253f, -0.15800631046295166f, 0.11618505418300629f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.10717332363128662f, 0.18894828855991364f, 0.024010704830288887f, 0.048958200961351395f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.03883105516433716f, -0.14718970656394958f, 0.08907093852758408f, -0.05685938522219658f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.009536072611808777f, -0.15131495893001556f, -0.09333064407110214f, 0.09017110615968704f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.08840540796518326f, 0.03670191764831543f, 0.02231864631175995f, -0.1174875870347023f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.003021670039743185f, -0.007762469816952944f, 0.1341056525707245f, 0.017733458429574966f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.07849393784999847f, -0.20727629959583282f, -0.11068599671125412f, -0.10114800184965134f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.024643296375870705f, 0.07396677136421204f, 0.09249058365821838f, -0.13467665016651154f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.014261573553085327f, -0.1513877660036087f, -0.1299092024564743f, -0.21597164869308472f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.06360956281423569f, -0.12869322299957275f, -0.040508437901735306f, -0.15677589178085327f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.12647899985313416f, -0.009653501212596893f, -0.10115396231412888f, -0.0469072125852108f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.04948235675692558f, 0.3186492323875427f, -0.16409911215305328f, -0.062127724289894104f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.06159565970301628f, 0.0009653886663727462f, -0.2936362624168396f, -0.2476653903722763f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.085232213139534f, -0.36037734150886536f, -0.05403885245323181f, -0.2318384051322937f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.03615871071815491f, -0.3891432583332062f, -0.32099559903144836f, -0.056581463664770126f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.1285555511713028f, -0.20945042371749878f, -0.1762600690126419f, -0.19847756624221802f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.0886705219745636f, -0.18156887590885162f, -0.23766429722309113f, -0.18459485471248627f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.15173137187957764f, -0.22157184779644012f, -0.21055233478546143f, 0.19193598628044128f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.034993767738342285f, 0.04295480251312256f, -0.03919004648923874f, -0.05310690402984619f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.0218877661973238f, 0.012538529001176357f, -0.0111236572265625f, -0.04681912809610367f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(0.0423133485019207f, -0.03135889396071434f, 0.1485026329755783f, 0.1327352523803711f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.04326585307717323f, 0.021389160305261612f, -0.20680053532123566f, 0.19476741552352905f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.30848562717437744f, -0.08357617259025574f, -0.02588050439953804f, -0.11051729321479797f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.14749330282211304f, 0.09252441674470901f, -0.16946765780448914f, 0.044837065041065216f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.09665317833423615f, 0.07750890403985977f, 0.08335922658443451f, -0.23149710893630981f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.04942179471254349f, 0.27649617195129395f, -0.06686173379421234f, -0.22644060850143433f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.06173958256840706f, -0.0676117092370987f, 0.06751731783151627f, -0.1939191371202469f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.26231759786605835f, 0.3582571744918823f, -0.04865490645170212f, 0.07695876806974411f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.21034123003482819f, -0.03640848398208618f, 0.02297218330204487f, 0.09255833178758621f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1559368073940277f, 0.15825794637203217f, -0.1325828582048416f, 0.22932755947113037f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.03258343040943146f, 0.22596606612205505f, 0.12975935637950897f, 0.055857133120298386f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.12491107732057571f, 0.06787016987800598f, 0.15037599205970764f, 0.2008882313966751f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.06971713155508041f, 0.05398308113217354f, 0.15506711602210999f, 0.24648883938789368f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.061199069023132324f, -0.0810488611459732f, 0.02926626242697239f, -0.016761325299739838f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.024840019643306732f, -0.04709390923380852f, -0.018427466973662376f, 0.026327136904001236f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.10311730206012726f, 0.005160795524716377f, 0.1001073345541954f, 0.0680646076798439f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.16478373110294342f, 0.040967267006635666f, 0.14414939284324646f, -0.051506705582141876f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.02740822546184063f, 0.07499983161687851f, -0.27728357911109924f, -0.08133052289485931f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.24158410727977753f, 0.12312832474708557f, 0.16322654485702515f, -0.1404053121805191f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.18488727509975433f, 0.045889753848314285f, -0.17600660026073456f, -0.25291866064071655f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.23519283533096313f, -0.05931099131703377f, -0.005122136790305376f, -0.05181710794568062f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1514403074979782f, -0.02093541994690895f, -0.300896018743515f, -0.0801619291305542f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.20027056336402893f, -0.3097259998321533f, 0.03735358640551567f, -0.17821599543094635f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.126076802611351f, 0.41553157567977905f, -0.25255510210990906f, -0.21796812117099762f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.04725664481520653f, 0.07810603082180023f, 0.09097238630056381f, -0.16523505747318268f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.0932183638215065f, -0.10656516999006271f, 0.10614973306655884f, -0.10125647485256195f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.21242447197437286f, -0.1298581212759018f, -0.22007423639297485f, 0.11653709411621094f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.141861230134964f, -0.1718762069940567f, -0.2027653604745865f, -0.17730242013931274f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.2694646418094635f, 0.10709407180547714f, 0.031116671860218048f, 0.03954607993364334f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.058000046759843826f, -0.1192534863948822f, 0.08596333116292953f, -0.0879463329911232f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.0661100372672081f, -0.20344893634319305f, -0.21338465809822083f, -0.04784102737903595f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.02546405792236328f, 0.02375025488436222f, 0.04234425723552704f, 0.12962529063224792f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);
				}
			}
			x_1 = ix + 1;
			if (x_1 >= 0 && x_1 < 9)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(0.1626684069633484f, -0.048568665981292725f, -0.17875586450099945f, -0.3712763488292694f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.11744768917560577f, -0.14928393065929413f, -0.05163872241973877f, -0.14954468607902527f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.2567882537841797f, 0.0663008838891983f, -0.10764319449663162f, -0.01428995281457901f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.13925504684448242f, 0.11115875840187073f, -0.24644844233989716f, 0.043793510645627975f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.19759254157543182f, -0.025713784620165825f, 0.0910557359457016f, 0.06867864727973938f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.13512898981571198f, 0.07875364273786545f, 0.22677898406982422f, -0.06479812413454056f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.2389332503080368f, 0.07402384281158447f, 0.29615136981010437f, 0.16859276592731476f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.0201315488666296f, 0.012398824095726013f, -0.0771351084113121f, 0.06785861402750015f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.09625338017940521f, -0.022125396877527237f, -0.024572137743234634f, -0.014129179529845715f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.09270308911800385f, -0.07977014780044556f, -0.11240647733211517f, -0.0806940570473671f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.07304373383522034f, -0.27511316537857056f, 0.05191974341869354f, 0.01534323114901781f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.11138900369405746f, 0.08511850982904434f, -0.027000799775123596f, 0.22189845144748688f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.020155547186732292f, -0.17480908334255219f, -0.0846560150384903f, 0.09447365999221802f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.09997180849313736f, 0.15202204883098602f, 0.08767606317996979f, -0.338413804769516f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.10058172792196274f, 0.018145576119422913f, 0.06684752553701401f, 0.00645312899723649f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.14743749797344208f, -0.08013495057821274f, -0.21290646493434906f, 0.02524711936712265f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.11185719072818756f, -0.042534418404102325f, -0.03385637328028679f, -0.3531649112701416f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.015257050283253193f, 0.2020006775856018f, 0.10528521984815598f, -0.14001691341400146f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.0011750957928597927f, -0.082459457218647f, -0.12759821116924286f, -0.03569073975086212f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.03192184492945671f, -0.07026053965091705f, -0.21971222758293152f, -0.05698302760720253f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.038808513432741165f, -0.06742026656866074f, -0.09521432965993881f, 0.14807750284671783f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.2067568153142929f, -0.11515781283378601f, -0.1599241942167282f, -0.22268125414848328f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.05162675306200981f, 0.07166021317243576f, 0.12609489262104034f, 0.12294188886880875f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.02481985278427601f, -0.21483196318149567f, -0.09855762124061584f, 0.16550175845623016f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.14500802755355835f, 0.059548519551754f, 0.3308034837245941f, 0.3628731369972229f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.07130851596593857f, 0.062488313764333725f, 0.1384422779083252f, 0.036747951060533524f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.13005965948104858f, 0.28218933939933777f, -0.015141122043132782f, 0.02547740936279297f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.19385986030101776f, -0.04902317002415657f, 0.16753503680229187f, 0.2298629879951477f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.2240622192621231f, -0.12539510428905487f, 0.1948038637638092f, 0.17880608141422272f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.037040531635284424f, 0.19880707561969757f, 0.22992706298828125f, 0.12023723125457764f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.09917189925909042f, 0.1766769140958786f, 0.006413509137928486f, -0.05816710367798805f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.09569744020700455f, 0.061036545783281326f, -0.04720854014158249f, 0.2668567895889282f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(0.013728193938732147f, 0.0077981832437217236f, 0.10341423749923706f, -0.24358230829238892f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.13108710944652557f, -0.0033721213694661856f, -0.11704697459936142f, -0.1532774716615677f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.15350152552127838f, -0.1504805088043213f, -0.08328844606876373f, -0.1563444882631302f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.004492789041250944f, -0.12022048234939575f, -0.12475118041038513f, -0.3324764668941498f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.07470257580280304f, -0.08518964052200317f, 0.27482226490974426f, 0.007149145472794771f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.2726145386695862f, 0.20164258778095245f, 0.1497577726840973f, 0.15159977972507477f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.10919202864170074f, 0.13982902467250824f, -0.14191626012325287f, 0.18422932922840118f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.2036689966917038f, -0.10725829005241394f, 0.19465473294258118f, 0.1856861114501953f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.13283248245716095f, -0.046403415501117706f, 0.08115983754396439f, -0.04267853498458862f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.16188368201255798f, -0.17596396803855896f, -0.3333974778652191f, -0.01997571997344494f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.0608874075114727f, 0.14034625887870789f, 0.02182801254093647f, 0.12082704156637192f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.03677212819457054f, -0.03931013122200966f, -0.12144067138433456f, -0.23535647988319397f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.057755403220653534f, -0.15061752498149872f, -0.10426121950149536f, 0.0818450003862381f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.16594015061855316f, -0.12620790302753448f, -0.17284949123859406f, -0.09133431315422058f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.33373770117759705f, 0.06714794784784317f, 0.09910763800144196f, -0.091134212911129f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.018913952633738518f, -0.07882807403802872f, -0.13758978247642517f, -0.15956903994083405f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.11969896405935287f, -0.03003367967903614f, 0.04783361777663231f, -0.1655673086643219f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.07633433490991592f, 0.09067988395690918f, 0.013712151907384396f, -0.08802169561386108f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.014690411277115345f, -0.19543741643428802f, 0.3030831217765808f, -0.0991118922829628f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.09019208699464798f, -0.1061098650097847f, -0.1956377476453781f, -0.2858557403087616f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.31022077798843384f, -0.04294903576374054f, -0.13054019212722778f, 0.021659737452864647f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.12336061149835587f, 0.08974906802177429f, -0.19659969210624695f, -0.18560385704040527f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.1500963568687439f, 0.11339682340621948f, 0.09737130254507065f, -0.009512812830507755f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.2368496060371399f, -0.17049278318881989f, 0.08705293387174606f, -0.07792013138532639f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.057756271213293076f, 0.11788707226514816f, 0.13349848985671997f, 0.20014341175556183f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.05513904616236687f, 0.019257280975580215f, 0.0842556282877922f, 0.2625061273574829f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.04229888319969177f, 0.10409404337406158f, 0.11815492808818817f, -0.16055764257907867f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.0009712768951430917f, -0.1352241337299347f, 0.28220894932746887f, 0.06531038880348206f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.01666831225156784f, -0.10334896296262741f, 0.2535361349582672f, -0.031487733125686646f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.08695898950099945f, -0.050848156213760376f, 0.08728361129760742f, 0.21667149662971497f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.08147645741701126f, -0.0007338192081078887f, -0.0008827901910990477f, 0.04105116426944733f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.028452545404434204f, -0.07434412837028503f, 0.19748936593532562f, 0.24408121407032013f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(0.09148428589105606f, -0.057816751301288605f, 0.041582949459552765f, 0.1215829998254776f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.0965370163321495f, 0.09713625907897949f, -0.23515766859054565f, 0.2327805608510971f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.23505526781082153f, -0.12807391583919525f, -0.032222896814346313f, 0.047216784209012985f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.1333324909210205f, -0.22644364833831787f, 0.15689612925052643f, -0.03795834630727768f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.2266518473625183f, 0.15847627818584442f, 0.12660999596118927f, 0.23498131334781647f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.07329563796520233f, 0.1588854193687439f, 0.07415341585874557f, 0.1600974202156067f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.023978974670171738f, 0.1742957979440689f, 0.12682899832725525f, 0.009003320708870888f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.22933240234851837f, -0.006619851570576429f, 0.017239047214388847f, 0.17022325098514557f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.13624517619609833f, -0.03415505588054657f, 0.17211049795150757f, -0.014778096228837967f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.1526096910238266f, 0.23305808007717133f, -0.036472998559474945f, 0.02080429531633854f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.05661711469292641f, 0.019464761018753052f, 0.1141495481133461f, 0.03869277983903885f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.09982910007238388f, 0.04915892705321312f, 0.05013133957982063f, 0.22459132969379425f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.013908788561820984f, -0.10044132173061371f, 0.07373436540365219f, 0.05582040920853615f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.08346447348594666f, 0.0564488060772419f, -0.05641448497772217f, -0.05142047256231308f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.2226104438304901f, 0.0824696347117424f, 0.05129119008779526f, 0.1257622241973877f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.09236686676740646f, -0.21780142188072205f, 0.07556068897247314f, 0.13748590648174286f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.253751665353775f, -0.11490815877914429f, 0.10040482133626938f, -0.18108844757080078f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.03796815127134323f, 0.19588401913642883f, -0.33732157945632935f, 0.14222556352615356f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.27214840054512024f, 0.0017436228226870298f, 0.06541287899017334f, 0.02757299318909645f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.02172924019396305f, -0.30141013860702515f, -0.17326906323432922f, 0.036073535680770874f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.22036729753017426f, -0.14188885688781738f, 0.33516600728034973f, -0.3026379644870758f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.018227335065603256f, 0.2478369027376175f, -0.18341003358364105f, 0.014453524723649025f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.04385610669851303f, -0.018981367349624634f, 0.20791350305080414f, -0.03378918394446373f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.007136792875826359f, 0.07107190042734146f, -0.05991358682513237f, -0.14701761305332184f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.10470729321241379f, -0.045170288532972336f, -0.12755650281906128f, -0.2400263249874115f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.08213762193918228f, 0.16513530910015106f, 0.029742687940597534f, -0.3423577845096588f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.10750743001699448f, -0.0016622910043224692f, 0.19819200038909912f, 0.17130355536937714f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.12368112057447433f, 0.056354861706495285f, -0.024020669981837273f, 0.13302072882652283f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.04662824049592018f, 0.05052928254008293f, -0.2919136881828308f, 0.11436482518911362f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.07557757198810577f, -0.1869203746318817f, -0.02421361766755581f, -0.07330622524023056f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.14249572157859802f, 0.10500723868608475f, 0.024650417268276215f, 0.08274875581264496f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.03441775590181351f, -0.013223189860582352f, 0.19914411008358002f, 0.3172086477279663f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);
				}
			}
			x_1 = ix + 2;
			if (x_1 >= 0 && x_1 < 9)
			{
				x_2 = jx + 0;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(-0.2644895613193512f, 0.1566508412361145f, -0.24624252319335938f, -0.17751887440681458f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.06302934139966965f, -0.186110258102417f, 0.03684452176094055f, -0.32427939772605896f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.1249229833483696f, 0.07680755853652954f, -0.054380692541599274f, -0.053053420037031174f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.14553137123584747f, 0.12468739598989487f, -0.15577495098114014f, 0.035027723759412766f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.04301430284976959f, 0.3720414936542511f, 0.14253005385398865f, 0.031788330525159836f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.28750374913215637f, 0.17148317396640778f, 0.1118086650967598f, 0.04996269568800926f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.1501016467809677f, 0.062033675611019135f, 0.18108409643173218f, 0.025379348546266556f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.03773471340537071f, 0.4486660361289978f, -0.04899151623249054f, -0.12619975209236145f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.1805652529001236f, 0.04227350279688835f, -0.05660999193787575f, -0.11645103245973587f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1514033079147339f, -0.23707041144371033f, 0.19568361341953278f, -0.16689081490039825f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.1805955022573471f, 0.04570326581597328f, -0.1777421236038208f, 0.06116043031215668f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.06581229716539383f, 0.16519953310489655f, -0.02273602969944477f, 0.045042432844638824f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.2673594057559967f, 0.22327299416065216f, -0.02497905120253563f, -0.12452283501625061f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.011106230318546295f, 0.019437992945313454f, 0.09376808255910873f, -0.061831094324588776f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.056083984673023224f, -0.008533003740012646f, -0.021749885752797127f, 0.003214888274669647f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.17688152194023132f, 0.21483732759952545f, 0.14793439209461212f, -0.01093981135636568f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.07064477354288101f, 0.02083173394203186f, -0.40393710136413574f, -0.13272999227046967f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1137848049402237f, -0.1367226392030716f, -0.023449383676052094f, -0.020137283951044083f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.00686305807903409f, -0.12556052207946777f, -0.19211530685424805f, 0.08713874220848083f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.14414790272712708f, 0.023584727197885513f, -0.1985165923833847f, -0.13585267961025238f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.05054014176130295f, 0.1935013085603714f, -0.4946722984313965f, -0.08588100224733353f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.3839125633239746f, -0.25460147857666016f, 0.09442199766635895f, -0.09235341846942902f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.15019312500953674f, -0.0021305987611413f, -0.23788779973983765f, 0.06326957046985626f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.06159317493438721f, 0.007533291820436716f, 0.019353117793798447f, 0.010718196630477905f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.03514665365219116f, 0.17390459775924683f, 0.08833740651607513f, 0.19559159874916077f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1201925128698349f, -0.18932537734508514f, -0.009139450266957283f, 0.33394014835357666f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.18207323551177979f, 0.0693916454911232f, -0.006721123121678829f, -0.014547520317137241f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.11715753376483917f, -0.017371097579598427f, 0.15077760815620422f, 0.037709251046180725f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.08860085904598236f, -0.22949129343032837f, 0.26564303040504456f, 0.25020426511764526f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.046770382672548294f, 0.3046991527080536f, 0.2310468554496765f, 0.1690782606601715f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.0013077860930934548f, 0.09405610710382462f, 0.30010277032852173f, -0.0821508914232254f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.26678910851478577f, -0.03424043208360672f, 0.017164727672934532f, -0.05974739044904709f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);
				}
				x_2 = jx + 1;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(-0.21668477356433868f, -0.0676698088645935f, -0.1043730229139328f, -0.41766828298568726f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.06396130472421646f, -0.2620542347431183f, -0.2482174038887024f, -0.3478444218635559f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.06803198903799057f, 0.13574530184268951f, -0.07337820529937744f, -0.06358493119478226f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.021926630288362503f, -0.08290880918502808f, -0.32370269298553467f, -0.39026424288749695f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.10908729583024979f, 0.29944345355033875f, 0.16743972897529602f, -0.0031821744050830603f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.04244757071137428f, 0.3200746178627014f, -0.026263829320669174f, 0.09080346673727036f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.20597492158412933f, 0.0984184518456459f, 0.20792970061302185f, -0.15097126364707947f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.04173285514116287f, 0.009958877228200436f, 0.08208294957876205f, 0.0030433444771915674f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.022752754390239716f, -0.07435603439807892f, -0.0258517824113369f, -0.12279680371284485f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.16819491982460022f, -0.1469649374485016f, -0.06963450461626053f, -0.127469003200531f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.07560926675796509f, -0.02298658899962902f, -0.1532452404499054f, -0.1839124858379364f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.09419401735067368f, 0.019515618681907654f, -0.2929300367832184f, -0.25432538986206055f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.04397213086485863f, 0.1290682703256607f, -0.18814043700695038f, -0.15467025339603424f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.18165192008018494f, -0.23877443373203278f, -0.2572348713874817f, -0.19259527325630188f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.12853209674358368f, 0.19996489584445953f, -0.23263482749462128f, 0.02458755299448967f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.2582989037036896f, 0.13552546501159668f, -0.19887065887451172f, -0.15101361274719238f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.2024690955877304f, -0.0703195184469223f, -0.09709826856851578f, -0.15126554667949677f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.10896965861320496f, 0.01572049967944622f, 0.15520863234996796f, -0.23297147452831268f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.06815598905086517f, 0.07628551125526428f, 0.021285898983478546f, -0.006132178008556366f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.2297549843788147f, -0.03730660304427147f, -0.21173664927482605f, -0.35535040497779846f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.033309243619441986f, 0.12692059576511383f, -0.1795741617679596f, -0.34178221225738525f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.03514605760574341f, -0.26561757922172546f, -0.0806206539273262f, -0.3733070194721222f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.13150615990161896f, -0.31259289383888245f, -0.13870303332805634f, -0.14278309047222137f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.09652407467365265f, 0.013759151101112366f, -0.18167074024677277f, -0.2883795201778412f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.25373375415802f, -0.12054918706417084f, -0.13585282862186432f, 0.10667005926370621f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.030019033700227737f, -0.06830912828445435f, 0.17982147634029388f, 0.14720536768436432f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.03060462884604931f, -0.12596668303012848f, -0.2364746481180191f, 0.12628304958343506f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.036452118307352066f, -0.1397680789232254f, 0.10278598964214325f, 0.079036645591259f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.030424196273088455f, -0.282406747341156f, 0.1608138233423233f, 0.29426461458206177f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.20532511174678802f, 0.2824290692806244f, -0.09317581355571747f, 0.20613713562488556f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.03534794971346855f, 0.06083746626973152f, -0.06158166378736496f, 0.05427844449877739f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.015854448080062866f, -0.01899722032248974f, 0.11932528018951416f, 0.042096097022295f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);
				}
				x_2 = jx + 2;
				if (x_2 >= 0 && x_2 < 4)
				{

					w = _mm_set_ps(0.017969390377402306f, -0.023381654173135757f, -0.1613560914993286f, -0.19076260924339294f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.19095823168754578f, -0.19079643487930298f, -0.0006844829767942429f, -0.18189910054206848f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.10807250440120697f, -0.015746520832180977f, -0.27502885460853577f, -0.05438588559627533f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.020849958062171936f, -0.12119825184345245f, 0.19301632046699524f, -0.19346509873867035f);
					x = _mm_load_ps1(&x4[x_1][x_2][0]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.027411052957177162f, 0.2842126786708832f, 0.22410765290260315f, -0.0045775361359119415f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.28860214352607727f, 0.04003719240427017f, 0.05508171766996384f, 0.13707248866558075f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.12001684308052063f, -0.058199692517519f, 0.21525247395038605f, -0.023765243589878082f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.003142858389765024f, 0.044553242623806f, 0.2921622693538666f, -0.0677696168422699f);
					x = _mm_load_ps1(&x4[x_1][x_2][1]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.21908700466156006f, -0.0017214483814314008f, -0.11239462345838547f, 0.14338621497154236f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.0019170134328305721f, -0.16135963797569275f, -0.06112820655107498f, 0.13349837064743042f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.030402833595871925f, -0.1873226761817932f, -0.2890661954879761f, -0.036003462970256805f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.17819473147392273f, 0.26924601197242737f, 0.022801930084824562f, -0.04050116240978241f);
					x = _mm_load_ps1(&x4[x_1][x_2][2]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.23584851622581482f, 0.21618087589740753f, -0.25040891766548157f, -0.042520880699157715f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.1579390913248062f, -0.2166721075773239f, -0.03352123871445656f, -0.01680917665362358f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.19295836985111237f, -0.264335572719574f, -0.11303918808698654f, -0.10013767331838608f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.011316923424601555f, 0.005909008905291557f, 0.27337709069252014f, 0.025518717244267464f);
					x = _mm_load_ps1(&x4[x_1][x_2][3]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.1450795829296112f, -0.16772906482219696f, -0.030317317694425583f, -0.2742081880569458f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.10958171635866165f, 0.16491122543811798f, -0.2562597692012787f, 0.01679929904639721f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.2172645628452301f, -0.1267915666103363f, -0.01926279254257679f, 0.17050676047801971f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.2527685761451721f, -0.08987683802843094f, -0.10898718237876892f, -0.029138684272766113f);
					x = _mm_load_ps1(&x4[x_1][x_2][4]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(-0.0908099040389061f, -0.033179450780153275f, -0.19587348401546478f, -0.08079136908054352f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.0018149956595152617f, -0.275462806224823f, 0.09427465498447418f, -0.03496591001749039f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.11633439362049103f, -0.37250956892967224f, -0.3504246771335602f, 0.029993409290909767f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.3675883114337921f, -0.10553662478923798f, -0.11183972656726837f, -0.35535314679145813f);
					x = _mm_load_ps1(&x4[x_1][x_2][5]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.08511900156736374f, 0.05890876054763794f, -0.13894709944725037f, -0.10191359370946884f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(0.054736390709877014f, -0.013531308621168137f, 0.29313597083091736f, -0.4049780070781708f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(0.22843685746192932f, -0.16703638434410095f, 0.19621770083904266f, -0.02588590234518051f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(-0.04473334550857544f, -0.013159524649381638f, -0.20094142854213715f, 0.17849819362163544f);
					x = _mm_load_ps1(&x4[x_1][x_2][6]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);

					w = _mm_set_ps(0.1479726880788803f, -0.01225103810429573f, -0.04637088626623154f, 0.17182274162769318f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][0]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][0], x);

					w = _mm_set_ps(-0.12330111116170883f, -0.05203752964735031f, 0.03235626965761185f, -0.10291454941034317f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][4]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][4], x);

					w = _mm_set_ps(-0.02259078063070774f, 0.19661125540733337f, -0.11702949553728104f, 0.1227530911564827f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][8]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][8], x);

					w = _mm_set_ps(0.09857655316591263f, -0.10586680471897125f, 0.20099154114723206f, 0.18514631688594818f);
					x = _mm_load_ps1(&x4[x_1][x_2][7]);
					y = _mm_mul_ps(w, x);
					x = _mm_load_ps((float*)&x5[x_out_1][x_out_2][12]);
					x = _mm_add_ps(x, y);
					_mm_store_ps((float*)&x5[x_out_1][x_out_2][12], x);
				}
			}
		}
	}
	for (int i = 0; i < 9; i += 1)
	{
		for (int j = 0; j < 4; j += 1)
		{

			x = _mm_load_ps((float*)&x5[i][j][0]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x5[i][j][0], x);

			x = _mm_load_ps((float*)&x5[i][j][4]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x5[i][j][4], x);

			x = _mm_load_ps((float*)&x5[i][j][8]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x5[i][j][8], x);

			x = _mm_load_ps((float*)&x5[i][j][12]);
			x = _mm_max_ps(x, _mm_setzero_ps());
			_mm_store_ps((float*)&x5[i][j][12], x);
		}
	}
	static float x6[2][2][16] = {0};
	for (int ix = 0; ix < 6; ix += 4)
	{
		int x_1, x_out_1;
		x_out_1 = ix / 4;
	for (int jx = 0; jx < 3; jx += 2)
	{
		int x_2, x_out_2;
		x_out_2 = jx / 2;
		x = _mm_load_ps((float*)&x5[ix][jx][0]);
		y = _mm_load_ps((float*)&x5[ix + 0][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 0][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);
		y = _mm_load_ps((float*)&x5[ix + 1][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 1][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);
		y = _mm_load_ps((float*)&x5[ix + 2][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 2][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);
		y = _mm_load_ps((float*)&x5[ix + 3][jx + 0][0]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 3][jx + 1][0]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][0], x);
		x = _mm_load_ps((float*)&x5[ix][jx][4]);
		y = _mm_load_ps((float*)&x5[ix + 0][jx + 0][4]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 0][jx + 1][4]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
		y = _mm_load_ps((float*)&x5[ix + 1][jx + 0][4]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 1][jx + 1][4]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
		y = _mm_load_ps((float*)&x5[ix + 2][jx + 0][4]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 2][jx + 1][4]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
		y = _mm_load_ps((float*)&x5[ix + 3][jx + 0][4]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 3][jx + 1][4]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][4], x);
		x = _mm_load_ps((float*)&x5[ix][jx][8]);
		y = _mm_load_ps((float*)&x5[ix + 0][jx + 0][8]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 0][jx + 1][8]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][8], x);
		y = _mm_load_ps((float*)&x5[ix + 1][jx + 0][8]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 1][jx + 1][8]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][8], x);
		y = _mm_load_ps((float*)&x5[ix + 2][jx + 0][8]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 2][jx + 1][8]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][8], x);
		y = _mm_load_ps((float*)&x5[ix + 3][jx + 0][8]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 3][jx + 1][8]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][8], x);
		x = _mm_load_ps((float*)&x5[ix][jx][12]);
		y = _mm_load_ps((float*)&x5[ix + 0][jx + 0][12]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 0][jx + 1][12]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][12], x);
		y = _mm_load_ps((float*)&x5[ix + 1][jx + 0][12]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 1][jx + 1][12]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][12], x);
		y = _mm_load_ps((float*)&x5[ix + 2][jx + 0][12]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 2][jx + 1][12]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][12], x);
		y = _mm_load_ps((float*)&x5[ix + 3][jx + 0][12]);
		x = _mm_max_ps(x, y);
		y = _mm_load_ps((float*)&x5[ix + 3][jx + 1][12]);
		x = _mm_max_ps(x, y);
		_mm_store_ps((float*)&x6[x_out_1][x_out_2][12], x);
		}
	}
	static float x7 alignas(16) [1][1][2] = {0};
	static __m128i cx7 alignas(16) [1][1][2];
	static unsigned char cx_in7 alignas(16) [2][2][16];

	for (int i = 0; i < 2; i++)
	    for (int j = 0; j < 2; j++)
	        for (int k = 0; k < 16; k++)
	            cx_in7[i][j][k] = x6[i][j][k] / 0.008444292470812798f;

	for (int i = 0; i < 1; i += 1)
	{
		for (int j = 0; j < 1; j += 1)
		{
			x7[i][j][0] = 0.10966279357671738f;
			cx7[i][j][0] = _mm_setzero_si128();
			x7[i][j][1] = -0.10966279357671738f;
			cx7[i][j][1] = _mm_setzero_si128();
		}
	}
	for (int ix = -0; ix < 1; ix += 1)
	{
		int x_1, x_out_1;
		x_out_1 = (ix + 0) / 1;
		for (int jx = -0; jx < 1; jx += 1)
		{
			int x_2, x_out_2;
			x_out_2 = (jx + 0) / 1;
			x_1 = ix + 0;
			x_2 = jx + 0;

			qx = _mm_lddqu_si128((__m128i*)&cx_in7[x_1][x_2][0]);
			qw = _mm_set_epi8(-59, -120, 36, 50, 67, 91, 27, -49, 77, 15, 103, 51, -47, -11, 35, 83);
			qx = _mm_maddubs_epi16(qx, qw);
			cx7[x_out_1][x_out_2][0] = _mm_adds_epi16(qx, cx7[x_out_1][x_out_2][0]);

			qx = _mm_lddqu_si128((__m128i*)&cx_in7[x_1][x_2][0]);
			qw = _mm_set_epi8(-25, 123, 58, -2, -12, -87, -7, -38, 4, 20, -36, -65, 7, 11, 64, -24);
			qx = _mm_maddubs_epi16(qx, qw);
			cx7[x_out_1][x_out_2][1] = _mm_adds_epi16(qx, cx7[x_out_1][x_out_2][1]);
			x_2 = jx + 1;

			qx = _mm_lddqu_si128((__m128i*)&cx_in7[x_1][x_2][0]);
			qw = _mm_set_epi8(-92, -92, -6, 113, 85, -2, 0, 48, -29, 74, 42, 72, -67, 49, 81, 115);
			qx = _mm_maddubs_epi16(qx, qw);
			cx7[x_out_1][x_out_2][0] = _mm_adds_epi16(qx, cx7[x_out_1][x_out_2][0]);

			qx = _mm_lddqu_si128((__m128i*)&cx_in7[x_1][x_2][0]);
			qw = _mm_set_epi8(102, 26, -46, -23, -82, 7, -63, -53, 21, -121, -22, -14, 35, -60, -96, -31);
			qx = _mm_maddubs_epi16(qx, qw);
			cx7[x_out_1][x_out_2][1] = _mm_adds_epi16(qx, cx7[x_out_1][x_out_2][1]);
			x_1 = ix + 1;
			x_2 = jx + 0;

			qx = _mm_lddqu_si128((__m128i*)&cx_in7[x_1][x_2][0]);
			qw = _mm_set_epi8(10, 40, -56, -116, 49, -57, -102, -47, 79, -72, 57, -105, -51, 73, -55, -44);
			qx = _mm_maddubs_epi16(qx, qw);
			cx7[x_out_1][x_out_2][0] = _mm_adds_epi16(qx, cx7[x_out_1][x_out_2][0]);

			qx = _mm_lddqu_si128((__m128i*)&cx_in7[x_1][x_2][0]);
			qw = _mm_set_epi8(48, -37, 16, 1, -60, 95, -4, 64, -121, 20, -95, 98, 38, -90, 99, 97);
			qx = _mm_maddubs_epi16(qx, qw);
			cx7[x_out_1][x_out_2][1] = _mm_adds_epi16(qx, cx7[x_out_1][x_out_2][1]);
			x_2 = jx + 1;

			qx = _mm_lddqu_si128((__m128i*)&cx_in7[x_1][x_2][0]);
			qw = _mm_set_epi8(1, 79, 126, 31, 4, 47, -89, 50, 30, -57, -23, 79, -60, 119, -111, 127);
			qx = _mm_maddubs_epi16(qx, qw);
			cx7[x_out_1][x_out_2][0] = _mm_adds_epi16(qx, cx7[x_out_1][x_out_2][0]);

			qx = _mm_lddqu_si128((__m128i*)&cx_in7[x_1][x_2][0]);
			qw = _mm_set_epi8(-2, -65, -97, -112, -37, -8, 100, 21, -45, 89, 18, -44, 87, -46, 20, -23);
			qx = _mm_maddubs_epi16(qx, qw);
			cx7[x_out_1][x_out_2][1] = _mm_adds_epi16(qx, cx7[x_out_1][x_out_2][1]);
		}
	}

	for (int i = 0; i < 1; i++)
	    for (int j = 0; j < 1; j++)
	        for (int k = 0; k < 2; k++){
	            qx = cx7[i][j][k];              
	            lo = _mm_srai_epi32(_mm_unpacklo_epi16(qx, qx), 16);
	            hi = _mm_srai_epi32(_mm_unpackhi_epi16(qx, qx), 16);
	            sum1 = _mm_hadd_epi32(hi, lo);
	            sum2 = _mm_hadd_epi32(sum1, sum1);
		        _mm_store_si128((__m128i*)res, sum2);
	            x7[i][j][k] += (res[0] + res[1]) * 0.004617639413968784f * 0.008444292470812798f;
	        }
	static float x8[1][1][2] = {0};
	static float max8 = 0;
		max8 = x7[0][0][0] > x7[0][0][1] ? x7[0][0][0] : x7[0][0][1];
	x8[0][0][0] = (float)exp(x7[0][0][0] - max8);
	x8[0][0][1] = (float)exp(x7[0][0][1] - max8);
	static float sum8;
	sum8 = x8[0][0][0] + x8[0][0][1];
	x8[0][0][0] /= sum8;
	x8[0][0][1] /= sum8;
	scores[0] = x8[0][0][0];
	scores[1] = x8[0][0][1];
	return;
}

#ifdef CNN_TEST
#include <stdio.h>
#ifdef TIMING
#include <ctime>
#endif
    
int main()
{
    int i, j, k, res, width, height, max_colour;
    unsigned char byte;
    float x[36 * 18 * 1];
    float scores[2];
    FILE *f = fopen("img.pgm", "r");
    fscanf (f, "P5\n%d %d\n%d\n", &width, &height, &max_colour);
    for (j = 0; j < 36; j++)
        for (i = 0; i < 18; i++)
            for (k = 0; k < 1; k++)
            {
                fread(&byte, sizeof(unsigned char), 1, f);
                x[j * 18 * 1 + i * 1 + k] = byte / 255.f;
            }
    fclose(f);
    res = 0;
#ifdef TIMING
    clock_t begin = clock();
	for (i = 0; i < TIMING; i++)  
		cnn_(x, scores);
	 clock_t end = clock();
	double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
	printf("%f %f, %f s ", scores[0], scores[1], elapsed_secs);
#else
    cnn_(x, scores);
#endif
    return scores[1] > scores[0];
}
#endif

#if defined(__clang__)
# pragma clang diagnostic pop
#endif

#if defined(_MSC_VER)
# pragma warning(pop)
#endif
