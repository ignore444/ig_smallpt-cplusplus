﻿// Copyright (c) 2014 hole
// This software is released under the MIT License (http://kagamin.net/hole/license.txt).
// A part of this software is based on smallpt (http://www.kevinbeason.com/smallpt/) and
// released under the MIT License (http://kagamin.net/hole/smallpt-license.txt).
#define _CRT_SECURE_NO_WARNINGS
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>

const double PI = 3.14159265358979323846;
const double INF = 1e20;
const double EPS = 1e-6;
const double MaxDepth = 5;

// *** その他の関数 ***
inline double rand01() { return (double)rand() / RAND_MAX; }

// *** データ構造 ***
struct Vec {
	double x, y, z;
	Vec(const double x_ = 0, const double y_ = 0, const double z_ = 0) : x(x_), y(y_), z(z_) {}
	inline Vec operator+(const Vec& b) const { return Vec(x + b.x, y + b.y, z + b.z); }
	inline Vec operator-(const Vec& b) const { return Vec(x - b.x, y - b.y, z - b.z); }
	inline Vec operator*(const double b) const { return Vec(x * b, y * b, z * b); }
	inline Vec operator/(const double b) const { return Vec(x / b, y / b, z / b); }
	inline const double LengthSquared() const { return x * x + y * y + z * z; }
	inline const double Length() const { return sqrt(LengthSquared()); }
};
inline Vec operator*(double f, const Vec& v) { return v * f; }
inline Vec Normalize(const Vec& v) { return v / v.Length(); }
// 要素ごとの積をとる
inline const Vec Multiply(const Vec& v1, const Vec& v2) {
	return Vec(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}
inline const double Dot(const Vec& v1, const Vec& v2) {
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}
inline const Vec Cross(const Vec& v1, const Vec& v2) {
	return Vec((v1.y * v2.z) - (v1.z * v2.y), (v1.z * v2.x) - (v1.x * v2.z), (v1.x * v2.y) - (v1.y * v2.x));
}
typedef Vec Color;
const Color BackgroundColor(0.0, 0.0, 0.0);

struct Ray {
	Vec org, dir;
	Ray(const Vec org_, const Vec& dir_) : org(org_), dir(dir_) {}
};

enum ReflectionType {
	DIFFUSE,    // 完全拡散面。いわゆるLambertian面。
	SPECULAR,   // 理想的な鏡面。
	REFRACTION, // 理想的なガラス的物質。

	GLOSSY, // Wardのグロシー
};

struct Sphere {
	double radius;
	Vec position;
	Color emission, color;
	ReflectionType ref_type;

	Sphere(const double radius_, const Vec& position_, const Color& emission_, const Color& color_, const ReflectionType ref_type_) :
		radius(radius_), position(position_), emission(emission_), color(color_), ref_type(ref_type_) {}
	// 入力のrayに対する交差点までの距離を返す。交差しなかったら0を返す。
	const double intersect(const Ray& ray) {
		Vec o_p = position - ray.org;
		const double b = Dot(o_p, ray.dir), det = b * b - Dot(o_p, o_p) + radius * radius;
		if (det >= 0.0) {
			const double sqrt_det = sqrt(det);
			const double t1 = b - sqrt_det, t2 = b + sqrt_det;
			if (t1 > EPS)		return t1;
			else if (t2 > EPS)	return t2;
		}
		return 0.0;
	}
};

// *** レンダリングするシーンデータ ****
// from small ppt
Sphere spheres[] = {
	Sphere(1e5, Vec(1e5 + 1,40.8,81.6), Color(), Color(0.75, 0.25, 0.25),DIFFUSE),// 左
	Sphere(1e5, Vec(-1e5 + 99,40.8,81.6),Color(), Color(0.25, 0.25, 0.75),DIFFUSE),// 右
	Sphere(1e5, Vec(50,40.8, 1e5),     Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 奥
	Sphere(1e5, Vec(50,40.8,-1e5 + 170), Color(), Color(), DIFFUSE),// 手前
	Sphere(1e5, Vec(50, 1e5, 81.6),    Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 床
	Sphere(1e5, Vec(50,-1e5 + 81.6,81.6),Color(), Color(0.75, 0.75, 0.75),DIFFUSE),// 天井
	Sphere(16.5,Vec(50,16.5,47),       Color(), Color(1,1,1) * .99, SPECULAR),// 鏡
	Sphere(16.5,Vec(20,16.5,70),       Color(), Color(1,1,1) * .99, GLOSSY), // グロッシーな物体
	Sphere(16.5,Vec(80,16.5,78),       Color(), Color(1,1,1) * .99, GLOSSY), // グロッシーな物体
	Sphere(5.0, Vec(50.0, 75.0, 81.6),Color(12,12,12), Color(), DIFFUSE), //照明
};

// *** レンダリング用関数 ***
// シーンとの交差判定関数
inline bool intersect_scene(const Ray& ray, double* t, int* id) {
	const double n = sizeof(spheres) / sizeof(Sphere);
	*t = INF;
	*id = -1;
	for (int i = 0; i < int(n); i++) {
		double d = spheres[i].intersect(ray);
		if (d > 0.0 && d < *t) {
			*t = d;
			*id = i;
		}
	}
	return *t < INF;
}

// ray方向からの放射輝度を求める
Color radiance(const Ray& ray, const int depth) {
	double t; // レイからシーンの交差位置までの距離
	int id;   // 交差したシーン内オブジェクトのID
	if (!intersect_scene(ray, &t, &id))
		return BackgroundColor;

	const Sphere& obj = spheres[id];
	const Vec hitpoint = ray.org + t * ray.dir; // 交差位置
	const Vec normal = Normalize(hitpoint - obj.position); // 交差位置の法線
	const Vec orienting_normal = Dot(normal, ray.dir) < 0.0 ? normal : (-1.0 * normal); // 交差位置の法線（物体からのレイの入出を考慮）
	// 色の反射率最大のものを得る。ロシアンルーレットで使う。
	// ロシアンルーレットの閾値は任意だが色の反射率等を使うとより良い。
	double russian_roulette_probability = std::max(obj.color.x, std::max(obj.color.y, obj.color.z));
	// 一定以上レイを追跡したらロシアンルーレットを実行し追跡を打ち切るかどうかを判断する
	if (depth > MaxDepth) {
		if (rand01() >= russian_roulette_probability)
			return obj.emission;
	}
	else
		russian_roulette_probability = 1.0; // ロシアンルーレット実行しなかった

	switch (obj.ref_type) {

		// WardのモデルによるグロシーなBRDF
	case GLOSSY: {

		// 各種パラメータ。
		// 決め方がださいね。
		const double lo_s = 0.75;
		double alpha_x, alpha_y;
		if (id == 7) {
			alpha_x = 0.75;
			alpha_y = 0.75;
		}
		else {
			alpha_x = 0.25;
			alpha_y = 0.25;
		}

		const Vec in = -1.0 * ray.dir;
		Vec halfv;
		Vec dir;

		// orienting_normalの方向を基準とした正規直交基底(w, u, v)を作る。この基底に対する半球内で次のレイを飛ばす。
		Vec w, u, v;
		w = orienting_normal;
		if (fabs(w.x) > 0.1)
			u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
		else
			u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));
		v = Cross(w, u);

		// 重点サンプリングする。ただし、生成されるハーフベクトルしだいでは
		// 反射方向が半球外に出てしまうのでそういう場合は棄却する。
		do {
			const double u1 = rand01();
			const double u2 = rand01();
			double phi = atan(alpha_y / alpha_x * tan(2.0 * PI * u2));
			if (0.25 <= u2 && u2 <= 0.75)
				phi += PI;
			else if (0.75 < u2)
				phi += 2.0 * PI;
			const double theta = atan(sqrt(-log(u1) / (pow(cos(phi), 2) / pow(alpha_x, 2) + pow(sin(phi), 2) / pow(alpha_y, 2))));

			halfv = Normalize((u * cos(phi) * sin(theta) + v * sin(phi) * sin(theta) + w * cos(theta)));
			dir = 2.0 * Dot(in, halfv) * halfv - in;
		} while (Dot(orienting_normal, dir) < 0.0);

		// 重み。brdf * cosθ / pdf(ω)をまとめたものになっている。
		const double weight = lo_s * Dot(halfv, in) * pow(Dot(halfv, orienting_normal), 3) *
			sqrt(Dot(dir, orienting_normal) / Dot(in, orienting_normal));

		return obj.emission + Multiply(obj.color, radiance(Ray(hitpoint, dir), depth + 1))
			* weight
			/ russian_roulette_probability;
	} break;


	case DIFFUSE: {
		// orienting_normalの方向を基準とした正規直交基底(w, u, v)を作る。この基底に対する半球内で次のレイを飛ばす。
		Vec w, u, v;
		w = orienting_normal;
		if (fabs(w.x) > 0.1)
			u = Normalize(Cross(Vec(0.0, 1.0, 0.0), w));
		else
			u = Normalize(Cross(Vec(1.0, 0.0, 0.0), w));
		v = Cross(w, u);
		// コサイン項を使った重点的サンプリング
		const double r1 = 2 * PI * rand01();
		const double r2 = rand01(), r2s = sqrt(r2);
		Vec dir = Normalize((u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1.0 - r2)));

		// レンダリング方程式に従えば Le + Li(ray) * BRDF * cosθ / pdf(ray) になる。
		// ただし、上でコサイン項による重点的サンプリングをしたためpdf(ray) = cosθ/πになり、
		// Diffuse面のBRDF = 1/πなので、これらを代入すると Le + Li(ray) となる。
		// これにロシアンルーレットの確率を除算したものが最終的な計算式になる。
		return obj.emission + Multiply(obj.color, radiance(Ray(hitpoint, dir), depth + 1)) / russian_roulette_probability;
	} break;
	case SPECULAR: {
		// 完全鏡面なのでレイの反射方向は決定的。
		// ロシアンルーレットの確率で除算するのは上と同じ。
		return obj.emission + Multiply(obj.color,
			radiance(Ray(hitpoint, ray.dir - normal * 2.0 * Dot(normal, ray.dir)), depth + 1)) / russian_roulette_probability;
	} break;
	case REFRACTION: {
		Ray reflection_ray = Ray(hitpoint, ray.dir - normal * 2.0 * Dot(normal, ray.dir));
		bool into = Dot(normal, orienting_normal) > 0.0; // レイがオブジェクトから出るのか、入るのか

		// Snellの法則
		const double nc = 1.0; // 真空の屈折率
		const double nt = 1.5; // オブジェクトの屈折率
		const double nnt = into ? nc / nt : nt / nc;
		const double ddn = Dot(ray.dir, orienting_normal);
		const double cos2t = 1.0 - nnt * nnt * (1.0 - ddn * ddn);

		if (cos2t < 0.0) { // 全反射した
			return obj.emission + Multiply(obj.color, (radiance(reflection_ray, depth + 1))) / russian_roulette_probability;
		}
		// 屈折していく方向
		Vec tdir = Normalize(ray.dir * nnt - normal * (into ? 1.0 : -1.0) * (ddn * nnt + sqrt(cos2t)));

		// SchlickによるFresnelの反射係数の近似
		const double a = nt - nc, b = nt + nc;
		const double R0 = (a * a) / (b * b);
		const double c = 1.0 - (into ? -ddn : Dot(tdir, normal));
		const double Re = R0 + (1.0 - R0) * pow(c, 5.0);
		const double Tr = 1.0 - Re; // 屈折光の運ぶ光の量
		const double probability = 0.25 + 0.5 * Re;

		// 一定以上レイを追跡したら屈折と反射のどちらか一方を追跡する。（さもないと指数的にレイが増える）
		// ロシアンルーレットで決定する。
		if (depth > 2) {
			if (rand01() < probability) { // 反射
				return obj.emission +
					Multiply(obj.color, radiance(reflection_ray, depth + 1) * Re)
					/ probability
					/ russian_roulette_probability;
			}
			else { // 屈折
				return obj.emission +
					Multiply(obj.color, radiance(Ray(hitpoint, tdir), depth + 1) * Tr)
					/ (1.0 - probability)
					/ russian_roulette_probability;
			}
		}
		else { // 屈折と反射の両方を追跡
			return obj.emission +
				Multiply(obj.color, radiance(reflection_ray, depth + 1) * Re
					+ radiance(Ray(hitpoint, tdir), depth + 1) * Tr) / russian_roulette_probability;
		}
	} break;
	}
}


// *** .hdrフォーマットで出力するための関数 ***
struct HDRPixel {
	unsigned char r, g, b, e;
	HDRPixel(const unsigned char r_ = 0, const unsigned char g_ = 0, const unsigned char b_ = 0, const unsigned char e_ = 0) :
		r(r_), g(g_), b(b_), e(e_) {};
	unsigned char get(int idx) {
		switch (idx) {
		case 0: return r;
		case 1: return g;
		case 2: return b;
		case 3: return e;
		} return 0;
	}

};

// doubleのRGB要素を.hdrフォーマット用に変換
HDRPixel get_hdr_pixel(const Color& color) {
	double d = std::max(color.x, std::max(color.y, color.z));
	if (d <= 1e-32)
		return HDRPixel();
	int e;
	double m = frexp(d, &e); // d = m * 2^e
	d = m * 256.0 / d;
	return HDRPixel(color.x * d, color.y * d, color.z * d, e + 128);
}

// 書き出し用関数
void save_hdr_file(const std::string& filename, const Color* image, const int width, const int height) {
	FILE* fp = fopen(filename.c_str(), "wb");
	if (fp == NULL) {
		std::cerr << "Error: " << filename << std::endl;
		return;
	}
	// .hdrフォーマットに従ってデータを書きだす
	// ヘッダ
	unsigned char ret = 0x0a;
	fprintf(fp, "#?RADIANCE%c", (unsigned char)ret);
	fprintf(fp, "# Made with 100%% pure HDR Shop%c", ret);
	fprintf(fp, "FORMAT=32-bit_rle_rgbe%c", ret);
	fprintf(fp, "EXPOSURE=1.0000000000000%c%c", ret, ret);

	// 輝度値書き出し
	fprintf(fp, "-Y %d +X %d%c", height, width, ret);
	for (int i = height - 1; i >= 0; i--) {
		std::vector<HDRPixel> line;
		for (int j = 0; j < width; j++) {
			HDRPixel p = get_hdr_pixel(image[j + i * width]);
			line.push_back(p);
		}
		fprintf(fp, "%c%c", 0x02, 0x02);
		fprintf(fp, "%c%c", (width >> 8) & 0xFF, width & 0xFF);
		for (int i = 0; i < 4; i++) {
			for (int cursor = 0; cursor < width;) {
				const int cursor_move = std::min(127, width - cursor);
				fprintf(fp, "%c", cursor_move);
				for (int j = cursor; j < cursor + cursor_move; j++)
					fprintf(fp, "%c", line[j].get(i));
				cursor += cursor_move;
			}
		}
	}

	fclose(fp);
}

int main(int argc, char** argv) {
	int width = 640;
	int height = 480;
	int samples = 16;

	// カメラ位置
	Ray camera(Vec(50.0, 52.0, 295.6), Normalize(Vec(0.0, -0.042612, -1.0)));
	// シーン内でのスクリーンのx,y方向のベクトル
	Vec cx = Vec(width * 0.5135 / height);
	Vec cy = Normalize(Cross(cx, camera.dir)) * 0.5135;
	Color* image = new Color[width * height];

#pragma omp parallel for schedule(dynamic, 1)
	for (int y = 0; y < height; y++) {
		int used_sample = 0;
		for (int x = 0; x < width; x++) {
			int image_index = y * width + x;
			image[image_index] = Color();

			// 2x2のサブピクセルサンプリング
			for (int sy = 0; sy < 2; sy++) {
				for (int sx = 0; sx < 2; sx++) {
					Color accumulated_radiance = Color();

					// 一つのサブピクセルあたりsamples回サンプリングする
					for (int s = 0; s < samples; s++) {
						// テントフィルターによってサンプリング
						// ピクセル範囲で一様にサンプリングするのではなく、ピクセル中央付近にサンプルがたくさん集まるように偏りを生じさせる
						const double r1 = 2.0 * rand01(), dx = r1 < 1.0 ? sqrt(r1) - 1.0 : 1.0 - sqrt(2.0 - r1);
						const double r2 = 2.0 * rand01(), dy = r2 < 1.0 ? sqrt(r2) - 1.0 : 1.0 - sqrt(2.0 - r2);
						Vec dir = cx * (((sx + 0.5 + dx) / 2.0 + x) / width - 0.5) +
							cy * (((sy + 0.5 + dy) / 2.0 + y) / height - 0.5) + camera.dir;
						accumulated_radiance = accumulated_radiance +
							radiance(Ray(camera.org + dir * 130.0, Normalize(dir)), 0) / samples;
					}

					image[image_index] = image[image_index] + accumulated_radiance;
				}
			}
		}
		std::cerr << "Rendering (average: " << ((double)used_sample / width) << " spp) " << (100.0 * y / (height - 1)) << "%" << std::endl;
	}

	// .hdrフォーマットで出力
	save_hdr_file(std::string("image.hdr"), image, width, height);
}