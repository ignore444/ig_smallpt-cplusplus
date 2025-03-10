#define _CRT_SECURE_NO_WARNINGS

#include <math.h>   // smallpt, a Path Tracer by Kevin Beason, 2008
#include <stdlib.h> 
#include <stdio.h>  

#define M_PI 3.1415926535897932384626433832795
double erand48() 
{
	return (double)rand() / (double)RAND_MAX;
}

struct Vec 
{        
	double x, y, z;                  // position, also color (r,g,b)
	Vec(double x_ = 0, double y_ = 0, double z_ = 0) { x = x_; y = y_; z = z_; }
	Vec operator+(const Vec& b) const	{ return Vec(x + b.x, y + b.y, z + b.z); }
	Vec operator-(const Vec& b) const	{ return Vec(x - b.x, y - b.y, z - b.z); }
	Vec operator*(double b) const		{ return Vec(x * b, y * b, z * b); }
	Vec mult(const Vec& b) const		{ return Vec(x * b.x, y * b.y, z * b.z); }
	Vec& norm()							{ return *this = *this * (1 / sqrt(x * x + y * y + z * z)); }
	double dot(const Vec& b) const		{ return x * b.x + y * b.y + z * b.z; } // cross:
	Vec operator%(Vec& b)				{ return Vec(y * b.z - z * b.y, z * b.x - x * b.z, x * b.y - y * b.x); }
};

struct Ray 
{ 
	Vec origin, dir; 
	Ray(Vec o_, Vec d_) 
		: origin(o_), dir(d_) 
	{
	} 
};

enum Refl_t 
{ 
	DIFF, 
	SPEC, 
	REFR 
};  // material types, used in radiance()

struct Sphere 
{
	double rad;					// radius
	Vec p, emission, color;     // position, emission, color
	Refl_t refl;				// reflection type (DIFFuse, SPECular, REFRactive)
	Sphere(double rad_, Vec p_, Vec e_, Vec c_, Refl_t refl_) 
		: rad(rad_), p(p_), emission(e_), color(c_), refl(refl_) 
	{
	}
	double intersect(const Ray& r) const 
	{ // returns distance, 0 if nohit
		Vec op = p - r.origin; // Solve t^2*d.d + 2*t*(o-p).d + (o-p).(o-p)-R^2 = 0
		double	t, 
				eps = 1e-4, 
				b = op.dot(r.dir), 
				det = b * b - op.dot(op) + rad * rad;

		if (det < 0) 
		{
			return 0; 
		}
		else 
		{
			det = sqrt(det);
		}
		return (t = b - det) > eps ? t : ((t = b + det) > eps ? t : 0);
	}
};

Sphere spheres[] = 
{	//Scene: 
	//		radius, position,					emission,		color,				material
  Sphere(	1e5,	Vec(1e5 + 1,40.8,81.6),		Vec(),			Vec(.75,.25,.25),	DIFF),	//Left
  Sphere(	1e5,	Vec(-1e5 + 99,40.8,81.6),	Vec(),			Vec(.25,.25,.75),	DIFF),	//Rght
  Sphere(	1e5,	Vec(50,40.8, 1e5),			Vec(),			Vec(.75,.75,.75),	DIFF),	//Back
  Sphere(	1e5,	Vec(50,40.8,-1e5 + 170),	Vec(),			Vec(),				DIFF),	//Frnt
  Sphere(	1e5,	Vec(50, 1e5, 81.6),			Vec(),			Vec(.75,.75,.75),	DIFF),	//Botm
  Sphere(	1e5,	Vec(50,-1e5 + 81.6,81.6),	Vec(),			Vec(.75,.75,.75),	DIFF),	//Top

  Sphere(	16.5,	Vec(27,16.5,47),			Vec(),			Vec(1,1,1) * .999,	SPEC),	//Mirr
  Sphere(	16.5,	Vec(73,16.5,78),			Vec(),			Vec(1,1,1) * .999,	REFR),	//Glas
  Sphere(	16.5,	Vec(53,16.5,108),			Vec(),			Vec(1,1,1) * .999,	DIFF),	//Diffuse

  Sphere(	10,		Vec(80,70,110.6),			Vec(5, 0, 0),   Vec(),				DIFF),	//Lite
  Sphere(	600,	Vec(50,681.6 - .27,81.6),	Vec(12,12,12),  Vec(),				DIFF)	//Lite
};

inline double clamp(double x) 
{ 
	return x < 0 ? 0 : x>1 ? 1 : x; 
}
inline int toInt(double x) 
{ 
	return int(pow(clamp(x), 1 / 2.2) * 255 + .5); 
}
inline bool intersect(const Ray& r, double& t, int& id) 
{
	double n = sizeof(spheres) / sizeof(Sphere), d, inf = t = 1e20;
	for (int i = int(n); i--;) 
	{
		if ((d = spheres[i].intersect(r)) && d < t) 
		{ 
			t = d; 
			id = i; 
		}
	}
	return t < inf;
}
Vec radiance(const Ray& ray, int depth) 
{
	double t;										// distance to intersection
	int id = 0;										// id of intersected object
	if (!intersect(ray, t, id)) return Vec();		// if miss, return black
	const Sphere& obj = spheres[id];				// the hit object
	Vec hitPosition = ray.origin + ray.dir * t,		// ray intersection point
		n = (hitPosition - obj.p).norm(),			// sphere normal
		nl = n.dot(ray.dir) < 0 ? n : n * -1,		// oriented surface normal
		f = obj.color;								// object color ( BRDF modulator )
	double p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z; // max refl

	if (depth > 100) 
		return obj.emission; // *** Added to prevent stack overflow

	depth++;

	if (depth > 5) 
	{
		if (erand48() < p) 
		{
			f = f * (1 / p); 
		}
		else 
		{
			return obj.emission; //R.R.
		}
	}

	if (obj.refl == DIFF) 
	{// Ideal DIFFUSE reflection                  
		double	r1 = 2 * M_PI * erand48(),	// angle around
				r2 = erand48(),
				r2s = sqrt(r2);				// distance from center

		// Frame( U,V,W )
		Vec w = nl,														// w = normal
			u = ((fabs(w.x) > .1 ? Vec(0, 1) : Vec(1)) % w).norm(),		// u is perpendicular to w
			v = w % u;													// v is perpendicular to u and w

		Vec dir = (u * cos(r1) * r2s + v * sin(r1) * r2s + w * sqrt(1 - r2)).norm();	// dir is random reflection ray direction
		return obj.emission + f.mult(radiance(Ray(hitPosition, dir), depth));
	}
	else if (obj.refl == SPEC)            
	{// Ideal SPECULAR reflection
		return obj.emission + f.mult(radiance(Ray(hitPosition, ray.dir - n * 2 * n.dot(ray.dir)), depth));
	}

	// Ideal dielectric REFRACTION
	Ray reflRay(hitPosition, ray.dir - n * 2 * n.dot(ray.dir));		
	bool into = n.dot(nl) > 0;								// Ray from outside going in?
	double nc = 1, nt = 1.5, nnt = into ? nc / nt : nt / nc, ddn = ray.dir.dot(nl), cos2t;
	if ((cos2t = 1 - nnt * nnt * (1 - ddn * ddn)) < 0)		// Total internal reflection
	{
		return obj.emission + f.mult(radiance(reflRay, depth));
	}
	Vec tdir = (ray.dir * nnt - n * ((into ? 1 : -1) * (ddn * nnt + sqrt(cos2t)))).norm();
	double	a = nt - nc, 
			b = nt + nc, 
			R0 = a * a / (b * b), 
			c = 1 - (into ? -ddn : tdir.dot(n));
	double	Re = R0 + (1 - R0) * c * c * c * c * c, 
			Tr = 1 - Re, 
			P = .25 + .5 * Re, 
			RP = Re / P, 
			TP = Tr / (1 - P);

	return obj.emission + f.mult(depth > 2 ? (erand48() < P ?	// Russian roulette
		radiance(reflRay, depth) * RP : radiance(Ray(hitPosition, tdir), depth) * TP) :
		radiance(reflRay, depth) * Re + radiance(Ray(hitPosition, tdir), depth) * Tr);
}

int main(int argc, char* argv[]) 
{
	int w = 1024, 
		h = 768, 
		samps = argc == 2 ? atoi(argv[1]) / 4 : 10; // # samples

	Ray cam( Vec(50, 52, 295.6), Vec(0, -0.042612, -1).norm() ); // cam pos, dir
	Vec cx = Vec(w * .5135 / h), 
		cy = (cx % cam.dir).norm() * .5135, 
		r, 
		*color = new Vec[w * h];

#pragma omp parallel for schedule(dynamic, 1) private(r)
	for (int y = 0; y < h; y++) 
	{// Loop over image rows
		fprintf(stderr,"\rRendering (%d spp) %5.2f%%",samps*4,100.*y/(h-1));
		
		for (unsigned short x = 0; x < w; x++)   
		{// Loop cols
			int index = (h - y - 1) * w + x;
			for (int sy = 0; sy < 2; sy++)     
			{// 2x2 subpixel rows
				for (int sx = 0; sx < 2; sx++) 
				{// 2x2 subpixel cols
					r = Vec();
					for (int s = 0; s < samps; s++) 
					{
						double	r1 = 2 * erand48(), 
								dx = r1 < 1 ? sqrt(r1) - 1 : 1 - sqrt(2 - r1);
						double	r2 = 2 * erand48(), 
								dy = r2 < 1 ? sqrt(r2) - 1 : 1 - sqrt(2 - r2);
						Vec d = cx * (((sx + .5 + dx) / 2 + x) / w - .5) + cy * (((sy + .5 + dy) / 2 + y) / h - .5) + cam.dir;
						r = r + radiance(Ray(cam.origin + d * 140, d.norm()), 0) * (1. / samps);
					} // Camera rays are pushed ^^^^^ forward to start in interior
					color[index] = color[index] + Vec(clamp(r.x), clamp(r.y), clamp(r.z)) * .25;
				}
			}
		}
	}

	FILE* f = fopen("image.ppm", "w");
	fprintf(f, "P3\n%d %d\n%d\n", w, h, 255);
	for (int i = 0; i < w * h; i++)
	{
		fprintf(f, "%d %d %d ", toInt(color[i].x), toInt(color[i].y), toInt(color[i].z));
	}
}